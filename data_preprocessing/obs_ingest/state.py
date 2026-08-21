"""Persistent per-tar state for the streaming pipeline.

One JSON file (``staging/obs_pipeline_state.json``) holds the state machine
for every tar plus the run-level view spec.  Writes are atomic (tmp + rename)
and serialized through a single lock, so a crash at any point leaves either
the previous or the next state — never a partial file.

State machine::

    pending ──download──▶ downloaded ──extract──▶ extracted ──convert+commit──▶ converted
        │                                          │
        └───────────▶ failed_permanent ◀───────────┘   (retried only with --force-retry-failed)
        └───────────▶ skipped                (global --max-episodes cap reached)

Transient failures (network, gzip CRC) do not get a state of their own: the
worker backs off and requeues the tar, and ``attempts`` records how often it
has been retried.  Only permanent failures are terminal.

Commit invariant: ``mark_converted`` writes the tar's norm-stats snapshot and
the ``converted`` status in ONE atomic save — a tar marked converted has
always been added to the dataset (``DatasetWriter.add_task`` is idempotent,
so a re-run after a crash in the window before the mark converges).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

STATUS_PENDING = "pending"
STATUS_DOWNLOADED = "downloaded"
STATUS_EXTRACTED = "extracted"
STATUS_CONVERTED = "converted"
STATUS_FAILED_PERMANENT = "failed_permanent"
STATUS_SKIPPED = "skipped"

#: Statuses that never get scheduled again (not even with retries).
TERMINAL_STATUSES = (STATUS_CONVERTED, STATUS_FAILED_PERMANENT, STATUS_SKIPPED)


@dataclass
class TarState:
    key: str  # OBS object key, e.g. "Agibot_Beta_Lerobot_Amap/task_327.tar.gz"
    tar_id: str  # basename without ".tar.gz", e.g. "task_327"
    size: int  # object size in bytes from the listing
    status: str = STATUS_PENDING
    attempts: int = 0
    error: str = ""
    episode_count: Optional[int] = None  # Parquet files counted at extract time
    norm_stats: Optional[dict] = None  # NormAccumulator.stats_snapshot() at commit
    info_json: Optional[dict] = None  # probed meta/info.json (view-spec resolution)
    bytes_downloaded: int = 0  # cumulative, for progress/throughput reporting


class PipelineState:
    """Loads, mutates, and atomically persists the per-tar state machine."""

    def __init__(self, path: Path):
        self.path = path
        # RLock: mark_* helpers hold the lock and then call save(), which
        # re-acquires it for the atomic write.
        self._lock = threading.RLock()
        self.tars: dict[str, TarState] = {}
        self.view_spec: Optional[dict] = None
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            print(
                f"WARNING: could not read {self.path}; starting with empty state "
                "(previously converted tars will be re-checked against the manifest)"
            )
            return
        for entry in data.get("tars", []):
            tar = TarState(**entry)
            self.tars[tar.tar_id] = tar
        self.view_spec = data.get("view_spec")

    def save(self) -> None:
        """Atomically persist the whole state (tmp + rename)."""
        data = {
            "view_spec": self.view_spec,
            "tars": [
                {
                    "key": tar.key,
                    "tar_id": tar.tar_id,
                    "size": tar.size,
                    "status": tar.status,
                    "attempts": tar.attempts,
                    "error": tar.error,
                    "episode_count": tar.episode_count,
                    "norm_stats": tar.norm_stats,
                    "info_json": tar.info_json,
                    "bytes_downloaded": tar.bytes_downloaded,
                }
                for tar in sorted(self.tars.values(), key=lambda tar: tar.tar_id)
            ],
        }
        with self._lock:
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)

    def add_tar(self, key: str, size: int) -> TarState:
        tar_id = os.path.basename(key)
        if tar_id.endswith(".tar.gz"):
            tar_id = tar_id[: -len(".tar.gz")]
        tar = self.tars.get(tar_id)
        if tar is None:
            tar = TarState(key=key, tar_id=tar_id, size=size)
            self.tars[tar_id] = tar
        else:
            # Listing refreshes may update the object size.
            tar.size = size
        return tar

    def update(self, tar: TarState, **fields) -> None:
        """Set fields on a TarState and persist."""
        with self._lock:
            for name, value in fields.items():
                setattr(tar, name, value)
            self.save()

    def mark_downloaded(self, tar: TarState, bytes_downloaded: int) -> None:
        with self._lock:
            tar.status = STATUS_DOWNLOADED
            tar.bytes_downloaded += bytes_downloaded
            self.save()

    def mark_extracted(self, tar: TarState, episode_count: int) -> None:
        with self._lock:
            tar.status = STATUS_EXTRACTED
            tar.episode_count = episode_count
            self.save()

    def mark_converted(self, tar: TarState, norm_stats: dict) -> None:
        """Commit: status + norm snapshot in one atomic save (see module docstring)."""
        with self._lock:
            tar.status = STATUS_CONVERTED
            tar.norm_stats = norm_stats
            tar.error = ""
            self.save()

    def mark_failed(self, tar: TarState, error: str, attempts: int) -> None:
        with self._lock:
            tar.status = STATUS_FAILED_PERMANENT
            tar.error = error
            tar.attempts = attempts
            self.save()

    def mark_skipped(self, tar: TarState, error: str) -> None:
        with self._lock:
            tar.status = STATUS_SKIPPED
            tar.error = error
            self.save()

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for tar in self.tars.values():
                counts[tar.status] = counts.get(tar.status, 0) + 1
            return counts

    def print_status(self) -> None:
        counts = self.status_counts()
        print(f"pipeline state: {self.path}")
        if not self.tars:
            print("  (empty)")
            return
        for status in (STATUS_PENDING, STATUS_DOWNLOADED, STATUS_EXTRACTED,
                       STATUS_CONVERTED, STATUS_FAILED_PERMANENT, STATUS_SKIPPED):
            if counts.get(status):
                print(f"  {status:<16} {counts[status]}")
        failed = [tar for tar in self.tars.values() if tar.status == STATUS_FAILED_PERMANENT]
        for tar in sorted(failed, key=lambda tar: tar.tar_id):
            print(f"  FAIL {tar.tar_id}: {tar.error[:120]}")
        if self.view_spec:
            print(f"  view_spec: {json.dumps(self.view_spec, ensure_ascii=False)}")
