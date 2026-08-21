"""Safe tar extraction into the staging area.

Extraction is the bridge between a downloaded ``task_XXX.tar.gz`` and the
LeRobot directory layout the converter expects.  Safety and atomicity:

* every member name is validated (no absolute paths, no ``..``, no symlink /
  hardlink / device members) before anything is written;
* free disk space is checked against the uncompressed member sizes up front;
* files land in ``extracted/.<tar_id>.tmp`` and the directory is published to
  ``extracted/<tar_id>`` with a single rename — a crashed run never exposes a
  half-extracted dataset to the converter;
* the layout is validated afterwards (exactly one dataset directory named
  after the tar, with meta/info.json and data/chunk-*/episode_*.parquet), so
  conversion failures surface here with clear messages.

Error taxonomy for the pipeline:

* ``ExtractionError`` — permanent (layout/name violations): retrying cannot heal it;
* ``ExtractionCorruptError`` — the gzip stream failed CRC: the tar file must be
  re-downloaded, then extraction retried.
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path


class ExtractionError(ValueError):
    """Permanent extraction failure (bad member names, wrong layout)."""


class ExtractionCorruptError(RuntimeError):
    """The downloaded tar is corrupt (gzip CRC failed); re-download and retry."""


#: Minimum free-space headroom to keep on the staging filesystem.
FREE_SPACE_MARGIN = 1024 ** 3  # 1 GiB


class StagingPaths:
    """On-disk layout of the staging area (owned by the streaming pipeline).

    ``tars/<tar_id>.tar.gz`` — downloaded objects (preallocated, so apparent
    size equals the object size from the first byte).
    ``parts/<tar_id>.part.N`` — per-part completion markers.
    ``extracted/<tar_id>/<tar_id>/...`` — the extracted LeRobot dataset dirs.
    """

    def __init__(self, root: Path):
        self.root = root
        self.tars = root / "tars"
        self.parts = root / "parts"
        self.extracted = root / "extracted"

    def ensure_dirs(self) -> None:
        for directory in (self.tars, self.parts, self.extracted):
            directory.mkdir(parents=True, exist_ok=True)

    def tar_path(self, tar_id: str) -> Path:
        return self.tars / f"{tar_id}.tar.gz"

    def extracted_root(self, tar_id: str) -> Path:
        return self.extracted / tar_id

    def parts_done(self, tar_id: str) -> list[Path]:
        """Part markers present on disk (any object-size match)."""
        return sorted(self.parts.glob(f"{tar_id}.part.*")) if self.parts.is_dir() else []


@dataclass
class ExtractedLayout:
    dataset_dir: Path  # staging/extracted/<tar_id>/<tar_id>
    episode_count: int  # data/chunk-*/episode_*.parquet files
    extracted_bytes: int  # total size of the extracted files


def _member_error(member: tarfile.TarInfo) -> str | None:
    name = member.name
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return f"absolute/volume path {name!r}"
    if name.startswith("\\") or "\\" in name:
        return f"backslash in member name {name!r}"
    for component in name.split("/"):
        if component == "..":
            return f"parent-directory component in member name {name!r}"
    if member.issym() or member.islnk():
        return f"link member {name!r}"
    if not (member.isfile() or member.isdir()):
        return f"special member {name!r}"
    return None


def extract_tar(tar_path: Path, tar_id: str, staging: StagingPaths) -> ExtractedLayout:
    """Extract ``tar_path`` into the staging area and validate the layout.

    Raises ``ExtractionError`` for permanent problems and
    ``ExtractionCorruptError`` when the tar itself is corrupt.
    """
    paths = staging
    final_dir = paths.extracted_root(tar_id)
    temp_dir = paths.extracted / f".{tar_id}.tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # Any read of the gzip stream (member headers or file bodies) can hit
        # a CRC failure / truncation, so the whole open covers both.
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                for member in members:
                    error = _member_error(member)
                    if error is not None:
                        raise ExtractionError(f"{tar_path}: unsafe member {error}")
                total_bytes = sum(member.size for member in members if member.isfile())
                free = shutil.disk_usage(paths.extracted).free
                if total_bytes + FREE_SPACE_MARGIN > free:
                    raise ExtractionError(
                        f"insufficient disk space on {paths.extracted}: need "
                        f"~{total_bytes / 1024 ** 3:.1f} GiB, have {free / 1024 ** 3:.1f} GiB free"
                    )
                tar.extractall(path=temp_dir)
        except ExtractionError:
            raise
        except (tarfile.ReadError, zlib.error, EOFError, OSError) as exc:
            raise ExtractionCorruptError(
                f"{tar_path}: gzip stream corrupt or truncated: {exc}"
            ) from exc

        # Layout: exactly one top-level dataset directory named after the tar;
        # stray top-level files (e.g. "[].conversion_completed.json") are ignored.
        entries = [entry for entry in temp_dir.iterdir() if entry.is_dir()]
        if len(entries) != 1:
            raise ExtractionError(
                f"{tar_id}: expected exactly one top-level directory, found "
                f"{[entry.name for entry in entries]}"
            )
        dataset_dir = entries[0]
        if dataset_dir.name != tar_id:
            raise ExtractionError(
                f"{tar_id}: expected the dataset directory to be named {tar_id!r}, "
                f"found {dataset_dir.name!r}"
            )
        if not (dataset_dir / "meta" / "info.json").is_file():
            raise ExtractionError(f"{tar_id}: missing meta/info.json")
        parquets = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
        if not parquets:
            raise ExtractionError(f"{tar_id}: no data/chunk-*/episode_*.parquet files found")
        if not (dataset_dir / "meta" / "tasks.jsonl").is_file():
            print(f"WARNING: {tar_id}: no meta/tasks.jsonl; the converter will fall back "
                  f"to the directory name as the task instruction")

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temp_dir, final_dir)

        extracted_bytes = sum(
            file.stat().st_size for file in final_dir.rglob("*") if file.is_file()
        )
        return ExtractedLayout(
            dataset_dir=final_dir / tar_id,
            episode_count=len(parquets),
            extracted_bytes=extracted_bytes,
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def remove_extracted_dir(staging: StagingPaths, tar_id: str) -> None:
    target = staging.extracted_root(tar_id)
    if target.exists():
        shutil.rmtree(target)
    leftover = staging.extracted / f".{tar_id}.tmp"
    if leftover.exists():
        shutil.rmtree(leftover)


def remove_download_artifacts(staging: StagingPaths, tar_id: str) -> None:
    """Delete the tar file and its part markers (e.g. before a re-download)."""
    tar_path = staging.tar_path(tar_id)
    if tar_path.exists():
        tar_path.unlink()
    for marker in staging.parts.glob(f"{tar_id}.part.*"):
        marker.unlink()
