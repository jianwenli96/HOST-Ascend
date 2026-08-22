"""Pipeline driver: bounded queues, worker threads, backpressure, progress.

Three stages — download (parallel ranged parts), extract (safe tar
decompression), convert (the shared converter) — are connected by bounded
queues so all of them overlap: while tar N converts, N+1 extracts and N+2
downloads.  A ``StagingBudget`` caps the staging disk footprint; worker
threads block on reservations instead of filling the disk.  Every tar is
"active" from scheduling until it reaches a terminal state; the run loop
finishes when the active count reaches zero.

Failure taxonomy per stage:

* download: ``OBSFatalError`` (4xx / ignored Range header) -> permanent;
  anything else -> transient retry (part markers make retries resume);
* extract: ``ExtractionError`` (layout) -> permanent;
  ``ExtractionCorruptError`` (gzip CRC) -> delete the tar, re-download, retry;
* convert: ``ValueError`` / ``json.JSONDecodeError`` (bad data) -> permanent;
  anything else -> transient retry (the converter's phase A is idempotent).

Transient retries are bounded by ``--retries`` and the attempt counter is
persisted, so a restart cannot retry a doomed tar forever.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import extract as extract_mod
from . import obsio
from .convert import convert_one_tar
from .extract import StagingPaths
from .state import (
    STATUS_CONVERTED,
    STATUS_DOWNLOADED,
    STATUS_EXTRACTED,
    STATUS_FAILED_PERMANENT,
    STATUS_PENDING,
    STATUS_SKIPPED,
    PipelineState,
    TarState,
)


def format_gib(value: float) -> str:
    return f"{value / 1024 ** 3:.1f} GiB"


class StagingBudget:
    """Bounded staging usage: reserve bytes before downloading a tar, release
    after cleanup.  A single tar larger than the cap is allowed to overshoot
    (otherwise it could never be processed)."""

    def __init__(self, cap_bytes: int):
        self.cap_bytes = cap_bytes
        self._cond = threading.Condition()
        self._reserved: dict[str, int] = {}

    def reserve(self, tar_id: str, amount: int) -> None:
        with self._cond:
            while amount <= self.cap_bytes:
                if self.used() + amount <= self.cap_bytes:
                    break
                self._cond.wait()
            self._reserved[tar_id] = self._reserved.get(tar_id, 0) + amount

    def release(self, tar_id: str) -> None:
        with self._cond:
            self._reserved.pop(tar_id, None)
            self._cond.notify_all()

    def used(self) -> int:
        return sum(self._reserved.values())


class Stats:
    """Run-local counters for the progress monitor (lock-protected)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.bytes_downloaded = 0
        self.tars_converted = 0
        self.stage_counts: dict[str, int] = {}


class StreamingPipeline:
    def __init__(
        self,
        args,
        infos: dict,
        bucket: str,
        keys: list[tuple[str, int]],
        state: PipelineState,
        staging: StagingPaths,
        output_dir: Path,
        writer,
        accumulator,
        episode_cap: Optional[int],
        committed_episodes: int,
    ):
        self.args = args
        self.infos = infos
        self.bucket = bucket
        self.keys = keys
        self.state = state
        self.staging = staging
        self.output_dir = output_dir
        self.writer = writer
        self.accumulator = accumulator
        self.episode_cap = episode_cap
        self._episodes_scheduled = committed_episodes
        self._cap_lock = threading.Lock()

        self.download_workers = args.download_workers
        self.extract_workers = args.extract_workers
        self.convert_workers = args.convert_workers
        self.parts_per_tar = args.parts_per_tar
        self.retries = args.retries
        self.timeout = args.timeout
        self.keep_extracted = args.keep_extracted
        self.force_retry_failed = args.force_retry_failed
        self.limit = args.limit
        self.max_staging_bytes = int(args.max_staging_gb * 1024 ** 3)

        if self.convert_workers > 1:
            effective = max(1, args.workers // self.convert_workers)
            if effective != args.workers:
                print(
                    f"NOTE: {self.convert_workers} convert workers -> {effective} "
                    f"ffmpeg encodes per converter (--workers {args.workers})"
                )
                args.workers = effective

        self._lock = threading.Lock()
        self._active = 0

    # ------------------------------------------------------------------ setup

    def _build_schedule(self) -> list[TarState]:
        for key, size in self.keys:
            self.state.add_tar(key, size)
        for tar in self.state.tars.values():
            if tar.status == STATUS_FAILED_PERMANENT and self.force_retry_failed:
                tar.status = STATUS_PENDING
                tar.attempts = 0
                tar.error = ""
        self.state.save()

        def priority(tar: TarState) -> int:
            if tar.status == STATUS_EXTRACTED:
                return 3  # finish in-flight conversions first
            if tar.status == STATUS_DOWNLOADED or self.staging.parts_done(tar.tar_id):
                return 2
            return 1

        todo = [
            tar
            for tar in self.state.tars.values()
            if tar.status not in (STATUS_CONVERTED, STATUS_SKIPPED)
            and not (tar.status == STATUS_FAILED_PERMANENT and not self.force_retry_failed)
        ]
        todo.sort(key=lambda tar: (priority(tar), tar.size))  # small tars first
        if self.limit:
            todo = todo[: self.limit]
        return todo

    # -------------------------------------------------------------- run loop

    def describe_schedule(self) -> list[TarState]:
        """Build the schedule and print a summary (--dry-run)."""
        scheduled = self._build_schedule()
        total = sum(tar.size for tar in scheduled)
        print(f"would schedule {len(scheduled)} tars ({format_gib(total)}):")
        for tar in scheduled[:10]:
            print(f"  {tar.tar_id:<12} {format_gib(tar.size):>10}  {tar.status}")
        if len(scheduled) > 10:
            print(f"  ... and {len(scheduled) - 10} more")
        return scheduled

    def run(self) -> bool:
        """Drive the pipeline to completion; returns True when nothing failed."""
        scheduled = self._build_schedule()
        if not scheduled:
            print("nothing to do")
            return True
        scheduled_ids = [tar.tar_id for tar in scheduled]
        total_size = sum(tar.size for tar in scheduled)
        print(
            f"scheduling {len(scheduled)} tars ({format_gib(total_size)}) with "
            f"{self.download_workers} download / {self.extract_workers} extract / "
            f"{self.convert_workers} convert workers, {self.parts_per_tar} parts/tar, "
            f"staging cap {format_gib(self.max_staging_bytes)}"
        )

        stop_event = threading.Event()
        stats = Stats()
        budget = StagingBudget(self.max_staging_bytes)
        commit_lock = threading.Lock() if self.convert_workers > 1 else None

        with self._lock:
            self._active = len(scheduled_ids)
        download_q: queue.Queue = queue.Queue(maxsize=self.download_workers * 2)
        extract_q: queue.Queue = queue.Queue(maxsize=self.extract_workers * 2)
        convert_q: queue.Queue = queue.Queue(maxsize=self.convert_workers * 2)

        threads: list[threading.Thread] = []
        for _ in range(self.download_workers):
            threads.append(
                self._spawn(
                    lambda: self._worker_loop(
                        download_q, stop_event, stats, "download",
                        lambda tar: self._handle_download(tar, download_q, extract_q, stop_event, budget, stats),
                    )
                )
            )
        for _ in range(self.extract_workers):
            threads.append(
                self._spawn(
                    lambda: self._worker_loop(
                        extract_q, stop_event, stats, "extract",
                        lambda tar: self._handle_extract(tar, download_q, extract_q, convert_q, stop_event, budget),
                    )
                )
            )
        for _ in range(self.convert_workers):
            threads.append(
                self._spawn(
                    lambda: self._worker_loop(
                        convert_q, stop_event, stats, "convert",
                        lambda tar: self._handle_convert(tar, convert_q, stop_event, budget, stats, commit_lock),
                    )
                )
            )
        threads.append(self._spawn(lambda: self._monitor_loop(scheduled, stop_event, stats, budget)))
        feeder = self._spawn(lambda: self._feed(scheduled_ids, download_q, stop_event))

        try:
            feeder.join()
            while True:
                with self._lock:
                    active = self._active
                if active == 0 or stop_event.is_set():
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\ninterrupted: draining in-flight work and saving state", flush=True)
            stop_event.set()

        for thread in threads:
            thread.join(timeout=60)
        self.state.save()

        counts = self.state.status_counts()
        print(
            f"done: {counts.get(STATUS_CONVERTED, 0)} converted, "
            f"{counts.get(STATUS_FAILED_PERMANENT, 0)} failed permanently, "
            f"{counts.get(STATUS_SKIPPED, 0)} skipped"
        )
        if counts.get(STATUS_FAILED_PERMANENT):
            for tar in self.state.tars.values():
                if tar.status == STATUS_FAILED_PERMANENT:
                    print(f"  FAIL {tar.tar_id}: {tar.error[:200]}")
        return counts.get(STATUS_FAILED_PERMANENT, 0) == 0

    def _spawn(self, target: Callable[[], None]) -> threading.Thread:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def _feed(self, ids: list[str], q: queue.Queue, stop_event: threading.Event) -> None:
        for tar_id in ids:
            if not self._put(q, tar_id, stop_event):
                return

    def _put(self, q: queue.Queue, tar_id: str, stop_event: threading.Event) -> bool:
        while True:
            if stop_event.is_set():
                return False
            try:
                q.put(tar_id, timeout=1)
                return True
            except queue.Full:
                continue

    def _active_done(self) -> None:
        with self._lock:
            self._active -= 1

    def _worker_loop(
        self,
        q: queue.Queue,
        stop_event: threading.Event,
        stats: Stats,
        stage: str,
        handler: Callable[[TarState], bool],
    ) -> None:
        while True:
            if stop_event.is_set():
                return
            try:
                tar_id = q.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    if self._active == 0:
                        return
                continue
            try:
                with stats.lock:
                    stats.stage_counts[stage] = stats.stage_counts.get(stage, 0) + 1
                still_active = handler(self.state.tars[tar_id])
                if not still_active:
                    self._active_done()
            except Exception as exc:  # handler bug guard: never die silently
                print(f"[{stage}] unexpected error handling {tar_id}: {exc}", flush=True)
                tar = self.state.tars.get(tar_id)
                if tar is not None:
                    self.state.mark_failed(
                        tar, f"unexpected {stage} error: {exc}", tar.attempts
                    )
                self._active_done()
            finally:
                with stats.lock:
                    stats.stage_counts[stage] -= 1
                q.task_done()

    # ------------------------------------------------------------- stage D

    def _handle_download(
        self,
        tar: TarState,
        download_q: queue.Queue,
        extract_q: queue.Queue,
        stop_event: threading.Event,
        budget: StagingBudget,
        stats: Stats,
    ) -> bool:
        tar_id = tar.tar_id
        budget.reserve(tar_id, tar.size)
        t0 = time.time()
        try:
            factory = obsio.make_stream_factory(self.infos, self.bucket, tar.key, self.timeout)

            def progress(amount: int) -> None:
                with stats.lock:
                    stats.bytes_downloaded += amount

            fetched = obsio.download_tar(
                self.staging.tar_path(tar_id),
                self.staging.parts,
                tar_id,
                tar.key,
                tar.size,
                factory,
                self.parts_per_tar,
                self.retries,
                stop_event,
                progress,
            )
            self.state.mark_downloaded(tar, fetched)
            elapsed = time.time() - t0
            if fetched:
                print(
                    f"[download] {tar_id}: {format_gib(tar.size)} in {elapsed:.0f}s "
                    f"({fetched / 1024 ** 2 / max(elapsed, 1e-3):.0f} MiB/s)",
                    flush=True,
                )
            else:
                print(f"[download] {tar_id}: resumed from part markers, nothing to fetch", flush=True)
            return self._put(extract_q, tar_id, stop_event)
        except obsio.StopRequested:
            budget.release(tar_id)
            return False
        except obsio.OBSFatalError as exc:
            self.state.mark_failed(tar, str(exc), tar.attempts)
            budget.release(tar_id)
            return False
        except Exception as exc:
            attempts = tar.attempts + 1
            budget.release(tar_id)
            if attempts < self.retries:
                self.state.update(tar, attempts=attempts)
                delay = min(2 ** attempts, 30)
                print(
                    f"[download] {tar_id} failed ({exc}); retry {attempts}/{self.retries} "
                    f"in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                return self._put(download_q, tar_id, stop_event)
            self.state.mark_failed(tar, f"download failed after {attempts} attempts: {exc}", attempts)
            return False

    # ------------------------------------------------------------- stage E

    def _handle_extract(
        self,
        tar: TarState,
        download_q: queue.Queue,
        extract_q: queue.Queue,
        convert_q: queue.Queue,
        stop_event: threading.Event,
        budget: StagingBudget,
    ) -> bool:
        tar_id = tar.tar_id
        t0 = time.time()
        try:
            layout = extract_mod.extract_tar(self.staging.tar_path(tar_id), tar_id, self.staging)
            budget.reserve(tar_id, max(0, layout.extracted_bytes - tar.size))
            with self._cap_lock:
                if (
                    self.episode_cap is not None
                    and self._episodes_scheduled + layout.episode_count > self.episode_cap
                ):
                    self.state.mark_skipped(
                        tar, f"global --max-episodes cap ({self.episode_cap}) reached"
                    )
                    extract_mod.remove_extracted_dir(self.staging, tar_id)
                    extract_mod.remove_download_artifacts(self.staging, tar_id)
                    budget.release(tar_id)
                    return False
                self._episodes_scheduled += layout.episode_count
            self.state.mark_extracted(tar, layout.episode_count)
            print(
                f"[extract] {tar_id}: {layout.episode_count} episodes, "
                f"{format_gib(layout.extracted_bytes)} in {time.time() - t0:.0f}s",
                flush=True,
            )
            return self._put(convert_q, tar_id, stop_event)
        except extract_mod.ExtractionError as exc:
            self.state.mark_failed(tar, str(exc), tar.attempts)
            budget.release(tar_id)
            return False
        except extract_mod.ExtractionCorruptError as exc:
            attempts = tar.attempts + 1
            extract_mod.remove_download_artifacts(self.staging, tar_id)  # force re-download
            budget.release(tar_id)
            if attempts < self.retries:
                self.state.update(tar, attempts=attempts)
                delay = min(2 ** attempts, 30)
                print(
                    f"[extract] {tar_id} corrupt ({exc}); re-downloading "
                    f"(attempt {attempts}/{self.retries}) in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                return self._put(download_q, tar_id, stop_event)
            self.state.mark_failed(tar, f"tar corrupt after {attempts} downloads: {exc}", attempts)
            return False
        except Exception as exc:
            attempts = tar.attempts + 1
            budget.release(tar_id)
            if attempts < self.retries:
                self.state.update(tar, attempts=attempts)
                delay = min(2 ** attempts, 30)
                print(
                    f"[extract] {tar_id} failed ({exc}); retry {attempts}/{self.retries} "
                    f"in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                return self._put(extract_q, tar_id, stop_event)
            self.state.mark_failed(tar, f"extraction failed after {attempts} attempts: {exc}", attempts)
            return False

    # ------------------------------------------------------------- stage C

    def _handle_convert(
        self,
        tar: TarState,
        convert_q: queue.Queue,
        stop_event: threading.Event,
        budget: StagingBudget,
        stats: Stats,
        commit_lock: Optional[threading.Lock],
    ) -> bool:
        tar_id = tar.tar_id
        t0 = time.time()
        try:
            count = convert_one_tar(
                tar, self.staging, self.args, self.output_dir, self.writer,
                self.accumulator, commit_lock, self.state, self.keep_extracted,
            )
            with stats.lock:
                stats.tars_converted += 1
            print(f"[convert] {tar_id} OK: {count} episodes in {time.time() - t0:.0f}s", flush=True)
            budget.release(tar_id)
            return False
        except (ValueError, json.JSONDecodeError) as exc:
            self.state.mark_failed(tar, str(exc), tar.attempts)
            budget.release(tar_id)
            print(
                f"[convert] {tar_id} PERMANENT FAIL: {exc}; staging files kept "
                f"for inspection (delete them manually or re-run with --force-retry-failed)",
                flush=True,
            )
            return False
        except Exception as exc:
            attempts = tar.attempts + 1
            if attempts < self.retries:
                self.state.update(tar, attempts=attempts)
                delay = min(2 ** attempts, 30)
                print(
                    f"[convert] {tar_id} failed ({exc}); retry {attempts}/{self.retries} "
                    f"in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                return self._put(convert_q, tar_id, stop_event)
            self.state.mark_failed(tar, f"conversion failed after {attempts} attempts: {exc}", attempts)
            budget.release(tar_id)
            print(f"[convert] {tar_id} PERMANENT FAIL after {attempts} attempts: {exc}", flush=True)
            return False

    # ------------------------------------------------------------ monitor

    def _monitor_loop(
        self,
        scheduled: list[TarState],
        stop_event: threading.Event,
        stats: Stats,
        budget: StagingBudget,
    ) -> None:
        terminal = (STATUS_CONVERTED, STATUS_FAILED_PERMANENT, STATUS_SKIPPED)
        scheduled_ids = {tar.tar_id for tar in scheduled}
        last_bytes = 0
        last_time = time.time()
        while not stop_event.is_set():
            time.sleep(20)
            if stop_event.is_set():
                return
            with stats.lock:
                bytes_this_run = stats.bytes_downloaded
            now = time.time()
            rate = (bytes_this_run - last_bytes) / max(1e-3, now - last_time)
            done = sum(1 for tar in scheduled if tar.status == STATUS_CONVERTED)
            failed = sum(1 for tar in scheduled if tar.status == STATUS_FAILED_PERMANENT)
            with self._lock:
                active = self._active
            remaining = sum(
                tar.size for tar in scheduled if tar.tar_id in scheduled_ids and tar.status not in terminal
            )
            line = (
                f"[progress] {done}/{len(scheduled)} converted, {failed} failed, {active} in "
                f"flight | download {rate / 1024 ** 2:.0f} MiB/s | remaining "
                f"{format_gib(remaining)} | staging reserved {format_gib(budget.used())}"
            )
            if rate > 1024 * 1024:
                line += f" | ETA {remaining / rate / 3600:.1f}h"
            print(line, flush=True)
            last_bytes, last_time = bytes_this_run, now
