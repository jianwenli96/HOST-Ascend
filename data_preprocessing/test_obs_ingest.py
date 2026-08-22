#!/usr/bin/env python3
"""Synthetic tests for the OBS streaming ingestion pipeline.

Standalone (no pytest): run ``python test_obs_ingest.py`` from
``data_preprocessing/`` in an environment with numpy and pyarrow (the
host_alignment conda env has them).  The OBS SDK itself is NOT required —
download tests inject a fake client factory, following the mock-stream
pattern of ``test_resumable.py`` in the OBS SDK checkout.

Covers: part download resume across mid-stream deaths, fatal status
classification, extraction safety/layout/corruption, the per-tar state
machine, DatasetWriter idempotency + manifest rebuild, norm-stat merging
exactness, and view-spec resolution.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obs_ingest import extract as extract_mod
from obs_ingest import obsio, state as state_mod, viewspec
from obs_ingest.extract import StagingPaths


class FakeClient:
    def close(self):
        pass


class FakeStream:
    """Serves data[offset:]; dies with RuntimeError after die_after local bytes."""

    def __init__(self, data, offset, die_after=None):
        self.data = data
        self.pos = offset
        self.read_sofar = 0
        self.die_after = die_after

    def read(self, amt):
        if self.die_after is not None and self.read_sofar >= self.die_after:
            raise RuntimeError("simulated mid-stream death")
        limit = amt if self.die_after is None else min(amt, self.die_after - self.read_sofar)
        chunk = self.data[self.pos : self.pos + limit]
        self.pos += len(chunk)
        self.read_sofar += len(chunk)
        return chunk

    def close(self):
        pass


def make_factory(data, die_schedule):
    """factory(offset) -> (client, stream); die_schedule: {connection_no: bytes_then_die}."""
    calls = {"n": 0}

    def factory(offset):
        calls["n"] += 1
        return FakeClient(), FakeStream(data, offset, die_after=die_schedule.get(calls["n"]))

    return factory, calls


def build_tar_gz(members, mode="w:gz"):
    buf = io.BytesIO()
    tf = tarfile.open(fileobj=buf, mode=mode)
    for name, content in members:
        info = tarfile.TarInfo(name)
        payload = content.encode("utf-8") if isinstance(content, str) else content
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    tf.close()
    return buf.getvalue()


def task_layout_tar(dir_name="task_327"):
    return build_tar_gz(
        [
            (f"{dir_name}/meta/info.json", '{"codebase_version": "v2.1"}'),
            (f"{dir_name}/meta/tasks.jsonl", '{"task_index": 0, "task": "Test task."}\n'),
            (f"{dir_name}/data/chunk-000/episode_000000.parquet", os.urandom(4096)),
            (f"{dir_name}/data/chunk-000/episode_000001.parquet", os.urandom(4096)),
            (f"{dir_name}/videos/chunk-000/obs/episode_000000.mp4", os.urandom(4096)),
        ]
    )


def check(condition, label):
    assert condition, label
    print(f"PASS: {label}")


def test_plan_parts():
    check(obsio.plan_parts(1000, 8) == [(0, 1000)], "plan_parts collapses small objects to 1 part")
    check(len(obsio.plan_parts(10 ** 8, 8)) == 8, "plan_parts splits large objects")
    check(
        obsio.plan_parts(10, 0) == [(0, 10)],
        "plan_parts handles zero requested parts",
    )
    total = sum(end - start for start, end in obsio.plan_parts(9999, 8))
    check(total == 9999, "plan_parts covers every byte exactly")


def test_part_download_resume_and_markers(tmp):
    data = os.urandom(3 * 1024 * 1024)
    # Force multi-part behaviour with a small object.
    obsio.MIN_PART_SIZE = 100_000
    obsio.CHUNK_SIZE = 64 * 1024

    # Kill the first 4 connections: with 4 parallel parts, one connection per
    # part dies mid-transfer and the part resumes on a fresh one.
    factory, calls = make_factory(data, {n: 400_000 for n in range(1, 5)})
    tar_path = tmp / "tars" / "task_001.tar.gz"
    parts_dir = tmp / "parts"
    fetched = obsio.download_tar(
        tar_path, parts_dir, "task_001", "pre/task_001.tar.gz", len(data),
        factory, parts=4, retries=3,
    )
    check(fetched == len(data), "downloaded every byte across resumed connections")
    check(calls["n"] >= 8, "each part reconnected after the simulated death")
    check(tar_path.read_bytes() == data, "downloaded file content matches the object")
    markers = sorted(parts_dir.glob("task_001.part.*"))
    check(len(markers) == 4, "one marker per part")

    # Re-run: all markers intact -> zero bytes fetched, zero connections.
    factory2, calls2 = make_factory(data, {})
    fetched2 = obsio.download_tar(
        tar_path, parts_dir, "task_001", "pre/task_001.tar.gz", len(data),
        factory2, parts=4, retries=3,
    )
    check(fetched2 == 0 and calls2["n"] == 0, "completed parts are skipped entirely")

    # Remove one marker: only that part is re-fetched.
    markers[1].unlink()
    factory3, calls3 = make_factory(data, {})
    fetched3 = obsio.download_tar(
        tar_path, parts_dir, "task_001", "pre/task_001.tar.gz", len(data),
        factory3, parts=4, retries=3,
    )
    part_sizes = [end - start for start, end in obsio.plan_parts(len(data), 4)]
    check(fetched3 == part_sizes[1], "only the missing part was re-downloaded")
    check(tar_path.read_bytes() == data, "file content intact after part re-download")

    # A changed object size invalidates nothing (markers store the old size):
    # a new download run with a different size re-fetches everything.
    grown = data + os.urandom(10)
    factory4, calls4 = make_factory(grown, {})
    fetched4 = obsio.download_tar(
        tar_path, parts_dir, "task_001", "pre/task_001.tar.gz", len(grown),
        factory4, parts=4, retries=3,
    )
    check(fetched4 == len(grown), "changed object size re-downloads all parts")
    check(tar_path.read_bytes() == grown, "regrown file content matches the new object")


def test_download_fatal(tmp):
    data = os.urandom(100_000)

    def fatal_factory(offset):
        raise obsio.OBSFatalError("server ignored Range header")

    tar_path = tmp / "tars" / "task_002.tar.gz"
    parts_dir = tmp / "parts"
    try:
        obsio.download_tar(
            tar_path, parts_dir, "task_002", "pre/task_002.tar.gz", len(data),
            fatal_factory, parts=2, retries=3,
        )
        raise AssertionError("expected OBSFatalError")
    except obsio.OBSFatalError:
        check(True, "OBSFatalError propagates immediately (permanent)")

    def transient_factory(offset):
        if offset == 0:
            raise RuntimeError("transport error")
        return FakeClient(), FakeStream(data, offset)

    try:
        obsio.download_tar(
            tmp / "tars" / "task_003.tar.gz", parts_dir, "task_003",
            "pre/task_003.tar.gz", len(data), transient_factory, parts=2, retries=1,
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        check("after 1 attempts" in str(exc), "transient errors retry then fail with a clear message")


def test_extract_safe_layout_corrupt(tmp):
    staging = StagingPaths(tmp / "staging")
    staging.ensure_dirs()

    tar_path = staging.tars / "task_327.tar.gz"
    tar_path.write_bytes(task_layout_tar("task_327"))
    layout = extract_mod.extract_tar(tar_path, "task_327", staging)
    check(layout.episode_count == 2, "extraction counts the parquet files")
    check(
        (layout.dataset_dir / "meta" / "info.json").is_file(),
        "layout validation accepts the standard task dir",
    )
    check(
        not (staging.extracted / ".task_327.tmp").exists(),
        "the temporary directory is gone after publishing",
    )

    # Reject wrong directory name.
    tar_path.write_bytes(task_layout_tar("task_999"))
    try:
        extract_mod.extract_tar(tar_path, "task_327", staging)
        raise AssertionError("expected ExtractionError")
    except extract_mod.ExtractionError as exc:
        check("task_327" in str(exc), "mismatched dataset dir name is a permanent error")

    # Reject path traversal, absolute paths, and symlinks.
    for bad in (
        "../evil.txt",
        "/abs/evil.txt",
    ):
        bad_tar = build_tar_gz([(bad, "x"), ("task_327/meta/info.json", "{}")])
        tar_path.write_bytes(bad_tar)
        try:
            extract_mod.extract_tar(tar_path, "task_327", staging)
            raise AssertionError(f"expected ExtractionError for {bad!r}")
        except extract_mod.ExtractionError:
            check(True, f"unsafe member {bad!r} rejected")

    buf = io.BytesIO()
    tf = tarfile.open(fileobj=buf, mode="w:gz")
    link = tarfile.TarInfo("task_327/meta/info.json")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    tf.addfile(link)
    tf.close()
    tar_path.write_bytes(buf.getvalue())
    try:
        extract_mod.extract_tar(tar_path, "task_327", staging)
        raise AssertionError("expected ExtractionError for symlink")
    except extract_mod.ExtractionError:
        check(True, "symlink members rejected")

    # Missing info.json -> permanent error.
    tar_path.write_bytes(build_tar_gz([("task_327/data/chunk-000/episode_000000.parquet", b"x")]))
    try:
        extract_mod.extract_tar(tar_path, "task_327", staging)
        raise AssertionError("expected ExtractionError")
    except extract_mod.ExtractionError:
        check(True, "missing meta/info.json is a permanent error")

    # Truncated gzip stream -> corrupt (transient) error.
    good = task_layout_tar("task_327")
    tar_path.write_bytes(good[: len(good) // 2])
    try:
        extract_mod.extract_tar(tar_path, "task_327", staging)
        raise AssertionError("expected ExtractionCorruptError")
    except extract_mod.ExtractionCorruptError:
        check(True, "truncated gzip stream classified as corrupt (re-download)")


def test_state_machine(tmp):
    path = tmp / "obs_pipeline_state.json"
    state = state_mod.PipelineState(path)
    tar = state.add_tar("Agibot_Beta_Lerobot_Amap/task_327.tar.gz", 123456)
    state.mark_downloaded(tar, 123456)
    state.mark_extracted(tar, 46)
    stats = {"min": {"left_arm_gripper": [0.0]}, "max": {"left_arm_gripper": [1.0]}}
    state.mark_converted(tar, stats)
    state.save()

    reloaded = state_mod.PipelineState(path)
    check(len(reloaded.tars) == 1, "state survives a save/reload round-trip")
    restored = reloaded.tars["task_327"]
    check(restored.status == state_mod.STATUS_CONVERTED, "status restored")
    check(restored.norm_stats == stats, "norm snapshot restored with the converted mark")
    check(restored.episode_count == 46, "episode count restored")
    check(reloaded.status_counts()[state_mod.STATUS_CONVERTED] == 1, "status_counts works")


def _fake_args():
    return argparse.Namespace(
        main_view="observation.images.head_compress",
        gripper_views=["observation.images.hand_left_compress"],
        no_gripper_view=False,
        jpeg_quality=95,
    )


def test_writer_idempotent_and_rebuild(tmp):
    import numpy as np  # local: this test needs the converter environment

    from convert_lerobot.outputs import DatasetWriter, verify_dataset_artifacts
    from convert_lerobot.source import Episode, TaskUnit

    output_dir = tmp / "output"
    output_dir.mkdir()
    task_dir = output_dir / "task_task_327"
    episode_dir = task_dir / "episode_000000"
    episode = Episode(
        source=(task_dir / "source.parquet").resolve(),
        source_dir_name="task_327",
        task_key="task",
        task_dir=task_dir.resolve(),
        episode_dir=episode_dir.resolve(),
        episode_name="episode_000000",
        frame_count=2,
        instruction="Test task.",
        videos={"head": (task_dir / "v.mp4").resolve()},
        view_shapes={"head": (224, 224)},
        source_video_keys={"head": "observation.images.head_compress"},
    )
    unit = TaskUnit(dataset_dir_name="task_327", task_index=0, task_dir=task_dir.resolve(), episodes=[episode])
    mapping_file = output_dir / "joint_action_mapping" / "mapping.json"

    writer = DatasetWriter(output_dir, "Test", _fake_args(), mapping_file)
    writer.add_task(unit)
    writer.add_task(unit)  # crash-retry simulation: must not duplicate
    check(len(writer.video_paths) == 1, "add_task is idempotent (no duplicate episodes)")
    check(len(writer.episode_records) == 1, "manifest records stay deduplicated")

    rebuilt = DatasetWriter.from_manifest(output_dir, "Test", _fake_args(), mapping_file)
    check(rebuilt.video_paths == writer.video_paths, "from_manifest restores video_paths")
    check(rebuilt.task_counts == writer.task_counts, "from_manifest restores task counts")
    check(rebuilt.cam_mapping == writer.cam_mapping, "from_manifest restores cam_mapping")
    check(rebuilt.episode_records == writer.episode_records, "from_manifest restores episode records")

    try:
        DatasetWriter.from_manifest(output_dir, "Other", _fake_args(), mapping_file)
        raise AssertionError("expected dataset id mismatch error")
    except ValueError:
        check(True, "from_manifest refuses a different dataset id")

    # verify_dataset_artifacts against on-disk files.
    episode_dir.mkdir(parents=True)
    (episode_dir / "instruction.txt").write_text("Test task.\n")
    (episode_dir / "task_paths.json").write_text("{}")
    (episode_dir / "task_paths_eval.json").write_text("{}")
    (episode_dir / "episode_000000.json").write_text(json.dumps({"data": [{}, {}]}))
    verify_dataset_artifacts(output_dir, "Test", verify_episode_trajectories=True)
    check(True, "verify_dataset_artifacts passes on a consistent dataset")

    with (output_dir / "Test_video_paths.json").open("w") as file:
        json.dump([], file)
    try:
        verify_dataset_artifacts(output_dir, "Test")
        raise AssertionError("expected divergence error")
    except ValueError:
        check(True, "verify_dataset_artifacts catches video_paths divergence")


def test_merge_stats():
    import numpy as np

    from convert_lerobot.trajectory import NormAccumulator

    rng = np.random.default_rng(0)
    direct = NormAccumulator()
    snapshots = []
    for _ in range(3):
        acc = NormAccumulator()
        trajectory = {
            "left_arm_position": rng.uniform(-1, 1, (100, 3)),
            "left_arm_gripper": rng.uniform(0, 1, (100, 1)),
        }
        acc.update(trajectory)
        direct.update(trajectory)
        snapshots.append(acc.stats_snapshot())
    merged = NormAccumulator()
    merged.merge_stats(snapshots)
    check(
        json.dumps(merged.stats_snapshot(), sort_keys=True)
        == json.dumps(direct.stats_snapshot(), sort_keys=True),
        "merging per-tar snapshots reproduces live accumulation exactly",
    )


def test_viewspec():
    compress_info = {
        "features": {
            "observation.images.head_compress": {"dtype": "video", "shape": [224, 224, 3]},
            "observation.images.hand_left_compress": {"dtype": "video", "shape": [224, 224, 3]},
            "observation.images.hand_right_compress": {"dtype": "video", "shape": [224, 224, 3]},
        }
    }
    full_info = {
        "features": {
            "observation.images.head": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.hand_left": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.hand_right": {"dtype": "video", "shape": [480, 640, 3]},
        }
    }
    defaults = argparse.Namespace(
        main_view=viewspec.DEFAULT_MAIN_VIEW,
        gripper_views=list(viewspec.DEFAULT_GRIPPER_VIEWS),
        output_size=None,
    )
    spec = viewspec.resolve_view_spec([compress_info, compress_info], defaults)
    check(spec["source"] == "compress" and spec["output_size"] is None, "all-compress probes resolve to compress views")

    spec = viewspec.resolve_view_spec([compress_info, full_info], defaults)
    check(spec["source"] == "full_res", "mixed probes fall back to the full-resolution views")

    explicit = argparse.Namespace(
        main_view="observation.images.head_center_fisheye",
        gripper_views=["observation.images.hand_left"],
        output_size=[224, 224],
    )
    spec = viewspec.resolve_view_spec([], explicit)
    check(spec["source"] == "explicit" and spec["main_view"].endswith("fisheye"), "explicit view flags skip probing")

    try:
        viewspec.resolve_view_spec([], defaults)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        check(True, "empty probe with default views raises a clear error")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_plan_parts()
        test_part_download_resume_and_markers(tmp)
        test_download_fatal(tmp)
        test_extract_safe_layout_corrupt(tmp)
        test_state_machine(tmp)
        test_merge_stats()
        test_writer_idempotent_and_rebuild(tmp)
        test_viewspec()
    print("\nALL OBS INGESTION TESTS PASSED")


if __name__ == "__main__":
    main()
