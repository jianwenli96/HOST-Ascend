"""Stages D and E of the LeRobot converter: sidecar writing and verification.

Writes the metadata files that accompany the image sequences and trajectory
JSONs at three levels:

* episode level — instruction.txt, task_paths.json, task_paths_eval.json
* task level — instruction.json (consumed by build_task_dictionary.py)
* dataset level — {dataset_id}_video_paths.json, cam_mapping, conversion_manifest.json

Dataset-level files are rewritten from scratch after each task through
``DatasetWriter``, so the pipeline keeps working when tasks stream in one at
a time and a reader never observes a partial file.  The verify functions
re-read every written artifact and compare it against the source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .media import assert_images_match
from .source import (
    ACTION_KEYS,
    ACTION_SOURCE_COLUMNS,
    EXCLUDED_SOURCE_COLUMNS,
    JOINT_KEYS,
    JOINT_SOURCE_COLUMN,
    QUATERNION_ORDER,
    Episode,
    TaskUnit,
    video_key_to_view_dir,
    write_json_atomic,
    write_text_atomic,
)
from .trajectory import verify_robot_trajectory_data


def write_episode_sidecars(episode: Episode, episode_dirs: list[str], index: int) -> None:
    write_text_atomic(episode.episode_dir / "instruction.txt", episode.instruction + "\n")

    # Training may sample any same-task episode as the reference.
    write_json_atomic(episode.episode_dir / "task_paths.json", {"same": episode_dirs})

    # Evaluation is deterministic and pairs with a different same-task episode.
    peers = [dir_ for dir_ in episode_dirs if dir_ != str(episode.episode_dir)]
    eval_pool = peers or episode_dirs
    write_json_atomic(
        episode.episode_dir / "task_paths_eval.json",
        {"same": [eval_pool[index % len(eval_pool)]]},
    )


def write_task_sidecars(unit: TaskUnit) -> None:
    # Session-level instruction mapping, consumed by build_task_dictionary.py
    # (see README §1) to derive task groupings.
    write_json_atomic(
        unit.task_dir / "instruction.json",
        {
            episode.episode_name: {"detailed_instruction": episode.instruction}
            for episode in unit.episodes
        },
    )


class DatasetWriter:
    """Accumulates dataset-level sidecars, rewriting them after each task.

    The three dataset-level files are small, so they are written from scratch
    on every ``add_task``; this keeps streaming ingestion consistent task by
    task (a reader never sees a partial list) without a finalize step.
    """

    def __init__(
        self, output_dir: Path, dataset_id: str, args, mapping_file: Path
    ) -> None:
        self.output_dir = output_dir
        self.dataset_id = dataset_id
        self.args = args
        self.mapping_file = mapping_file
        self.view_dirs = [video_key_to_view_dir(args.main_view)]
        if not args.no_gripper_view:
            self.view_dirs.extend(video_key_to_view_dir(view) for view in args.gripper_views)
        self.video_paths: list[str] = []
        self.task_counts: dict[str, int] = {}
        self.cam_mapping: dict[str, list[str]] = {}
        self.episode_records: list[dict] = []
        self._seen_episode_dirs: set[str] = set()

    @classmethod
    def from_manifest(
        cls, output_dir: Path, dataset_id: str, args, mapping_file: Path
    ) -> "DatasetWriter":
        """Rebuild the writer from a previous run's conversion_manifest.json.

        ``_rewrite`` writes the manifest last of the three dataset-level files,
        so it acts as the commit marker: after a crash anywhere in an
        ``add_task``, the next add_task regenerates video_paths/cam_mapping
        from the manifest state.  Used by the streaming pipeline to resume
        into an output directory whose source files have been deleted.
        """
        manifest_path = output_dir / "conversion_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"Cannot resume into {output_dir}: conversion_manifest.json is missing. "
                "Use a fresh output directory, or restore the manifest from a backup."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("dataset_id") != dataset_id:
            raise ValueError(
                f"{manifest_path} belongs to dataset {manifest.get('dataset_id')!r}, "
                f"not {dataset_id!r}; refusing to mix datasets in one directory"
            )
        writer = cls(output_dir, dataset_id, args, mapping_file)
        records = list(manifest.get("episodes", []))
        writer.view_dirs = list(manifest.get("output_views") or writer.view_dirs)
        writer.video_paths = [record["episode_dir"] for record in records]
        writer.task_counts = dict(manifest.get("tasks", {}))
        writer.cam_mapping = {record["task_dir"]: writer.view_dirs for record in records}
        writer.episode_records = records
        writer._seen_episode_dirs = {record["episode_dir"] for record in records}
        return writer

    def add_task(self, unit: TaskUnit) -> None:
        # Idempotent: a retry of a task whose previous add_task committed but
        # whose state mark was lost (crash between the two) must not duplicate
        # episodes in video_paths.json / the manifest.
        new_episode_dirs = [
            str(episode.episode_dir)
            for episode in unit.episodes
            if str(episode.episode_dir) not in self._seen_episode_dirs
        ]
        if not new_episode_dirs:
            return
        self._seen_episode_dirs.update(new_episode_dirs)
        self.video_paths.extend(new_episode_dirs)
        self.task_counts[unit.task_dir.name] = len(unit.episodes)
        self.cam_mapping[str(unit.task_dir)] = self.view_dirs
        for episode in unit.episodes:
            self.episode_records.append(
                {
                    "source": str(episode.source),
                    "source_dataset_dir": episode.source_dir_name,
                    "task": episode.task_key,
                    "task_dir": str(episode.task_dir),
                    "episode": episode.episode_name,
                    "frame_count": episode.frame_count,
                    "source_video_keys": episode.source_video_keys,
                    "episode_dir": str(episode.episode_dir),
                }
            )
        self._rewrite()

    def _rewrite(self) -> None:
        write_json_atomic(
            self.output_dir / f"{self.dataset_id}_video_paths.json", self.video_paths
        )
        mapping_dir = self.output_dir / "cam_mapping"
        write_json_atomic(
            mapping_dir / f"{self.dataset_id}_cam_mapping.json", self.cam_mapping
        )
        manifest = {
            "dataset_id": self.dataset_id,
            "source_episode_count": len(self.video_paths),
            "main_episode_count": len(self.video_paths),
            "reference_episode_count": 0,
            "tasks": self.task_counts,
            "main_embodiment": "robot",
            "reference_embodiment": None,
            "output_views": self.view_dirs,
            "storage_format": "jpeg_image_sequence",
            "jpeg_quality": self.args.jpeg_quality,
            "video_paths_json": str(
                (self.output_dir / f"{self.dataset_id}_video_paths.json").resolve()
            ),
            "cam_mapping_dir": str(mapping_dir.resolve()),
            "robot_action": {
                "source_fields": list(ACTION_SOURCE_COLUMNS),
                "output_fields": list(ACTION_KEYS),
                "dimensions": 14,
                "position_units": "metres",
                "rotation_units": f"radians_xyz_euler_from_{QUATERNION_ORDER}_quaternion",
                "gripper_semantics": {"0": "closed", "1": "open"},
                "mapping_file": str(self.mapping_file.resolve()),
            },
            "robot_proprioception": {
                "source_fields": [JOINT_SOURCE_COLUMN],
                "output_fields": list(JOINT_KEYS),
                "dimensions": 14,
                "joint_units": "radians",
            },
            "excluded_source_columns": list(EXCLUDED_SOURCE_COLUMNS),
            "episodes": self.episode_records,
        }
        write_json_atomic(self.output_dir / "conversion_manifest.json", manifest)


def verify_task(unit: TaskUnit, output_size: Optional[tuple[int, int]] = None) -> None:
    """Verify one task's episode outputs and instruction.json against the source."""
    instruction_file = unit.task_dir / "instruction.json"
    if not instruction_file.is_file():
        raise ValueError(f"Missing required sidecar: {instruction_file}")
    with instruction_file.open("r", encoding="utf-8") as file:
        instructions = json.load(file)
    expected = {
        episode.episode_name: episode.instruction
        for episode in unit.episodes
    }
    actual = {
        name: entry.get("detailed_instruction", "")
        for name, entry in instructions.items()
    }
    if actual != expected:
        raise ValueError(f"Unexpected contents in {instruction_file}")

    for episode in unit.episodes:
        for view_dir in episode.videos:
            expected_shape = (
                output_size if output_size is not None else episode.view_shapes[view_dir]
            )
            assert_images_match(
                episode.episode_dir / view_dir, episode.frame_count, expected_shape
            )
        for filename in ("instruction.txt", "task_paths.json", "task_paths_eval.json"):
            if not (episode.episode_dir / filename).is_file():
                raise ValueError(f"Missing required sidecar: {episode.episode_dir / filename}")


def verify_dataset(episodes: list[Episode], output_dir: Path, dataset_id: str) -> None:
    """Verify dataset-level outputs against the full episode set.

    ``episodes`` is the list accumulated over the task loop.  Comparisons are
    order-insensitive because with streaming the list order is the arrival
    order of tasks, not a global sort.
    """
    video_paths_file = output_dir / f"{dataset_id}_video_paths.json"
    cam_mapping_file = output_dir / "cam_mapping" / f"{dataset_id}_cam_mapping.json"
    with video_paths_file.open("r", encoding="utf-8") as file:
        video_paths = json.load(file)
    expected_paths = [str(episode.episode_dir) for episode in episodes]
    if sorted(video_paths) != sorted(expected_paths):
        raise ValueError(f"Unexpected contents in {video_paths_file}")

    with cam_mapping_file.open("r", encoding="utf-8") as file:
        cam_mapping = json.load(file)
    expected_tasks = {str(episode.task_dir) for episode in episodes}
    expected_views = list(episodes[0].videos) if episodes else []
    if set(cam_mapping) != expected_tasks or any(
        views != expected_views for views in cam_mapping.values()
    ):
        raise ValueError(f"Unexpected contents in {cam_mapping_file}")

    verify_robot_trajectory_data(episodes, output_dir, dataset_id)

    print(
        f"Verified {len(episodes)} robot episodes across {len(expected_tasks)} tasks, "
        f"including 14D bimanual actions, 14D proprioception, and {len(expected_views)} "
        f"camera view(s)."
    )


def verify_dataset_artifacts(
    output_dir: Path, dataset_id: str, verify_episode_trajectories: bool = False
) -> None:
    """Source-free consistency check for streaming runs.

    ``verify_dataset`` above needs the source Parquet files, which a streaming
    run deletes after each tar commits; per-task acceptance is already done by
    ``verify_task`` at commit time.  This check instead verifies the
    dataset-level artifacts against each other and against the files on disk
    (existence always; per-frame trajectory JSON contents only when
    ``verify_episode_trajectories`` is set, since that re-parses every episode
    JSON of the dataset).
    """
    manifest_path = output_dir / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing required sidecar: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError(f"Unexpected dataset id in {manifest_path}: {manifest.get('dataset_id')!r}")
    records = list(manifest.get("episodes", []))
    episode_dirs = [record["episode_dir"] for record in records]
    if len(episode_dirs) != len(set(episode_dirs)):
        raise ValueError(f"Duplicate episode entries in {manifest_path}")

    video_paths_file = output_dir / f"{dataset_id}_video_paths.json"
    with video_paths_file.open("r", encoding="utf-8") as file:
        video_paths = json.load(file)
    if sorted(video_paths) != sorted(episode_dirs):
        raise ValueError(f"Contents of {video_paths_file} diverged from {manifest_path}")

    cam_mapping_file = output_dir / "cam_mapping" / f"{dataset_id}_cam_mapping.json"
    with cam_mapping_file.open("r", encoding="utf-8") as file:
        cam_mapping = json.load(file)
    expected_tasks = {record["task_dir"] for record in records}
    expected_views = list(manifest.get("output_views", []))
    if set(cam_mapping) != expected_tasks or any(
        views != expected_views for views in cam_mapping.values()
    ):
        raise ValueError(f"Contents of {cam_mapping_file} diverged from {manifest_path}")

    for record in records:
        episode_dir = Path(record["episode_dir"])
        for filename in ("instruction.txt", "task_paths.json", "task_paths_eval.json"):
            if not (episode_dir / filename).is_file():
                raise ValueError(f"Missing required sidecar: {episode_dir / filename}")
        trajectory_file = episode_dir / f"{record['episode']}.json"
        if not trajectory_file.is_file():
            raise ValueError(f"Missing required sidecar: {trajectory_file}")
        if verify_episode_trajectories:
            with trajectory_file.open("r", encoding="utf-8") as file:
                entries = json.load(file).get("data")
            if not isinstance(entries, list) or len(entries) != record["frame_count"]:
                actual_count = None if not isinstance(entries, list) else len(entries)
                raise ValueError(
                    f"Invalid action count in {trajectory_file}: expected "
                    f"{record['frame_count']}, got {actual_count}"
                )

    print(
        f"Verified dataset artifacts for {len(episode_dirs)} robot episodes across "
        f"{len(expected_tasks)} tasks ({dataset_id})."
    )
