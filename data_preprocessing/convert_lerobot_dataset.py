#!/usr/bin/env python3
"""Convert LeRobot robot datasets into HOST episode directories.

The source is a LeRobot v2.1 dataset in the GR00T convention (``codebase_version:
"v2.1"`` in ``meta/info.json``, e.g. the AgiBot A2D bimanual robot): per-episode
Parquet files under ``data/chunk-*/episode_XXXXXX.parquet`` and per-view MP4
videos under ``videos/chunk-*/<video_key>/episode_XXXXXX.mp4``, plus
``meta/tasks.jsonl`` task instructions.  This converter is the robot-only
counterpart of ``convert_human_and_robot.py``: there is no synchronized human
video, so each episode emits a single (Main) episode directory and its
``task_paths.json`` lists same-task *robot* episodes.

Trajectory JSON files contain 14-dimensional bimanual actions built from
``actions.end.position`` (3+3 XYZ positions), ``actions.end.orientation`` (4+4
xyzw quaternions converted to HOST [roll, pitch, yaw] Euler triples), and
``actions.effector.position`` (2 gripper commands, 0=closed 1=open), plus
14-dimensional proprioception from ``observation.states.joint.position`` (7+7
joint angles).  Source positions are already in metres and joint angles in
radians, so no scaling is applied.  Columns that do not map to the HOST
action/proprioception convention (waist, head, robot base, velocities,
``actions.joint.position``) are excluded and listed in the manifest.

Each task directory also receives a session-level ``instruction.json``
(episode name -> detailed instruction) so that ``build_task_dictionary.py``
can consume the converted output directly to (re)derive task groupings.

Frame sequences are written as JPEG images decoded from the source MP4s with
the ``ffmpeg`` CLI (the source videos are AV1-coded, which OpenCV builds
commonly cannot decode).  Image sequences are used instead of MP4 because the
alignment environment's decord build may not support the codecs available
through OpenCV.

The pipeline processes one task at a time (see ``convert_lerobot.source.iter_task_units``),
which keeps it compatible with streaming ingestion where tasks arrive over
time; the episodes of each task are encoded in parallel (``--workers``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import numpy  # noqa: F401 - dependency check for a friendly exit message
    import pyarrow.parquet  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the selected environment
    raise SystemExit(
        "Missing conversion dependency. Run this script in an environment with "
        "pyarrow and numpy (the host_alignment environment has them). "
        f"Original error: {exc}"
    ) from exc

from convert_lerobot.media import encode_task, require_ffmpeg
from convert_lerobot.outputs import (
    DatasetWriter,
    verify_dataset,
    verify_task,
    write_episode_sidecars,
    write_task_sidecars,
)
from convert_lerobot.source import (
    DEFAULT_GRIPPER_VIEWS,
    DEFAULT_MAIN_VIEW,
    Episode,
    TaskUnit,
    iter_task_units,
)
from convert_lerobot.trajectory import (
    NormAccumulator,
    load_trajectory,
    norm_mapping_path,
    write_episode_trajectory,
    write_norm_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v2 GR00T-format datasets to HOST episode directories."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="A LeRobot dataset directory, or a root containing several (each with meta/info.json).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination HOST data directory.")
    parser.add_argument(
        "--dataset-id",
        default="RobotTask",
        help="Dataset id used in <id>_video_paths.json and camera mapping (default: RobotTask).",
    )
    parser.add_argument(
        "--main-view",
        default=DEFAULT_MAIN_VIEW,
        help=f"Source video key for the main camera (default: {DEFAULT_MAIN_VIEW}).",
    )
    parser.add_argument(
        "--gripper-views",
        nargs="+",
        default=list(DEFAULT_GRIPPER_VIEWS),
        help=f"Source video keys for gripper cameras (default: {' '.join(DEFAULT_GRIPPER_VIEWS)}).",
    )
    parser.add_argument(
        "--no-gripper-view",
        action="store_true",
        help="Emit a single-view dataset (main view only).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=24,
        help="Reject episodes shorter than alignment CONFIG.TRAIN.NUM_FRAMES (default: 24).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality in the range 1..100, mapped to ffmpeg -q:v (default: 95).",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Output image size as HEIGHT WIDTH (e.g., --output-size 224 224). If not specified, images are kept at source resolution.",
    )
    parser.add_argument(
        "--short-instructions",
        action="store_true",
        help="Write only the part of the task text before the first '|' to instruction.txt.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most this many episodes (handy for quick verification runs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing image sequences (e.g. after changing --main-view/--gripper-view).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing conversion without writing files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Encode this many episodes of a task in parallel (default: min(4, cpu count)).",
    )
    args = parser.parse_args()

    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in the range 1..100")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def process_task(
    unit: TaskUnit,
    args: argparse.Namespace,
    output_dir: Path,
    output_size: Optional[tuple[int, int]],
    accumulator: NormAccumulator,
    dataset_writer: DatasetWriter,
) -> None:
    """Run every stage for one task unit: encode -> trajectory -> sidecars -> verify."""
    encode_task(unit.episodes, args.jpeg_quality, args.overwrite, output_size, args.workers)
    for episode in unit.episodes:
        trajectory = load_trajectory(episode.source)
        write_episode_trajectory(episode, trajectory)
        accumulator.update(trajectory)
    write_norm_mapping(output_dir, args.dataset_id, accumulator.finalize())
    episode_dirs = [str(episode.episode_dir) for episode in unit.episodes]
    for index, episode in enumerate(unit.episodes):
        write_episode_sidecars(episode, episode_dirs, index)
    write_task_sidecars(unit)
    dataset_writer.add_task(unit)
    verify_task(unit, output_size)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        require_ffmpeg()
        output_size = tuple(args.output_size) if args.output_size is not None else None
        episodes: list[Episode] = []

        if args.verify_only:
            for unit in iter_task_units(input_dir, output_dir, args):
                verify_task(unit, output_size)
                episodes.extend(unit.episodes)
            if not episodes:
                raise ValueError("No episodes discovered")
            verify_dataset(episodes, output_dir, args.dataset_id)
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        accumulator = NormAccumulator()
        dataset_writer = DatasetWriter(
            output_dir, args.dataset_id, args, norm_mapping_path(output_dir, args.dataset_id)
        )
        for unit in iter_task_units(input_dir, output_dir, args):
            process_task(unit, args, output_dir, output_size, accumulator, dataset_writer)
            episodes.extend(unit.episodes)
        if not episodes:
            raise ValueError("No episodes discovered")
        verify_dataset(episodes, output_dir, args.dataset_id)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
