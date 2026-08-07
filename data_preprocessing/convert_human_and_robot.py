#!/usr/bin/env python3
"""Convert paired HumanAndRobot HDF5 episodes for alignment/.

The source dataset stores a synchronized human video and robot video in each
HDF5 file.  alignment/ expects its main and reference videos to be separate
episode directories, so this converter emits one robot episode (main) and one
human episode (reference) per source file.  Robot episodes are written to the
dataset's primary ``*_video_paths.json`` and their ``task_paths.json`` files
point only to human episodes of the same task.

Robot trajectory JSON files contain the robot end-effector action from
``/end_position`` plus ``/gripper_state``, as well as 14-dimensional robot
proprioception from ``/qpos`` and ``/qvel``.  The v1 ``/action`` dataset is not
used because it represents the human hand pose in the robot frame.  Source
positions are converted from millimetres to metres, while source Euler angles,
joint positions, and joint velocities are converted from degrees to radians
for the HOST action/proprioception convention.

The source dataset's own visualization script passes frames directly from
HDF5 to OpenCV, which establishes that the stored channel order is BGR.  The
converter preserves that convention when encoding JPEG files.  Image
sequences are used instead of MP4 because the alignment environment's decord
build may not support the codecs available through OpenCV.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import cv2
    import h5py
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the selected environment
    raise SystemExit(
        "Missing conversion dependency. Run this script in an environment with "
        "h5py, opencv-python, and numpy (the host_alignment environment has them). "
        f"Original error: {exc}"
    ) from exc


SOURCE_VIEWS = {
    "human": "/cam_data/human_camera",
    "robot": "/cam_data/robot_camera",
}
ROBOT_ACTION_SOURCES = ("/end_position", "/gripper_state")
ROBOT_ACTION_KEYS = ("robot_position", "robot_rotation", "robot_gripper")
ROBOT_JOINT_SOURCES = ("/qpos", "/qvel")
ROBOT_JOINT_KEYS = ("robot_qpos", "robot_qvel")
ROBOT_TRAJECTORY_KEYS = ROBOT_ACTION_KEYS + ROBOT_JOINT_KEYS
POSITION_SCALE = 1.0e-3  # source millimetres -> metres
ROTATION_SCALE = np.pi / 180.0  # source degrees -> radians
JOINT_POSITION_SCALE = np.pi / 180.0  # source degrees -> radians
JOINT_VELOCITY_SCALE = np.pi / 180.0  # source degrees/second -> radians/second
OUTPUT_VIEW = "images"
DEFAULT_INSTRUCTIONS = {
    "grab_cup_v1": "Grab the cup.",
    "pull_plate_v1": "Pull the plate.",
}


@dataclass(frozen=True)
class EpisodePair:
    source: Path
    task_name: str
    episode_name: str
    frame_count: int
    height: int
    width: int
    human_dir: Path
    robot_dir: Path


def natural_key(path: Path) -> list[object]:
    """Sort paths with numeric components numerically (episode_2 before episode_10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired HumanAndRobot HDF5 files to alignment episode directories."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing task/*.hdf5.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination alignment data directory.")
    parser.add_argument(
        "--dataset-id",
        default="HumanAndRobot",
        help="Dataset id used in <id>_video_paths.json and camera mapping (default: HumanAndRobot).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of HDF5 frames read at once per camera (default: 16).",
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
        help="JPEG quality in the range 1..100 (default: 95).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing image sequences and remove stale converter-generated MP4 files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing conversion without writing files.",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in the range 1..100")
    return args


def task_instruction(task_name: str) -> str:
    if task_name in DEFAULT_INSTRUCTIONS:
        return DEFAULT_INSTRUCTIONS[task_name]
    readable = re.sub(r"_v\d+$", "", task_name).replace("_", " ").strip()
    return f"{readable[:1].upper()}{readable[1:]}." if readable else "Perform the task."


def discover_sources(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")
    sources = sorted(input_dir.glob("*/*.hdf5"), key=natural_key)
    if not sources:
        raise ValueError(f"No task/*.hdf5 files found under {input_dir}")
    return sources


def inspect_source(source: Path, output_dir: Path, min_frames: int) -> EpisodePair:
    with h5py.File(source, "r") as h5_file:
        required_datasets = (
            list(SOURCE_VIEWS.values())
            + list(ROBOT_ACTION_SOURCES)
            + list(ROBOT_JOINT_SOURCES)
        )
        missing = [dataset for dataset in required_datasets if dataset not in h5_file]
        if missing:
            raise ValueError(f"{source}: missing required datasets: {missing}")

        shapes = {name: tuple(h5_file[dataset].shape) for name, dataset in SOURCE_VIEWS.items()}
        if shapes["human"] != shapes["robot"]:
            raise ValueError(f"{source}: human/robot frame shapes differ: {shapes}")

        shape = shapes["human"]
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"{source}: expected camera shape [T,H,W,3], got {shape}")
        frame_count, height, width, _ = shape
        if frame_count < min_frames:
            raise ValueError(
                f"{source}: only {frame_count} frames; alignment requires at least {min_frames}"
            )
        end_shape = tuple(h5_file["/end_position"].shape)
        gripper_shape = tuple(h5_file["/gripper_state"].shape)
        qpos_shape = tuple(h5_file["/qpos"].shape)
        qvel_shape = tuple(h5_file["/qvel"].shape)
        if end_shape != (frame_count, 6):
            raise ValueError(f"{source}: expected /end_position shape {(frame_count, 6)}, got {end_shape}")
        if gripper_shape not in ((frame_count,), (frame_count, 1)):
            raise ValueError(
                f"{source}: expected /gripper_state shape {(frame_count,)} or "
                f"{(frame_count, 1)}, got {gripper_shape}"
            )
        if qpos_shape != (frame_count, 7):
            raise ValueError(f"{source}: expected /qpos shape {(frame_count, 7)}, got {qpos_shape}")
        if qvel_shape != (frame_count, 7):
            raise ValueError(f"{source}: expected /qvel shape {(frame_count, 7)}, got {qvel_shape}")

    task_name = source.parent.name
    episode_name = source.stem
    task_dir = output_dir / task_name
    return EpisodePair(
        source=source.resolve(),
        task_name=task_name,
        episode_name=episode_name,
        frame_count=frame_count,
        height=height,
        width=width,
        human_dir=(task_dir / f"human_{episode_name}").resolve(),
        robot_dir=(task_dir / f"robot_{episode_name}").resolve(),
    )


def assert_images_match(path: Path, episode: EpisodePair) -> None:
    if not path.is_dir():
        raise ValueError(f"Missing output image directory: {path}")
    files = sorted(path.glob("*.jpg"), key=natural_key)
    if len(files) != episode.frame_count:
        raise ValueError(
            f"Invalid image count in {path}: expected {episode.frame_count}, got {len(files)}"
        )
    expected_names = [f"{index}.jpg" for index in range(episode.frame_count)]
    if [file.name for file in files] != expected_names:
        raise ValueError(f"Unexpected image filenames in {path}; expected contiguous 0.jpg..N.jpg")
    for file in files:
        frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (episode.height, episode.width, 3):
            actual = None if frame is None else frame.shape
            raise ValueError(
                f"Invalid image {file}: expected {(episode.height, episode.width, 3)}, got {actual}"
            )


def encode_pair(episode: EpisodePair, chunk_size: int, jpeg_quality: int, overwrite: bool) -> None:
    targets = {
        "human": episode.human_dir / OUTPUT_VIEW,
        "robot": episode.robot_dir / OUTPUT_VIEW,
    }
    needs_write: dict[str, bool] = {}
    for name, target in targets.items():
        if target.exists() and not overwrite:
            assert_images_match(target, episode)
            needs_write[name] = False
        else:
            needs_write[name] = True

    stale_videos = [target.parent / f"{OUTPUT_VIEW}.mp4" for target in targets.values()]
    if not overwrite and any(path.exists() for path in stale_videos):
        raise ValueError(
            "Stale MP4 output conflicts with image-sequence loading; rerun with --overwrite "
            "to replace it"
        )

    if not any(needs_write.values()):
        print(f"[reuse] {episode.task_name}/{episode.episode_name}")
        return

    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)

    temporary = {
        name: target.with_name(f".{target.name}.converting")
        for name, target in targets.items()
        if needs_write[name]
    }
    try:
        for temp_path in temporary.values():
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir()

        with h5py.File(episode.source, "r") as h5_file:
            for start in range(0, episode.frame_count, chunk_size):
                end = min(start + chunk_size, episode.frame_count)
                for name, temp_path in temporary.items():
                    frames = h5_file[SOURCE_VIEWS[name]][start:end]
                    frame_min = int(frames.min())
                    frame_max = int(frames.max())
                    if frame_min < 0 or frame_max > 255:
                        raise ValueError(
                            f"{episode.source}:{SOURCE_VIEWS[name]} contains values outside "
                            f"8-bit range in frames {start}:{end}: [{frame_min}, {frame_max}]"
                        )
                    for offset, frame in enumerate(frames):
                        output_file = temp_path / f"{start + offset}.jpg"
                        written = cv2.imwrite(
                            str(output_file),
                            np.asarray(frame, dtype=np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                        )
                        if not written:
                            raise RuntimeError(f"OpenCV failed to write {output_file}")

        for name, temp_path in temporary.items():
            assert_images_match(temp_path, episode)
            target = targets[name]
            if target.exists():
                shutil.rmtree(target)
            temp_path.rename(target)

        for stale_video in stale_videos:
            if stale_video.exists():
                stale_video.unlink()
        print(
            f"[write] {episode.task_name}/{episode.episode_name}: "
            f"{episode.frame_count} JPEG frames per embodiment, {episode.width}x{episode.height}"
        )
    finally:
        for temp_path in temporary.values():
            if temp_path.exists():
                shutil.rmtree(temp_path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_robot_trajectory(source: Path) -> dict[str, np.ndarray]:
    """Load robot actions/proprioception and convert them to HOST physical units."""
    with h5py.File(source, "r") as h5_file:
        end_position = np.asarray(h5_file["/end_position"][...], dtype=np.float64)
        gripper = np.asarray(h5_file["/gripper_state"][...], dtype=np.float64).reshape(-1, 1)
        qpos = np.asarray(h5_file["/qpos"][...], dtype=np.float64)
        qvel = np.asarray(h5_file["/qvel"][...], dtype=np.float64)

    trajectory = {
        "robot_position": end_position[:, :3] * POSITION_SCALE,
        "robot_rotation": end_position[:, 3:] * ROTATION_SCALE,
        "robot_gripper": gripper,
        "robot_qpos": qpos * JOINT_POSITION_SCALE,
        "robot_qvel": qvel * JOINT_VELOCITY_SCALE,
    }
    for key, values in trajectory.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: non-finite values found in robot trajectory field {key!r}")
    return trajectory


def trajectory_norm_min_delta(episodes: list[EpisodePair]) -> dict[str, dict[str, list[float]]]:
    minima: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    for episode in episodes:
        for key, values in load_robot_trajectory(episode.source).items():
            current_min = values.min(axis=0)
            current_max = values.max(axis=0)
            minima[key] = current_min if key not in minima else np.minimum(minima[key], current_min)
            maxima[key] = current_max if key not in maxima else np.maximum(maxima[key], current_max)

    result = {}
    for key in ROBOT_TRAJECTORY_KEYS:
        delta = maxima[key] - minima[key]
        if np.any(delta <= 0):
            raise ValueError(
                f"Robot trajectory field {key!r} has non-positive normalization delta: {delta}"
            )
        result[key] = {
            "min": minima[key].tolist(),
            "delta": delta.tolist(),
        }
    return result


def write_robot_trajectory_data(
    episodes: list[EpisodePair], output_dir: Path, dataset_id: str
) -> tuple[Path, dict[str, dict[str, list[float]]]]:
    norms = trajectory_norm_min_delta(episodes)
    for episode in episodes:
        trajectory = load_robot_trajectory(episode.source)
        entries = []
        for index in range(episode.frame_count):
            entries.append(
                {
                    "robot_position": trajectory["robot_position"][index].tolist(),
                    "robot_rotation": trajectory["robot_rotation"][index].tolist(),
                    "robot_gripper": float(trajectory["robot_gripper"][index, 0]),
                    "robot_qpos": trajectory["robot_qpos"][index].tolist(),
                    "robot_qvel": trajectory["robot_qvel"][index].tolist(),
                }
            )
        action_file = episode.robot_dir / f"{episode.robot_dir.name}.json"
        write_json_atomic(action_file, {"data": entries})

    mapping_dir = output_dir / "joint_action_mapping"
    mapping_file = mapping_dir / f"{dataset_id}_joint_action_mapping.json"
    mapping = {
        str(output_dir.resolve()): {
            "action_keys": list(ROBOT_ACTION_KEYS),
            "joint_keys": list(ROBOT_JOINT_KEYS),
            "norm_min_delta": norms,
        }
    }
    write_json_atomic(mapping_file, mapping)
    return mapping_file, norms


def group_by_task(episodes: Iterable[EpisodePair]) -> dict[str, list[EpisodePair]]:
    grouped: dict[str, list[EpisodePair]] = {}
    for episode in episodes:
        grouped.setdefault(episode.task_name, []).append(episode)
    return grouped


def write_sidecars(
    episodes: list[EpisodePair], output_dir: Path, dataset_id: str, jpeg_quality: int,
    action_mapping_file: Path,
) -> None:
    grouped = group_by_task(episodes)
    for task_episodes in grouped.values():
        human_pool = [str(episode.human_dir) for episode in task_episodes]
        robot_pool = [str(episode.robot_dir) for episode in task_episodes]
        for episode in task_episodes:
            instruction = task_instruction(episode.task_name)
            for episode_dir in (episode.human_dir, episode.robot_dir):
                write_text_atomic(episode_dir / "instruction.txt", instruction + "\n")

            # Training may sample any same-task episode from the opposite embodiment.
            write_json_atomic(episode.robot_dir / "task_paths.json", {"same": human_pool})
            write_json_atomic(episode.human_dir / "task_paths.json", {"same": robot_pool})

            # Evaluation is deterministic and uses the synchronized source pair.
            write_json_atomic(
                episode.robot_dir / "task_paths_eval.json", {"same": [str(episode.human_dir)]}
            )
            write_json_atomic(
                episode.human_dir / "task_paths_eval.json", {"same": [str(episode.robot_dir)]}
            )

    video_paths = [str(episode.robot_dir) for episode in episodes]
    write_json_atomic(output_dir / f"{dataset_id}_video_paths.json", video_paths)

    cam_mapping = {
        str((output_dir / task_name).resolve()): [OUTPUT_VIEW]
        for task_name in grouped
    }
    mapping_dir = output_dir / "cam_mapping"
    write_json_atomic(mapping_dir / f"{dataset_id}_cam_mapping.json", cam_mapping)

    manifest = {
        "dataset_id": dataset_id,
        "source_episode_count": len(episodes),
        "main_episode_count": len(video_paths),
        "reference_episode_count": len(episodes),
        "tasks": {task: len(items) for task, items in grouped.items()},
        "main_embodiment": "robot",
        "reference_embodiment": "human",
        "output_view": OUTPUT_VIEW,
        "storage_format": "jpeg_image_sequence",
        "jpeg_quality": jpeg_quality,
        "video_paths_json": str((output_dir / f"{dataset_id}_video_paths.json").resolve()),
        "cam_mapping_dir": str(mapping_dir.resolve()),
        "robot_action": {
            "source_fields": ["/end_position", "/gripper_state"],
            "excluded_source_field": "/action",
            "output_fields": list(ROBOT_ACTION_KEYS),
            "dimensions": 7,
            "position_units": "metres",
            "rotation_units": "radians_xyz_euler",
            "gripper_semantics": {"0": "closed", "1": "open"},
            "mapping_file": str(action_mapping_file.resolve()),
        },
        "robot_proprioception": {
            "source_fields": list(ROBOT_JOINT_SOURCES),
            "output_fields": list(ROBOT_JOINT_KEYS),
            "dimensions": 14,
            "qpos_units": "radians",
            "qvel_units": "radians_per_second",
        },
        "episodes": [
            {
                "source": str(episode.source),
                "task": episode.task_name,
                "episode": episode.episode_name,
                "frame_count": episode.frame_count,
                "height": episode.height,
                "width": episode.width,
                "human_dir": str(episode.human_dir),
                "robot_dir": str(episode.robot_dir),
            }
            for episode in episodes
        ],
    }
    write_json_atomic(output_dir / "conversion_manifest.json", manifest)


def verify_robot_trajectory_data(
    episodes: list[EpisodePair], output_dir: Path, dataset_id: str
) -> None:
    mapping_file = (
        output_dir / "joint_action_mapping" / f"{dataset_id}_joint_action_mapping.json"
    )
    with mapping_file.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if len(mapping) != 1:
        raise ValueError(f"Expected one mapping entry in {mapping_file}, got {len(mapping)}")
    entry = next(iter(mapping.values()))
    if (
        entry.get("action_keys") != list(ROBOT_ACTION_KEYS)
        or entry.get("joint_keys") != list(ROBOT_JOINT_KEYS)
    ):
        raise ValueError(f"Unexpected action/joint keys in {mapping_file}")

    expected_norms = trajectory_norm_min_delta(episodes)
    actual_norms = entry.get("norm_min_delta", {})
    if set(actual_norms) != set(ROBOT_TRAJECTORY_KEYS):
        raise ValueError(
            f"Unexpected normalization keys in {mapping_file}: {sorted(actual_norms)}"
        )
    for key in ROBOT_TRAJECTORY_KEYS:
        for stat in ("min", "delta"):
            actual = np.asarray(actual_norms[key][stat], dtype=np.float64)
            expected = np.asarray(expected_norms[key][stat], dtype=np.float64)
            if not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError(f"Incorrect {key}.{stat} in {mapping_file}")

    for episode in episodes:
        action_file = episode.robot_dir / f"{episode.robot_dir.name}.json"
        with action_file.open("r", encoding="utf-8") as file:
            entries = json.load(file).get("data")
        if not isinstance(entries, list) or len(entries) != episode.frame_count:
            actual_count = None if not isinstance(entries, list) else len(entries)
            raise ValueError(
                f"Invalid action count in {action_file}: expected {episode.frame_count}, "
                f"got {actual_count}"
            )
        expected_trajectory = load_robot_trajectory(episode.source)
        actual_trajectory = {
            "robot_position": np.asarray([item["robot_position"] for item in entries]),
            "robot_rotation": np.asarray([item["robot_rotation"] for item in entries]),
            "robot_gripper": np.asarray([item["robot_gripper"] for item in entries]).reshape(-1, 1),
            "robot_qpos": np.asarray([item["robot_qpos"] for item in entries]),
            "robot_qvel": np.asarray([item["robot_qvel"] for item in entries]),
        }
        for key in ROBOT_TRAJECTORY_KEYS:
            if actual_trajectory[key].shape != expected_trajectory[key].shape or not np.allclose(
                actual_trajectory[key], expected_trajectory[key], rtol=0, atol=1e-12
            ):
                raise ValueError(f"Trajectory field {key!r} in {action_file} differs from source")


def verify_conversion(episodes: list[EpisodePair], output_dir: Path, dataset_id: str) -> None:
    for episode in episodes:
        for episode_dir in (episode.human_dir, episode.robot_dir):
            assert_images_match(episode_dir / OUTPUT_VIEW, episode)
            for filename in ("instruction.txt", "task_paths.json", "task_paths_eval.json"):
                if not (episode_dir / filename).is_file():
                    raise ValueError(f"Missing required sidecar: {episode_dir / filename}")

    video_paths_file = output_dir / f"{dataset_id}_video_paths.json"
    cam_mapping_file = output_dir / "cam_mapping" / f"{dataset_id}_cam_mapping.json"
    with video_paths_file.open("r", encoding="utf-8") as file:
        video_paths = json.load(file)
    expected_paths = [str(episode.robot_dir) for episode in episodes]
    if video_paths != expected_paths:
        raise ValueError(f"Unexpected contents in {video_paths_file}")

    with cam_mapping_file.open("r", encoding="utf-8") as file:
        cam_mapping = json.load(file)
    expected_tasks = {str((output_dir / episode.task_name).resolve()) for episode in episodes}
    if set(cam_mapping) != expected_tasks or any(views != [OUTPUT_VIEW] for views in cam_mapping.values()):
        raise ValueError(f"Unexpected contents in {cam_mapping_file}")

    verify_robot_trajectory_data(episodes, output_dir, dataset_id)

    print(
        f"Verified {len(episodes)} robot mains + {len(episodes)} human references "
        f"across {len(expected_tasks)} tasks, including 7D robot actions and 14D proprioception."
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        sources = discover_sources(input_dir)
        episodes = [inspect_source(source, output_dir, args.min_frames) for source in sources]

        if args.verify_only:
            verify_conversion(episodes, output_dir, args.dataset_id)
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            encode_pair(episode, args.chunk_size, args.jpeg_quality, args.overwrite)
        action_mapping_file, _ = write_robot_trajectory_data(
            episodes, output_dir, args.dataset_id
        )
        write_sidecars(
            episodes, output_dir, args.dataset_id, args.jpeg_quality,
            action_mapping_file,
        )
        verify_conversion(episodes, output_dir, args.dataset_id)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
