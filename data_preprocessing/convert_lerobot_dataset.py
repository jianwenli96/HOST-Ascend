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
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import numpy as np
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on the selected environment
    raise SystemExit(
        "Missing conversion dependency. Run this script in an environment with "
        "pyarrow and numpy (the host_alignment environment has them). "
        f"Original error: {exc}"
    ) from exc


DEFAULT_MAIN_VIEW = "observation.images.head"
DEFAULT_GRIPPER_VIEWS = ("observation.images.hand_left", "observation.images.hand_right")
ACTION_SOURCE_COLUMNS = (
    "actions.end.position",
    "actions.end.orientation",
    "actions.effector.position",
)
JOINT_SOURCE_COLUMN = "observation.states.joint.position"
ACTION_KEYS = (
    "left_arm_position",
    "left_arm_rotation",
    "left_arm_gripper",
    "right_arm_position",
    "right_arm_rotation",
    "right_arm_gripper",
)
JOINT_KEYS = ("left_arm_joints", "right_arm_joints")
TRAJECTORY_KEYS = ACTION_KEYS + JOINT_KEYS
QUATERNION_ORDER = "xyzw"  # source convention, verified against meta/stats_delta_state.json
EXCLUDED_SOURCE_COLUMNS = (
    "actions.joint.position",
    "actions.waist.position",
    "actions.head.position",
    "actions.robot.velocity",
    "observation.states.end.position",
    "observation.states.end.orientation",
    "observation.states.effector.position",
    "observation.states.joint.current_value",
    "observation.states.waist.position",
    "observation.states.head.position",
    "observation.states.robot.position",
    "observation.states.robot.orientation",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
)


@dataclass(frozen=True)
class Episode:
    source: Path
    source_dir_name: str
    task_key: str
    task_dir: Path
    episode_dir: Path
    episode_name: str
    frame_count: int
    instruction: str
    videos: dict[str, Path]
    view_shapes: dict[str, tuple[int, int]]
    source_video_keys: dict[str, str]


def natural_key(path: Path) -> list[object]:
    """Sort paths with numeric components numerically (episode_2 before episode_10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def video_key_to_view_dir(video_key: str) -> str:
    """Extract view directory name from video key.

    Args:
        video_key: Full video key path, e.g., "observation.images.hand_left"

    Returns:
        The last component of the video key, e.g., "hand_left"
    """
    return video_key.split(".")[-1]


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
    args = parser.parse_args()

    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in the range 1..100")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(
            "Missing required tools: " + ", ".join(missing) + ". The source videos are "
            "AV1-coded, so frames are extracted with the ffmpeg CLI. Install it (e.g. "
            "'brew install ffmpeg') or run on a machine where it is available."
        )


def jpeg_quality_to_qv(quality: int) -> int:
    """Map 1..100 JPEG quality to ffmpeg mjpeg -q:v 31..1 (95 -> 3)."""
    return 31 - round((quality - 1) * 30 / 99)


def discover_dataset_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")
    if (input_dir / "meta" / "info.json").is_file():
        return [input_dir]
    dataset_dirs = sorted(
        (path for path in input_dir.iterdir() if (path / "meta" / "info.json").is_file()),
        key=natural_key,
    )
    if not dataset_dirs:
        raise ValueError(f"No LeRobot dataset directories (meta/info.json) found under {input_dir}")
    return dataset_dirs


def load_task_texts(dataset_dir: Path) -> dict[int, str]:
    tasks_file = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks_file.is_file():
        return {}
    tasks: dict[int, str] = {}
    with tasks_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "task_index" in entry and entry.get("task"):
                tasks[int(entry["task_index"])] = entry["task"].strip()
    return tasks


def load_info(dataset_dir: Path) -> dict:
    with (dataset_dir / "meta" / "info.json").open("r", encoding="utf-8") as file:
        info = json.load(file)
    if not str(info.get("codebase_version", "")).startswith("v2"):
        print(
            f"WARNING: {dataset_dir} reports codebase_version "
            f"{info.get('codebase_version')!r}, not the LeRobot v2 GR00T convention this "
            f"converter targets; conversion may fail on the Parquet layout."
        )
    return info


def discover_episode_sources(dataset_dir: Path) -> list[tuple[Path, int]]:
    parquet_files = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"), key=natural_key)
    if not parquet_files:
        raise ValueError(f"No data/chunk-*/episode_*.parquet files found under {dataset_dir}")
    sources = []
    for path in parquet_files:
        match = re.search(r"episode_(\d+)\.parquet$", path.name)
        if match is None:
            raise ValueError(f"Unexpected Parquet filename (expected episode_XXXXXX.parquet): {path}")
        sources.append((path, int(match.group(1))))
    return sources


def resolve_source_video(
    info: dict, dataset_dir: Path, video_key: str, episode_index: int
) -> Path:
    template = info.get("video_path")
    if template:
        video_path = dataset_dir / template.format(
            episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
            video_key=video_key,
            episode_index=episode_index,
        )
    else:
        video_path = dataset_dir / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"
    if not video_path.is_file():
        raise ValueError(f"Missing source video for {video_key}: {video_path}")
    return video_path


def episode_task_index(source: Path) -> int:
    try:
        table = pq.read_table(source, columns=["task_index"])
    except Exception:
        return 0
    values = table.column("task_index").to_pylist()
    return int(values[0]) if values else 0


def episode_frame_count(source: Path) -> int:
    return pq.ParquetFile(source).metadata.num_rows


def instruction_short(task_text: str) -> str:
    short = task_text.split("|", 1)[0].strip()
    if not short.endswith((".", "!", "?")):
        short += "."
    return short


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "task"


def build_episodes(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Episode]:
    dataset_dirs = discover_dataset_dirs(input_dir)
    view_specs = [(video_key_to_view_dir(args.main_view), args.main_view)]
    if not args.no_gripper_view:
        for gripper_view in args.gripper_views:
            view_specs.append((video_key_to_view_dir(gripper_view), gripper_view))
    used_task_dirs: set[str] = set()
    task_dir_names: dict[tuple[str, int], str] = {}
    episodes: list[Episode] = []

    for dataset_dir in dataset_dirs:
        info = load_info(dataset_dir)
        features = info.get("features", {})
        for view_dir, video_key in view_specs:
            if video_key not in features or features[video_key].get("dtype") != "video":
                available = sorted(
                    key for key, feature in features.items() if feature.get("dtype") == "video"
                )
                raise ValueError(
                    f"{dataset_dir}: view {video_key!r} is not a video feature of this dataset. "
                    f"Available video features: {available}"
                )
        view_shapes = {
            view_dir: tuple(features[video_key]["shape"][:2])  # (height, width)
            for view_dir, video_key in view_specs
        }
        tasks = load_task_texts(dataset_dir)
        default_instruction = f"{slugify(dataset_dir.name).replace('_', ' ').title()}."

        for source, episode_index in discover_episode_sources(dataset_dir):
            frame_count = episode_frame_count(source)
            if frame_count < args.min_frames:
                raise ValueError(
                    f"{source}: only {frame_count} frames; alignment requires at least "
                    f"{args.min_frames}"
                )
            task_index = episode_task_index(source)
            task_text = tasks.get(task_index, default_instruction)
            instruction = task_text if not args.short_instructions else instruction_short(task_text)
            task_slug = slugify(instruction_short(task_text) if not args.short_instructions else instruction)

            # One task directory per (dataset dir, task); episodes of the same task share it.
            task_key = (str(dataset_dir), task_index)
            task_dir_name = task_dir_names.get(task_key)
            if task_dir_name is None:
                task_dir_name = f"{task_slug}_{dataset_dir.name}"
                if task_dir_name in used_task_dirs:
                    task_dir_name = f"{task_slug}_{task_index}_{dataset_dir.name}"
                if task_dir_name in used_task_dirs:
                    raise ValueError(f"Duplicate task directory name derived: {task_dir_name}")
                used_task_dirs.add(task_dir_name)
                task_dir_names[task_key] = task_dir_name
            task_dir = output_dir / task_dir_name
            episode_name = f"episode_{episode_index:06d}"
            videos = {
                view_dir: resolve_source_video(info, dataset_dir, video_key, episode_index)
                for view_dir, video_key in view_specs
            }
            episodes.append(
                Episode(
                    source=source.resolve(),
                    source_dir_name=dataset_dir.name,
                    task_key=task_slug,
                    task_dir=task_dir.resolve(),
                    episode_dir=(task_dir / episode_name).resolve(),
                    episode_name=episode_name,
                    frame_count=frame_count,
                    instruction=instruction,
                    videos=videos,
                    view_shapes=view_shapes,
                    source_video_keys={view_dir: video_key for view_dir, video_key in view_specs},
                )
            )
        if not tasks:
            print(
                f"WARNING: {dataset_dir} has no meta/tasks.jsonl; using the directory name as "
                f"the task instruction ({default_instruction!r})."
            )

    episodes.sort(key=lambda episode: (natural_key(episode.task_dir), natural_key(episode.episode_dir)))
    if args.max_episodes is not None and len(episodes) > args.max_episodes:
        print(f"NOTE: limiting conversion to the first {args.max_episodes} episodes (--max-episodes)")
        episodes = episodes[: args.max_episodes]
    return episodes


def ffprobe_json(path: Path, show_entries: str) -> list[dict]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        show_entries,
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe produced unparseable output for {path}") from exc


def validate_source_video(video: Path, view_dir: str, episode: Episode) -> None:
    streams = ffprobe_json(video, "stream=nb_frames,width,height,codec_name")
    if not streams:
        raise ValueError(f"{video}: no video stream found")
    stream = streams[0]
    expected_shape = episode.view_shapes[view_dir]
    actual_shape = (int(stream.get("height", 0)), int(stream.get("width", 0)))
    if actual_shape != expected_shape:
        raise ValueError(
            f"{video}: expected frame shape {expected_shape}, got {actual_shape} "
            f"(codec {stream.get('codec_name')})"
        )
    try:
        nb_frames = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError):
        return  # container does not store a frame count; the JPEG count check still applies
    if nb_frames != episode.frame_count:
        raise ValueError(
            f"{video}: video has {nb_frames} frames but the Parquet trajectory has "
            f"{episode.frame_count}; refusing to silently truncate"
        )


def assert_images_match(path: Path, frame_count: int, shape: tuple[int, int]) -> None:
    if not path.is_dir():
        raise ValueError(f"Missing output image directory: {path}")
    files = sorted(path.glob("*.jpg"), key=natural_key)
    if len(files) != frame_count:
        raise ValueError(f"Invalid image count in {path}: expected {frame_count}, got {len(files)}")
    expected_names = [f"{index}.jpg" for index in range(frame_count)]
    if [file.name for file in files] != expected_names:
        raise ValueError(f"Unexpected image filenames in {path}; expected contiguous 0.jpg..N.jpg")
    height, width = shape
    for file in (files[0], files[len(files) // 2], files[-1]):
        streams = ffprobe_json(file, "stream=width,height")
        actual = None
        if streams:
            actual = (int(streams[0].get("height", 0)), int(streams[0].get("width", 0)))
        if actual != shape:
            raise ValueError(f"Invalid image {file}: expected {(height, width, 3)}, got {actual}")


def encode_views(episode: Episode, jpeg_quality: int, overwrite: bool, output_size: Optional[tuple[int, int]] = None) -> None:
    needs_write: dict[str, bool] = {}
    for view_dir, video in episode.videos.items():
        target = episode.episode_dir / view_dir
        if target.exists() and not overwrite:
            assert_images_match(target, episode.frame_count, episode.view_shapes[view_dir])
            needs_write[view_dir] = False
        else:
            needs_write[view_dir] = True

    stale_videos = [episode.episode_dir / f"{view_dir}.mp4" for view_dir in episode.videos]
    if not overwrite and any(path.exists() for path in stale_videos):
        raise ValueError(
            "Stale MP4 output conflicts with image-sequence loading; rerun with --overwrite "
            "to replace it"
        )

    if not any(needs_write.values()):
        print(f"[reuse] {episode.task_dir.name}/{episode.episode_name}")
        return

    temporary = {
        view_dir: target.with_name(f".{target.name}.converting")
        for view_dir, target in (
            (view_dir, episode.episode_dir / view_dir) for view_dir in episode.videos
        )
        if needs_write[view_dir]
    }
    try:
        for temp_path in temporary.values():
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True)

        for view_dir, temp_path in temporary.items():
            validate_source_video(episode.videos[view_dir], view_dir, episode)
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(episode.videos[view_dir]),
            ]
            if output_size is not None:
                height, width = output_size
                command.extend(["-vf", f"scale={width}:{height}"])
            command.extend([
                "-q:v",
                str(jpeg_quality_to_qv(jpeg_quality)),
                "-start_number",
                "0",
                str(temp_path / "%d.jpg"),
            ])
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed on {episode.videos[view_dir]}: {completed.stderr.strip()[-500:]}"
                )
            output_shape = output_size if output_size is not None else episode.view_shapes[view_dir]
            assert_images_match(temp_path, episode.frame_count, output_shape)

        for view_dir, temp_path in temporary.items():
            target = episode.episode_dir / view_dir
            if target.exists():
                shutil.rmtree(target)
            temp_path.rename(target)

        for stale_video in stale_videos:
            if stale_video.exists():
                stale_video.unlink()
        views = "/".join(episode.videos)
        if output_size is not None:
            output_shapes = {view_dir: output_size for view_dir in episode.videos}
        else:
            output_shapes = episode.view_shapes
        shapes = " ".join(f"{width}x{height}" for height, width in output_shapes.values())
        print(
            f"[write] {episode.task_dir.name}/{episode.episode_name}: "
            f"{episode.frame_count} JPEG frames per view ({views}), {shapes}"
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


def quat_xyzw_to_rpy(quats: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions (...,4) to intrinsic-XYZ Euler [roll, pitch, yaw]."""
    x, y, z, w = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def column_to_array(column, expected_ndim: int, name: str) -> np.ndarray:
    values = column.to_pylist()
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: cannot convert column to a numeric array") from exc
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name}: expected a {expected_ndim}D column, got shape {array.shape}"
        )
    return array


def load_trajectory(source: Path) -> dict[str, np.ndarray]:
    """Load bimanual actions/proprioception and convert quaternions to HOST RPY."""
    table = pq.read_table(source, columns=list(ACTION_SOURCE_COLUMNS) + [JOINT_SOURCE_COLUMN])
    available = set(table.column_names)
    missing = [column for column in ACTION_SOURCE_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"{source}: missing required action columns: {missing}")

    frame_count = table.num_rows
    positions = column_to_array(table.column("actions.end.position"), 3, "actions.end.position")
    if positions.shape != (frame_count, 2, 3):
        raise ValueError(
            f"{source}: expected actions.end.position shape {(frame_count, 2, 3)}, "
            f"got {positions.shape}"
        )
    quats = column_to_array(
        table.column("actions.end.orientation"), 3, "actions.end.orientation"
    )
    if quats.shape != (frame_count, 2, 4):
        raise ValueError(
            f"{source}: expected actions.end.orientation shape {(frame_count, 2, 4)}, "
            f"got {quats.shape}"
        )
    norms = np.linalg.norm(quats, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-3):
        raise ValueError(
            f"{source}: {QUATERNION_ORDER} quaternions are not unit norm "
            f"(min/max norm {norms.min():.4f}/{norms.max():.4f})"
        )
    grippers = column_to_array(
        table.column("actions.effector.position"), 2, "actions.effector.position"
    )
    if grippers.shape != (frame_count, 2):
        raise ValueError(
            f"{source}: expected actions.effector.position shape {(frame_count, 2)}, "
            f"got {grippers.shape}"
        )

    rotations = quat_xyzw_to_rpy(quats)
    trajectory = {
        "left_arm_position": positions[:, 0, :],
        "left_arm_rotation": rotations[:, 0, :],
        "left_arm_gripper": grippers[:, 0:1],
        "right_arm_position": positions[:, 1, :],
        "right_arm_rotation": rotations[:, 1, :],
        "right_arm_gripper": grippers[:, 1:2],
    }
    if JOINT_SOURCE_COLUMN in available:
        joints = column_to_array(
            table.column(JOINT_SOURCE_COLUMN), 2, JOINT_SOURCE_COLUMN
        )
        if joints.shape != (frame_count, 14):
            raise ValueError(
                f"{source}: expected {JOINT_SOURCE_COLUMN} shape {(frame_count, 14)}, "
                f"got {joints.shape}"
            )
        trajectory["left_arm_joints"] = joints[:, :7]
        trajectory["right_arm_joints"] = joints[:, 7:]
    else:
        print(f"WARNING: {source}: {JOINT_SOURCE_COLUMN!r} missing; joints omitted")

    for key, values in trajectory.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: non-finite values found in trajectory field {key!r}")
    return trajectory


def trajectory_norm_min_delta(episodes: list[Episode]) -> dict[str, dict[str, list[float]]]:
    minima: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    for episode in episodes:
        for key, values in load_trajectory(episode.source).items():
            current_min = values.min(axis=0)
            current_max = values.max(axis=0)
            minima[key] = current_min if key not in minima else np.minimum(minima[key], current_min)
            maxima[key] = current_max if key not in maxima else np.maximum(maxima[key], current_max)

    result = {}
    for key in TRAJECTORY_KEYS:
        if key not in minima:
            continue
        delta = maxima[key] - minima[key]
        if np.any(delta <= 0):
            raise ValueError(
                f"Trajectory field {key!r} has non-positive normalization delta: {delta}"
            )
        result[key] = {
            "min": minima[key].tolist(),
            "delta": delta.tolist(),
        }
    return result


def write_robot_trajectory_data(
    episodes: list[Episode], output_dir: Path, dataset_id: str
) -> tuple[Path, dict[str, dict[str, list[float]]]]:
    norms = trajectory_norm_min_delta(episodes)
    present_action_keys = [key for key in ACTION_KEYS if key in norms]
    present_joint_keys = [key for key in JOINT_KEYS if key in norms]
    for episode in episodes:
        trajectory = load_trajectory(episode.source)
        entries = []
        for index in range(episode.frame_count):
            entries.append(
                {
                    "left_arm_position": trajectory["left_arm_position"][index].tolist(),
                    "left_arm_rotation": trajectory["left_arm_rotation"][index].tolist(),
                    "left_arm_gripper": float(trajectory["left_arm_gripper"][index, 0]),
                    "right_arm_position": trajectory["right_arm_position"][index].tolist(),
                    "right_arm_rotation": trajectory["right_arm_rotation"][index].tolist(),
                    "right_arm_gripper": float(trajectory["right_arm_gripper"][index, 0]),
                    **(
                        {
                            "left_arm_joints": trajectory["left_arm_joints"][index].tolist(),
                            "right_arm_joints": trajectory["right_arm_joints"][index].tolist(),
                        }
                        if "left_arm_joints" in trajectory
                        else {}
                    ),
                }
            )
        write_json_atomic(
            episode.episode_dir / f"{episode.episode_name}.json", {"data": entries}
        )

    mapping_dir = output_dir / "joint_action_mapping"
    mapping_file = mapping_dir / f"{dataset_id}_joint_action_mapping.json"
    mapping = {
        str(output_dir.resolve()): {
            "action_keys": present_action_keys,
            "joint_keys": present_joint_keys,
            "norm_min_delta": norms,
        }
    }
    write_json_atomic(mapping_file, mapping)
    return mapping_file, norms


def group_by_task(episodes: Iterable[Episode]) -> dict[str, list[Episode]]:
    grouped: dict[str, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.task_dir.name, []).append(episode)
    return grouped


def write_sidecars(
    episodes: list[Episode], output_dir: Path, dataset_id: str, args: argparse.Namespace,
    action_mapping_file: Path,
) -> None:
    grouped = group_by_task(episodes)
    for task_episodes in grouped.values():
        episode_dirs = [str(episode.episode_dir) for episode in task_episodes]
        task_dir = task_episodes[0].task_dir
        for index, episode in enumerate(task_episodes):
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

        # Session-level instruction mapping, consumed by build_task_dictionary.py
        # (see README §1) to derive task groupings.
        write_json_atomic(
            task_dir / "instruction.json",
            {
                episode.episode_name: {"detailed_instruction": episode.instruction}
                for episode in task_episodes
            },
        )

    video_paths = [str(episode.episode_dir) for episode in episodes]
    write_json_atomic(output_dir / f"{dataset_id}_video_paths.json", video_paths)

    view_dirs = [video_key_to_view_dir(args.main_view)]
    if not args.no_gripper_view:
        view_dirs.extend(video_key_to_view_dir(view) for view in args.gripper_views)
    cam_mapping = {str(episode.task_dir): view_dirs for episode in episodes}
    mapping_dir = output_dir / "cam_mapping"
    write_json_atomic(mapping_dir / f"{dataset_id}_cam_mapping.json", cam_mapping)

    manifest = {
        "dataset_id": dataset_id,
        "source_episode_count": len(episodes),
        "main_episode_count": len(video_paths),
        "reference_episode_count": 0,
        "tasks": {task: len(items) for task, items in grouped.items()},
        "main_embodiment": "robot",
        "reference_embodiment": None,
        "output_views": view_dirs,
        "storage_format": "jpeg_image_sequence",
        "jpeg_quality": args.jpeg_quality,
        "video_paths_json": str((output_dir / f"{dataset_id}_video_paths.json").resolve()),
        "cam_mapping_dir": str(mapping_dir.resolve()),
        "robot_action": {
            "source_fields": list(ACTION_SOURCE_COLUMNS),
            "output_fields": list(ACTION_KEYS),
            "dimensions": 14,
            "position_units": "metres",
            "rotation_units": f"radians_xyz_euler_from_{QUATERNION_ORDER}_quaternion",
            "gripper_semantics": {"0": "closed", "1": "open"},
            "mapping_file": str(action_mapping_file.resolve()),
        },
        "robot_proprioception": {
            "source_fields": [JOINT_SOURCE_COLUMN],
            "output_fields": list(JOINT_KEYS),
            "dimensions": 14,
            "joint_units": "radians",
        },
        "excluded_source_columns": list(EXCLUDED_SOURCE_COLUMNS),
        "episodes": [
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
            for episode in episodes
        ],
    }
    write_json_atomic(output_dir / "conversion_manifest.json", manifest)


def verify_robot_trajectory_data(
    episodes: list[Episode], output_dir: Path, dataset_id: str
) -> None:
    mapping_file = (
        output_dir / "joint_action_mapping" / f"{dataset_id}_joint_action_mapping.json"
    )
    with mapping_file.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if len(mapping) != 1:
        raise ValueError(f"Expected one mapping entry in {mapping_file}, got {len(mapping)}")
    entry = next(iter(mapping.values()))
    expected_norms = trajectory_norm_min_delta(episodes)
    if entry.get("action_keys") != [key for key in ACTION_KEYS if key in expected_norms]:
        raise ValueError(f"Unexpected action keys in {mapping_file}")
    if entry.get("joint_keys") != [key for key in JOINT_KEYS if key in expected_norms]:
        raise ValueError(f"Unexpected joint keys in {mapping_file}")

    actual_norms = entry.get("norm_min_delta", {})
    if set(actual_norms) != set(expected_norms):
        raise ValueError(
            f"Unexpected normalization keys in {mapping_file}: {sorted(actual_norms)}"
        )
    for key in expected_norms:
        for stat in ("min", "delta"):
            actual = np.asarray(actual_norms[key][stat], dtype=np.float64)
            expected = np.asarray(expected_norms[key][stat], dtype=np.float64)
            if not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError(f"Incorrect {key}.{stat} in {mapping_file}")

    for episode in episodes:
        action_file = episode.episode_dir / f"{episode.episode_name}.json"
        with action_file.open("r", encoding="utf-8") as file:
            entries = json.load(file).get("data")
        if not isinstance(entries, list) or len(entries) != episode.frame_count:
            actual_count = None if not isinstance(entries, list) else len(entries)
            raise ValueError(
                f"Invalid action count in {action_file}: expected {episode.frame_count}, "
                f"got {actual_count}"
            )
        expected_trajectory = load_trajectory(episode.source)
        actual_trajectory = {
            "left_arm_position": np.asarray([item["left_arm_position"] for item in entries]),
            "left_arm_rotation": np.asarray([item["left_arm_rotation"] for item in entries]),
            "left_arm_gripper": np.asarray([item["left_arm_gripper"] for item in entries]).reshape(-1, 1),
            "right_arm_position": np.asarray([item["right_arm_position"] for item in entries]),
            "right_arm_rotation": np.asarray([item["right_arm_rotation"] for item in entries]),
            "right_arm_gripper": np.asarray([item["right_arm_gripper"] for item in entries]).reshape(-1, 1),
        }
        if "left_arm_joints" in expected_trajectory:
            actual_trajectory["left_arm_joints"] = np.asarray(
                [item["left_arm_joints"] for item in entries]
            )
            actual_trajectory["right_arm_joints"] = np.asarray(
                [item["right_arm_joints"] for item in entries]
            )
        for key in expected_trajectory:
            if actual_trajectory[key].shape != expected_trajectory[key].shape or not np.allclose(
                actual_trajectory[key], expected_trajectory[key], rtol=0, atol=1e-12
            ):
                raise ValueError(f"Trajectory field {key!r} in {action_file} differs from source")


def verify_conversion(episodes: list[Episode], output_dir: Path, dataset_id: str, output_size: Optional[tuple[int, int]] = None) -> None:
    for task_dir in {episode.task_dir for episode in episodes}:
        instruction_file = task_dir / "instruction.json"
        if not instruction_file.is_file():
            raise ValueError(f"Missing required sidecar: {instruction_file}")
        with instruction_file.open("r", encoding="utf-8") as file:
            instructions = json.load(file)
        expected = {
            episode.episode_name: episode.instruction
            for episode in episodes
            if episode.task_dir == task_dir
        }
        actual = {
            name: entry.get("detailed_instruction", "")
            for name, entry in instructions.items()
        }
        if actual != expected:
            raise ValueError(f"Unexpected contents in {instruction_file}")

    for episode in episodes:
        for view_dir in episode.videos:
            assert_images_match(
                episode.episode_dir / view_dir,
                episode.frame_count,
                episode.view_shapes[view_dir],
            )
        for filename in ("instruction.txt", "task_paths.json", "task_paths_eval.json"):
            if not (episode.episode_dir / filename).is_file():
                raise ValueError(f"Missing required sidecar: {episode.episode_dir / filename}")

    video_paths_file = output_dir / f"{dataset_id}_video_paths.json"
    cam_mapping_file = output_dir / "cam_mapping" / f"{dataset_id}_cam_mapping.json"
    with video_paths_file.open("r", encoding="utf-8") as file:
        video_paths = json.load(file)
    expected_paths = [str(episode.episode_dir) for episode in episodes]
    if video_paths != expected_paths:
        raise ValueError(f"Unexpected contents in {video_paths_file}")

    with cam_mapping_file.open("r", encoding="utf-8") as file:
        cam_mapping = json.load(file)
    expected_tasks = {str(episode.task_dir) for episode in episodes}
    expected_views = [MAIN_VIEW_DIR] if not any(
        GRIPPER_VIEW_DIR in episode.videos for episode in episodes
    ) else [MAIN_VIEW_DIR, GRIPPER_VIEW_DIR]
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


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        episodes = build_episodes(input_dir, output_dir, args)
        if not episodes:
            raise ValueError("No episodes discovered")

        if args.verify_only:
            require_ffmpeg()
            output_size = tuple(args.output_size) if args.output_size is not None else None
            verify_conversion(episodes, output_dir, args.dataset_id, output_size)
            return 0

        require_ffmpeg()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_size = tuple(args.output_size) if args.output_size is not None else None
        for episode in episodes:
            encode_views(episode, args.jpeg_quality, args.overwrite, output_size)
        action_mapping_file, _ = write_robot_trajectory_data(
            episodes, output_dir, args.dataset_id
        )
        write_sidecars(episodes, output_dir, args.dataset_id, args, action_mapping_file)
        verify_conversion(episodes, output_dir, args.dataset_id, output_size)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
