"""Stage A of the LeRobot converter: source discovery and task-unit planning.

Discovers LeRobot v2 GR00T-format datasets under an input root, reads their
metadata (``meta/info.json``, ``meta/tasks.jsonl``, ``data/chunk-*/episode_*.parquet``)
and yields one ``TaskUnit`` per (dataset, task) — the pipeline's processing
unit.  Yielding task by task (instead of building one global episode list)
keeps the pipeline compatible with streaming ingestion, where tasks arrive
over time and not all data exists up front.

This module also hosts the constants shared by the other converter stages and
small helpers (atomic JSON/text writers, path ordering) so downstream modules
can import them without circular dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


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


@dataclass(frozen=True)
class TaskUnit:
    """One (dataset, task) group of episodes — the pipeline's processing unit.

    All episodes share the output ``task_dir`` and the task instruction slug;
    per-episode fields (videos, view shapes, ...) stay on ``Episode``.
    """

    dataset_dir_name: str
    task_index: int
    task_dir: Path
    episodes: list[Episode]


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


def instruction_short(task_text: str) -> str:
    short = task_text.split("|", 1)[0].strip()
    if not short.endswith((".", "!", "?")):
        short += "."
    return short


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "task"


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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


def iter_task_units(
    input_dir: Path, output_dir: Path, args: argparse.Namespace
) -> Iterator[TaskUnit]:
    """Discover episodes and yield one TaskUnit per (dataset, task).

    Streaming-friendly: tasks are yielded as they are discovered (dataset by
    dataset, in arrival order), instead of building one global episode list.
    Within a dataset, units and episodes are sorted naturally so batch runs
    are deterministic; note that with multiple datasets the dataset-level
    files therefore list episodes dataset by dataset rather than globally
    interleaved.
    """
    view_specs = [(video_key_to_view_dir(args.main_view), args.main_view)]
    if not args.no_gripper_view:
        for gripper_view in args.gripper_views:
            view_specs.append((video_key_to_view_dir(gripper_view), gripper_view))
    used_task_dirs: set[str] = set()
    task_dir_names: dict[tuple[str, int], str] = {}
    remaining = args.max_episodes  # None means unlimited

    for dataset_dir in discover_dataset_dirs(input_dir):
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
        if not tasks:
            print(
                f"WARNING: {dataset_dir} has no meta/tasks.jsonl; using the directory name as "
                f"the task instruction ({default_instruction!r})."
            )

        units: dict[tuple[str, int], list[Episode]] = {}
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
            units.setdefault(task_key, []).append(
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

        for task_key, episodes in sorted(
            units.items(), key=lambda item: natural_key(item[1][0].task_dir)
        ):
            episodes.sort(key=lambda episode: natural_key(episode.episode_dir))
            if remaining is not None:
                if remaining <= 0:
                    return
                if len(episodes) > remaining:
                    print(
                        f"NOTE: limiting conversion to the first {args.max_episodes} "
                        f"episodes (--max-episodes)"
                    )
                    episodes = episodes[:remaining]
                    yield TaskUnit(
                        dataset_dir_name=dataset_dir.name,
                        task_index=task_key[1],
                        task_dir=episodes[0].task_dir,
                        episodes=episodes,
                    )
                    return
                remaining -= len(episodes)
            yield TaskUnit(
                dataset_dir_name=dataset_dir.name,
                task_index=task_key[1],
                task_dir=episodes[0].task_dir,
                episodes=episodes,
            )
