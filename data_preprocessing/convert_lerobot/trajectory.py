"""Stage C of the LeRobot converter: trajectory conversion and normalization.

Loads the bimanual action/proprioception columns from each episode Parquet,
converts the xyzw quaternions to HOST [roll, pitch, yaw] Euler triples, and
writes the per-episode trajectory JSON.  ``NormAccumulator`` maintains the
dataset-wide min/delta normalization statistics as a running min/max so the
joint/action mapping file can be rewritten after each task (streaming-safe,
no all-episodes-first pass).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .source import (
    ACTION_KEYS,
    ACTION_SOURCE_COLUMNS,
    JOINT_KEYS,
    JOINT_SOURCE_COLUMN,
    QUATERNION_ORDER,
    TRAJECTORY_KEYS,
    Episode,
    write_json_atomic,
)


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


class NormAccumulator:
    """Running min/max over trajectory fields, streamed across tasks.

    Equivalent to the previous two-pass ``trajectory_norm_min_delta`` (min and
    max are exact and order-independent, so accumulated results are identical),
    but does not require every episode to be present up front.
    """

    def __init__(self) -> None:
        self._min: dict[str, np.ndarray] = {}
        self._max: dict[str, np.ndarray] = {}

    def update(self, trajectory: dict[str, np.ndarray]) -> None:
        for key, values in trajectory.items():
            current_min = values.min(axis=0)
            current_max = values.max(axis=0)
            self._min[key] = (
                current_min if key not in self._min else np.minimum(self._min[key], current_min)
            )
            self._max[key] = (
                current_max if key not in self._max else np.maximum(self._max[key], current_max)
            )

    def finalize(self) -> dict[str, dict[str, list[float]]]:
        result = {}
        for key in TRAJECTORY_KEYS:
            if key not in self._min:
                continue
            delta = self._max[key] - self._min[key]
            if np.any(delta <= 0):
                raise ValueError(
                    f"Trajectory field {key!r} has non-positive normalization delta: {delta}"
                )
            result[key] = {
                "min": self._min[key].tolist(),
                "delta": delta.tolist(),
            }
        return result


def norm_mapping_path(output_dir: Path, dataset_id: str) -> Path:
    return output_dir / "joint_action_mapping" / f"{dataset_id}_joint_action_mapping.json"


def write_episode_trajectory(episode: Episode, trajectory: dict[str, np.ndarray]) -> None:
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


def write_norm_mapping(
    output_dir: Path, dataset_id: str, norms: dict[str, dict[str, list[float]]]
) -> Path:
    """(Re)write the joint/action mapping with the stats accumulated so far."""
    mapping_file = norm_mapping_path(output_dir, dataset_id)
    mapping = {
        str(output_dir.resolve()): {
            "action_keys": [key for key in ACTION_KEYS if key in norms],
            "joint_keys": [key for key in JOINT_KEYS if key in norms],
            "norm_min_delta": norms,
        }
    }
    write_json_atomic(mapping_file, mapping)
    return mapping_file


def verify_robot_trajectory_data(
    episodes: list[Episode], output_dir: Path, dataset_id: str
) -> None:
    mapping_file = norm_mapping_path(output_dir, dataset_id)
    with mapping_file.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if len(mapping) != 1:
        raise ValueError(f"Expected one mapping entry in {mapping_file}, got {len(mapping)}")
    entry = next(iter(mapping.values()))
    accumulator = NormAccumulator()
    for episode in episodes:
        accumulator.update(load_trajectory(episode.source))
    expected_norms = accumulator.finalize()
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
