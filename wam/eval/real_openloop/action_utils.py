"""
Action normalization / denormalization utilities for FastWAM eval.

Ported from UniVLA2 model_wrapper_real.py (lines 200-457).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as _SciRot

import os.path as osp
import sys

_tools_dir = osp.join(osp.dirname(__file__), "tools")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from joint_action_mapping_norms import (
    load_joint_action_mapping_cache,
    get_default_mapping_entry,
    parse_field_norms,
    load_action_norm_bounds,
    load_joint_norm_min_delta,
)

logger = logging.getLogger(__name__)


def build_action_col_map(
    dataset_name: str,
    joint_action_mapping_dir: str,
) -> Optional[dict]:
    """Build {key: (start_col, dim)} mapping for 6D rotation fields.

    Mirrors EmuVLARealModel.__init__ lines 204-229.
    Returns None if no mapping exists for the dataset.
    """
    cache = load_joint_action_mapping_cache(joint_action_mapping_dir)
    mapping = cache.get(dataset_name)
    if not mapping:
        logger.warning(
            "[build_action_col_map] No mapping for dataset '%s'; "
            "6D→Euler conversion will be skipped.",
            dataset_name,
        )
        return None

    entry = get_default_mapping_entry(mapping)
    action_keys = entry.get("action_keys", [])
    nmd = entry["norm_min_delta"]
    col_map = {}
    col_offset = 0
    for key in action_keys:
        if key.endswith("_rotation"):
            dim = 6   # use_6d_rotation=True expands rotation to 6D
        else:
            mn, _ = parse_field_norms(key, nmd)
            dim = len(mn)
        col_map[key] = (col_offset, dim)
        col_offset += dim

    logger.info("[build_action_col_map] col_map: %s", col_map)
    return col_map


def load_joint_keys(
    dataset_name: str,
    joint_action_mapping_dir: str,
) -> list:
    """Return joint_keys list from the mapping file for a dataset."""
    cache = load_joint_action_mapping_cache(joint_action_mapping_dir)
    mapping = cache.get(dataset_name)
    if not mapping:
        logger.warning(
            "[load_joint_keys] No mapping for dataset '%s'; returning empty list.",
            dataset_name,
        )
        return []
    entry = get_default_mapping_entry(mapping)
    return entry.get("joint_keys") or []


def decode_6d_to_euler(
    action: np.ndarray,
    action_col_map: dict,
) -> np.ndarray:
    """Convert 6D rotation fields → Euler ZYX [roll, pitch, yaw].

    Gram-Schmidt orthogonalization ensures numerical stability.
    Ported from EmuVLARealModel._decode_6d_to_euler (lines 386-432).

    Args:
        action: [D_6d] or [T, D_6d] — denormalized action with 6D rotation fields
        action_col_map: {key: (start_col, dim)} mapping

    Returns:
        [D_euler] or [T, D_euler] — rotation fields shrunk from 6D to 3D
    """
    batched = action.ndim > 1
    if not batched:
        action = action[None]

    sorted_keys = sorted(action_col_map.items(), key=lambda kv: kv[1][0])
    n_rot6d = sum(1 for k, (_, d) in sorted_keys if k.endswith("_rotation") and d == 6)
    D_out = action.shape[1] - 3 * n_rot6d
    out = np.empty((action.shape[0], D_out), dtype=np.float32)
    out_ptr = 0

    for key, (start, dim) in sorted_keys:
        if key.endswith("_rotation") and dim == 6:
            block = action[:, start:start + 6].astype(np.float64)
            a_col = block[:, 0:3]
            b_col = block[:, 3:6]
            a_col = a_col / np.linalg.norm(a_col, axis=-1, keepdims=True).clip(min=1e-8)
            b_col = b_col - (b_col * a_col).sum(-1, keepdims=True) * a_col
            b_col = b_col / np.linalg.norm(b_col, axis=-1, keepdims=True).clip(min=1e-8)
            c_col = np.cross(a_col, b_col)
            R = np.stack([a_col, b_col, c_col], axis=-1)  # [T, 3, 3]
            euler_zyx = _SciRot.from_matrix(
                R.reshape(-1, 3, 3)
            ).as_euler("ZYX")  # [T, 3] = [yaw, pitch, roll]
            out[:, out_ptr:out_ptr + 3] = euler_zyx[:, ::-1].astype(np.float32)  # → [roll, pitch, yaw]
            out_ptr += 3
        else:
            out[:, out_ptr:out_ptr + dim] = action[:, start:start + dim]
            out_ptr += dim

    if not batched:
        out = out[0]
    return out


def denormalize_action(
    action_normed: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    *,
    use_6d_rotation: bool = False,
    action_col_map: Optional[dict] = None,
) -> np.ndarray:
    """Denormalize action from [-1, 1] to raw values.

    Formula: raw = 0.5 * (normed + 1) * (high - low) + low

    When use_6d_rotation=True, additionally converts 6D rotation fields
    to Euler ZYX [roll, pitch, yaw].

    Args:
        action_normed: [T, D] or [D] in [-1, 1]
        norm_low: [D] lower bound
        norm_high: [D] upper bound
        use_6d_rotation: whether to apply 6D→Euler conversion
        action_col_map: required when use_6d_rotation=True

    Returns:
        raw action [T, D_euler] or [T, D_6d] depending on use_6d_rotation
    """
    raw = (0.5 * (action_normed + 1.0) * (norm_high - norm_low) + norm_low).astype(np.float32)

    if use_6d_rotation and action_col_map is not None:
        raw = decode_6d_to_euler(raw, action_col_map)

    return raw


def relative_to_absolute(
    rel_actions: np.ndarray,
    follower_ref_dict: dict,
    action_col_map: dict,
) -> np.ndarray:
    """Convert denormalized relative actions (6D rotation) back to absolute.

    Inverse of Emu3SFTDataset._compute_relative_actions:
        pos_abs = pos_rel + pos_follow_ref
        R_abs   = R_rel @ R_follow_ref   (since R_rel = R_master @ R_follow^T)

    Must be called BEFORE decode_6d_to_euler — rotation fields are 6D in/out.

    Args:
        rel_actions:       [T, D] denormalized relative (rotation in 6D)
        follower_ref_dict: {str: list|float} raw follower state keyed by joint field name
                           (e.g. {'follow_left_position': [x,y,z],
                                  'follow_left_rotation': [roll,pitch,yaw], ...})
        action_col_map:    {action_key: (start_col, dim)} from build_action_col_map

    Returns:
        [T, D] absolute actions (rotation still in 6D)
    """
    out = rel_actions.copy().astype(np.float64)

    for action_key, (start, dim) in action_col_map.items():
        follow_key = action_key.replace('master', 'follow')
        if follow_key not in follower_ref_dict:
            continue

        ref_val = follower_ref_dict[follow_key]

        if '_position' in action_key:
            # pos_abs = pos_rel + pos_follow_ref
            out[:, start:start + 3] += np.array(ref_val, dtype=np.float64)

        elif '_rotation' in action_key:
            if dim != 6:
                logger.warning(
                    "[relative_to_absolute] rotation key '%s' has dim=%d (expected 6 for 6D); "
                    "skipping relative→absolute conversion for this field.",
                    action_key, dim,
                )
                continue
            # 1. Recover R_rel [T, 3, 3] from 6D (first two columns + cross product)
            col1 = out[:, start:start + 3]
            col2 = out[:, start + 3:start + 6]
            col3 = np.cross(col1, col2)                          # [T, 3]
            R_rel = np.stack([col1, col2, col3], axis=-1)        # [T, 3, 3]

            # 2. Build R_follow [3, 3] from follower Euler ZYX [roll, pitch, yaw]
            rpy = ref_val
            R_follow = _SciRot.from_euler(
                "ZYX", [rpy[2], rpy[1], rpy[0]]
            ).as_matrix()                                        # [3, 3]

            # 3. R_abs = R_rel @ R_follow
            R_abs = R_rel @ R_follow                             # [T, 3, 3]

            # 4. Write back as 6D (first two columns)
            out[:, start:start + 3]     = R_abs[:, :, 0]
            out[:, start + 3:start + 6] = R_abs[:, :, 1]

        # gripper / other: unchanged (already absolute)

    return out.astype(np.float32)
