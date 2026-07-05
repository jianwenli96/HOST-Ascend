# -*- coding: utf-8 -*-
"""
Shared joint_action_mapping.json → norm vectors for training (datasets.py) and eval (EmuVLARealModel).

Must stay mathematically aligned with train/datasets.py:
  parse_field_norms / _build_norm_vectors / joint flatten in _load_episode_raw.
"""
from __future__ import annotations

import json
import logging
import math
import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "load_joint_action_mapping_cache",
    "get_default_mapping_entry",
    "parse_field_norms",
    "build_norm_vectors",
    "build_joint_norm_vectors",
    "load_action_norm_bounds",
    "load_joint_norm_min_delta",
]


def load_joint_action_mapping_cache(joint_action_mapping_dir: str) -> Dict[str, Any]:
    """Eagerly load all *_joint_action_mapping.json files.

    Returns:
        {dataset_name: mapping_dict}  (same as CustomDataset._joint_action_mapping_cache)
    """
    cache: Dict[str, Any] = {}
    if not osp.isdir(joint_action_mapping_dir):
        logging.warning(
            "[joint_action_mapping_norms] joint_action_mapping_dir not found: %s",
            joint_action_mapping_dir,
        )
        return cache
    for fname in os.listdir(joint_action_mapping_dir):
        if not fname.endswith("_joint_action_mapping.json"):
            continue
        ds = fname[: -len("_joint_action_mapping.json")]
        fpath = osp.join(joint_action_mapping_dir, fname)
        try:
            with open(fpath, "r") as f:
                cache[ds] = json.load(f)
            logging.info(
                "[joint_action_mapping_norms] Loaded joint_action_mapping for '%s'", ds
            )
        except Exception as e:
            logging.warning(
                "[joint_action_mapping_norms] Failed to load %s: %s", fpath, e
            )
    return cache


def get_default_mapping_entry(mapping: dict) -> dict:
    """Same as RealEpisodeDataset._build_action_norms: first entry in the file."""
    if not mapping:
        raise RuntimeError(
            "[joint_action_mapping_norms] mapping dict is empty — cannot pick entry."
        )
    return next(iter(mapping.values()))


def parse_field_norms(field_key: str, nmd: dict) -> Tuple[list, list]:
    if field_key not in nmd:
        raise RuntimeError(
            f"[parse_field_norms] '{field_key}' missing from norm_min_delta"
        )
    stats = nmd[field_key]
    min_vals = stats["min"]
    delta_vals = stats["delta"]
    if any(
        v == "nan" or (isinstance(v, float) and math.isnan(v))
        for v in min_vals + delta_vals
    ):
        raise RuntimeError(
            f"[parse_field_norms] 'nan' found in norm_min_delta for field '{field_key}'. "
            "Provide valid statistics before using this dataset."
        )
    return list(min_vals), list(delta_vals)


def build_norm_vectors(
    action_keys,
    nmd,
    *,
    ds_factor: int = 1,
    use_relative: bool = False,
    use_6d_rotation: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    nm_list, nd_list = [], []
    for key in action_keys:
        if use_relative:
            if use_6d_rotation and key.endswith("_rotation"):
                # Relative 6D rotation: values inherently in [-1, 1]
                nm_list.extend([-1.0] * 6)
                nd_list.extend([2.0] * 6)
            elif "_position" in key:
                # Position: use _relative suffix norms (world-frame delta range)
                rel_key = key + "_relative"
                abs_mn, _ = parse_field_norms(key, nmd)
                dim = len(abs_mn)
                try:
                    mn, nd = parse_field_norms(rel_key, nmd)
                except Exception:
                    logging.warning(
                        "[build_norm_vectors] '%s' not in nmd; "
                        "falling back to passthrough normalisation.",
                        rel_key,
                    )
                    mn, nd = [-1.0] * dim, [2.0] * dim
                scale = ds_factor
                nm_list.extend(v * scale for v in mn)
                nd_list.extend(v * scale for v in nd)
            elif "gripper" in key:
                # Gripper: values are absolute (not converted to relative),
                # use absolute norms.
                mn, nd = parse_field_norms(key, nmd)
                nm_list.extend(mn)
                nd_list.extend(nd)
            else:
                logging.warning(
                    "[build_norm_vectors] unexpected key '%s' in relative mode; "
                    "falling back to absolute norms.",
                    key,
                )
                mn, nd = parse_field_norms(key, nmd)
                nm_list.extend(mn)
                nd_list.extend(nd)
        else:
            if use_6d_rotation and key.endswith("_rotation"):
                nm_list.extend([-1.0] * 6)
                nd_list.extend([2.0] * 6)
            else:
                mn, nd = parse_field_norms(key, nmd)
                nm_list.extend(mn)
                nd_list.extend(nd)
    return np.array(nm_list, dtype=np.float32), np.array(nd_list, dtype=np.float32)


def build_joint_norm_vectors(
    joint_keys: list, nmd: dict, *, use_6d_rotation: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not joint_keys:
        return None, None
    j_norm_min, j_norm_delta = [], []
    for key in joint_keys:
        if use_6d_rotation and key.endswith("_rotation"):
            # 6D rotation: identity norm (values already in [-1, 1])
            j_norm_min.extend([-1.0] * 6)
            j_norm_delta.extend([2.0] * 6)
        else:
            min_vals, delta_vals = parse_field_norms(key, nmd)
            j_norm_min.extend(min_vals)
            j_norm_delta.extend(delta_vals)
    j_nm = np.array(j_norm_min, dtype=np.float32)
    j_nd = np.array(j_norm_delta, dtype=np.float32)
    return j_nm, j_nd


def load_action_norm_bounds(
    dataset_name: str,
    joint_action_mapping_dir: str,
    *,
    use_6d_rotation: bool,
    use_relative_action: bool,
    ds_factor: int = 1,
    mapping_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(norm_low, norm_high) with norm_high = nm + nd."""
    if mapping_cache is None:
        mapping_cache = load_joint_action_mapping_cache(joint_action_mapping_dir)
    mapping = mapping_cache.get(dataset_name)
    if not mapping:
        raise RuntimeError(
            f"[load_action_norm_bounds] No joint_action_mapping for dataset "
            f"'{dataset_name}'. Check joint_action_mapping_dir."
        )
    entry = get_default_mapping_entry(mapping)
    nm, nd = build_norm_vectors(
        entry["action_keys"],
        entry["norm_min_delta"],
        ds_factor=ds_factor,
        use_relative=use_relative_action,
        use_6d_rotation=use_6d_rotation,
    )
    return nm, nm + nd


def load_joint_norm_min_delta(
    dataset_name: str,
    joint_action_mapping_dir: str,
    *,
    use_6d_rotation: bool = False,
    mapping_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(j_nm, j_nd) or (None, None) if joint_keys empty."""
    if mapping_cache is None:
        mapping_cache = load_joint_action_mapping_cache(joint_action_mapping_dir)
    mapping = mapping_cache.get(dataset_name)
    if not mapping:
        raise RuntimeError(
            f"[load_joint_norm_min_delta] No joint_action_mapping for dataset "
            f"'{dataset_name}'. Check joint_action_mapping_dir."
        )
    entry = get_default_mapping_entry(mapping)
    joint_keys = entry.get("joint_keys") or []
    return build_joint_norm_vectors(
        joint_keys, entry["norm_min_delta"], use_6d_rotation=use_6d_rotation
    )
