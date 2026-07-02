"""
Open-loop evaluation dataset for FastWAM.

Inherits all file-loading helpers from Emu3SFTDataset (mydatasets.py).
Does NOT call super().__init__() — that requires training-specific args
(tokenizer, DataArguments, etc.).

Design mirrors UniVLA2 eval/real_openloop/real_dataset.py.
"""
from __future__ import annotations

import logging
import os.path as osp
import sys
from typing import Optional

import numpy as np

# Ensure project src is importable
_project_src = osp.join(osp.dirname(__file__), "..", "..", "src")
if _project_src not in sys.path:
    sys.path.insert(0, _project_src)

from fastwam.datasets.custom.mydatasets import (
    Emu3SFTDataset,
    ACTION_DOWNSAMPLE_FACTOR,
)

logger = logging.getLogger(__name__)


class FastWAMEvalDataset(Emu3SFTDataset):
    """Per-episode dataset that loads GT actions, frame paths, and joint data.

    Returns raw (unnormalized) data for comparison against model predictions.
    No tokenization or video tensor construction is performed here — that
    happens inside the model wrapper.
    """

    def __init__(
        self,
        data_path: str,
        views: list,
        joint_action_mapping_dir: str,
        dataset_name: str,
        *,
        use_6d_rotation: bool = False,
        use_relative_action: bool = False,
        dataset_image_size: Optional[dict] = None,
        context_len: int = 128,
        action_downsample_factor: Optional[dict] = None,
        cam_mapping_dir: str = "/open_data/cgy/cam_mapping/2views",
        dataset_fps: Optional[dict] = None,
        task_max_frames: Optional[dict] = None,
        remove_static_frames: bool = False,
        static_rot_threshold: float = 0.01,
        static_trans_threshold: float = 0.01,
        static_gripper_threshold: float = 0.01,
    ):
        """
        Args:
            data_path: Comma-separated JSON path list file(s), same format
                       as training --data_path. Each JSON is a list of episode
                       directory paths.
            views:     Camera view names (e.g. ['cam_high_rgb', 'cam_wrist_left_rgb']).
            joint_action_mapping_dir: Directory with *_joint_action_mapping.json files.
            dataset_name: Dataset id (e.g. '10042') for norm lookup.
            use_6d_rotation: Match training config (Euler → 6D rotation conversion).
            use_relative_action: Match training config.
            dataset_image_size: Per-dataset {name: (W, H)} image size dict.
            context_len: T5 text embedding length.
            action_downsample_factor: Per-dataset temporal stride dict.
            cam_mapping_dir: Directory with cam_mapping JSON files.
        """
        # ---- Minimal state required by inherited helpers ----
        self.data_format      = "json"
        if dataset_image_size is None:
            raise ValueError("dataset_image_size must be provided (from config)")
        self.dataset_image_size = {}
        for k, v in dataset_image_size.items():
            self.dataset_image_size[str(k)] = (int(v[1]), int(v[0]))  # store as (W, H)

        self.views              = views
        self.use_6d_rotation    = use_6d_rotation
        self.use_relative_action = use_relative_action
        self.context_len        = context_len

        # ---- Init mapping caches (required by _load_action_and_joint) ----
        self.joint_action_mapping_dir = joint_action_mapping_dir
        self._init_joint_action_mapping()

        # ---- Init cam_mapping (required by _construct_json_item view resolution) ----
        self.cam_mapping_dir = cam_mapping_dir
        self._init_cam_mapping()

        # ---- Attrs required by _construct_json_item ----
        self.dataset_fps              = dataset_fps if dataset_fps is not None else {}
        if task_max_frames is None:
            raise ValueError("task_max_frames must be provided (from config)")
        self.task_max_frames          = {int(k): list(v) for k, v in task_max_frames.items()}
        # Sampling interval multipliers (required by inherited methods, not used in eval)
        self.sampling_interval_min_mult = 0.5
        self.sampling_interval_max_mult = 2.0
        self.task_paths_filename = "task_paths.json"
        self.task_video_drop_prob = 0.0
        self.task_text_drop_prob = 0.0
        self.actions                  = True
        self.joints                   = True
        self.skip_no_action           = True
        self.remove_static_frames     = remove_static_frames
        self.static_rot_threshold     = static_rot_threshold
        self.static_trans_threshold   = static_trans_threshold
        self.static_gripper_threshold = static_gripper_threshold
        if remove_static_frames:
            logging.warning(
                "[FastWAMEvalDataset] remove_static_frames=True: GT will be filtered "
                "with thresholds rot=%.4f, trans=%.4f, grip=%.4f",
                static_rot_threshold, static_trans_threshold, static_gripper_threshold,
            )
        self.use_indicator_prompt     = False
        self.use_indicator_positive   = False
        self.raw_image                = True
        self.num_view_probs           = {len(views): 1.0}
        self.action_downsample_factor = (
            action_downsample_factor if action_downsample_factor is not None
            else ACTION_DOWNSAMPLE_FACTOR
        )

        # ---- Path transform ----
        self._path_transforms = {}

        # ---- Populate episode list ----
        self.data = self._load_from_json(data_path)
        logger.info("[FastWAMEvalDataset] Loaded %d episodes from %s", len(self.data), data_path)

        # ---- Build norm_low / norm_high ----
        self.dataset_name = dataset_name
        _ds_factor = int((self.action_downsample_factor or {}).get(dataset_name, 1))
        self.norm_low, self.norm_high = self._build_action_norms(dataset_name, _ds_factor)
        logger.info("[FastWAMEvalDataset] action norm_low shape=%s, norm_high shape=%s",
                     self.norm_low.shape, self.norm_high.shape)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _build_action_norms(self, dataset_name: str, ds_factor: int = 1):
        """Compute (norm_low, norm_high) from mapping JSON.

        Identical to UniVLA2 RealEpisodeDataset._build_action_norms.
        """
        mapping = self._load_joint_action_mapping(dataset_name)
        if not mapping:
            raise RuntimeError(
                f"[FastWAMEvalDataset] No mapping for dataset '{dataset_name}'. "
                f"Check joint_action_mapping_dir={self.joint_action_mapping_dir}"
            )
        entry = next(iter(mapping.values()))
        nm, nd = self._build_norm_vectors(
            entry["action_keys"],
            entry["norm_min_delta"],
            ds_factor=ds_factor,
            use_relative=self.use_relative_action,
            use_6d_rotation=self.use_6d_rotation,
        )
        return nm, nm + nd   # norm_low, norm_high

    def _unnormalize_actions(self, normed: np.ndarray) -> Optional[np.ndarray]:
        """Reverse normalization: raw = 0.5 * (normed + 1) * (high - low) + low."""
        if normed is None:
            return None
        D = normed.shape[-1]
        if D != len(self.norm_low):
            raise ValueError(
                f"Action dim mismatch: normed has {D} cols but norm_low has {len(self.norm_low)}"
            )
        return (
            0.5 * (normed + 1.0) * (self.norm_high - self.norm_low) + self.norm_low
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        """Return per-episode data without tokenization.

        Uses _construct_json_item (inherited) with forced_views for
        consistent downsampling and frame filtering.

        Returns:
            dict:
                episode_dir        (str)
                dataset_name       (str)
                task_description   (str)
                frame_files        (dict[str, list[str]])  view → file paths
                gt_actions_raw     (np.ndarray [T, D])     unnormalized
                gt_actions_normed  (np.ndarray [T, D])     [-1, 1]
                joint_data         (np.ndarray [T, D_joint] or None)  raw joints
        """
        item = self._construct_json_item(i, forced_views=self.views)

        video_info = dict(self.data[i])
        dataset_name = video_info.get("dataset", "")

        _ep_raw = item.get("_ep_raw")

        # item["action"] is now raw unnormalized (normalization moved to
        # _get_single_item in training). Normalize GT with absolute norms here.
        raw_actions = item["action"]
        if raw_actions is not None and _ep_raw is not None:
            a_nm, a_nd = _ep_raw['a_nm'], _ep_raw['a_nd']
            gt_actions_normed = np.clip(
                2.0 * (raw_actions.astype(np.float32) - a_nm) / a_nd - 1.0,
                -1.0, 1.0,
            )
            gt_actions_raw = raw_actions.astype(np.float32)
        else:
            gt_actions_normed = None
            gt_actions_raw = None

        views_list = item["_selected_views"]
        frame_files = {v: item["views_images"][vi] for vi, v in enumerate(views_list)}

        joint_out = None
        joint_entries = []
        if _ep_raw is not None and _ep_raw.get("raw_joints") is not None:
            joint_out = _ep_raw["raw_joints"].astype(np.float32)
            joint_entries = _ep_raw.get("raw_joint_entries", [])

        # Load raw progress list from info_dtw.json (or info.json fallback)
        _primary_view_files = list(frame_files.values())[0]
        progress_list = self.load_progress_info(_primary_view_files, dataset_name)

        return {
            "episode_dir":       video_info["video_dir"],
            "dataset_name":      dataset_name,
            "task_description":  item["text"],
            "frame_files":       frame_files,
            "gt_actions_raw":    gt_actions_raw,
            "gt_actions_normed": gt_actions_normed,
            "joint_data":        joint_out,
            "joint_entries":     joint_entries,
            "progress_list":     progress_list,
        }

    def load_frame_numpy(self, path: str) -> Optional[np.ndarray]:
        """Load a single frame as (H, W, 3) uint8 numpy array.

        Used by the eval loop to build observation dicts.
        """
        try:
            target_size = self.dataset_image_size.get(
                self.dataset_name, (224, 224)
            )
            pil = self._load_frame(path, target_size=target_size)
            if pil is None:
                return None
            return np.array(pil, dtype=np.uint8)
        except Exception as e:
            logging.warning("[load_frame_numpy] Failed for %s: %s", path, e)
            return None
