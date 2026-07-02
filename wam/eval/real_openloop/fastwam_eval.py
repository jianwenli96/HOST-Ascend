"""
FastWAM model wrapper for real open-loop evaluation.

Loads model from checkpoint_dir/config.yaml + checkpoint .pt file.
All config params (cam_mapping, normalization, image pixels, etc.) are read
from the saved training config — only a few essential params are external.

Design mirrors UniVLA2 EmuVLARealModel but adapted for FastWAM's
diffusion-based infer_joint interface (pixel tensors, not VQ tokens).
"""
from __future__ import annotations

import glob
import json
import logging
import os
import os.path as osp
import sys
import textwrap
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure project src is importable
_project_src = osp.join(osp.dirname(__file__), "..", "..", "src")
if _project_src not in sys.path:
    sys.path.insert(0, _project_src)

from omegaconf import OmegaConf
from hydra.utils import instantiate

from action_utils import (
    build_action_col_map,
    decode_6d_to_euler,
    denormalize_action,
    load_action_norm_bounds,
    load_joint_keys,
    load_joint_norm_min_delta,
    relative_to_absolute,
)
from fastwam.datasets.custom.mydatasets import Emu3SFTDataset
from fastwam.utils.video_io import save_mp4
from scipy.spatial.transform import Rotation as _SciRotation
from action_viz import save_action_plot

logger = logging.getLogger(__name__)


def _vstack_pil(imgs):
    """Vertically stack a list of PIL images into one."""
    total_h = sum(im.size[1] for im in imgs)
    w = imgs[0].size[0]
    out = Image.new("RGB", (w, total_h))
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.size[1]
    return out


def _update_latest_symlink(target_path: str, latest_name: str):
    """Atomically update a 'latest_*' symlink in the same directory."""
    d = os.path.dirname(target_path)
    latest = os.path.join(d, latest_name)
    tmp = latest + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.basename(target_path), tmp)
    os.replace(tmp, latest)


class FastWAMEval:
    """Stateful model wrapper for real open-loop evaluation.

    Loads model + normalization from a training checkpoint directory.
    Provides infer_real() as the single entry point for raw sensor data → action.

    Usage:
        wrapper = FastWAMEval(checkpoint_dir, dataset_name='10042', views=[...])
        wrapper.set_task_video(episode_dir)   # once per task video episode
        for step in range(num_steps):
            result = wrapper.infer_real(obs, joints, instruction)
            # result["action_raw"]: denormalized action
    """

    def __init__(
        self,
        checkpoint_dir: str,
        dataset_name: str,
        views: list[str],
        *,
        checkpoint_step: Optional[int] = None,
        diffusion_steps: int = 20,
        use_task_video: bool = True,
        use_task_description: bool = True,
        device: str = "cuda",
        remove_static_frames: bool = False,
        static_rot_threshold: float = 0.01,
        static_trans_threshold: float = 0.01,
        static_gripper_threshold: float = 0.01,
        task_video_static_threshold_ratio: float = 0.5,
    ):
        """
        Args:
            checkpoint_dir: Training output dir containing config.yaml + checkpoints/weights/
            dataset_name: Dataset id (e.g. '10042') for norm lookup. Passed from eval script.
            views: Camera view names. Passed from eval script.
            checkpoint_step: Specific step to load (None = latest).
            diffusion_steps: Number of diffusion denoising steps for inference.
            use_task_video: If True, task_video must be provided; if False, run without.
            use_task_description: If True (and use_task_video=True), load text embedding;
                                 if False, zero out context (tv_only mode).
            device: CUDA device.
        """
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.views = views
        self.use_task_video = use_task_video
        self.use_task_description = use_task_description
        self.dataset_name = dataset_name
        self.remove_static_frames = remove_static_frames
        self.static_rot_threshold = static_rot_threshold
        self.static_trans_threshold = static_trans_threshold
        self.static_gripper_threshold = static_gripper_threshold
        self.task_video_static_threshold_ratio = task_video_static_threshold_ratio

        # ---- Load training config ----
        config_path = osp.join(checkpoint_dir, "config.yaml")
        if not osp.exists(config_path):
            raise FileNotFoundError(f"config.yaml not found in {checkpoint_dir}")
        self.config = OmegaConf.load(config_path)
        cfg = self.config

        # ---- Extract data params from config ----
        dcfg = cfg.data.train
        self.joint_action_mapping_dir = dcfg.joint_action_mapping_dir
        self.cam_mapping_dir          = dcfg.cam_mapping_dir
        self.use_6d_rotation          = bool(dcfg.use_6d_rotation)
        self.use_relative_action      = bool(dcfg.use_relative_action)
        # Per-dataset image size: (W, H)
        cfg_img_size = getattr(dcfg, 'dataset_image_size', None)
        if cfg_img_size is None:
            raise ValueError(
                "config.yaml missing 'dataset_image_size'. "
                "This checkpoint was trained with an old config. "
                "Add dataset_image_size to data.train in config.yaml."
            )
        self.dataset_image_size = {}
        for k, v in cfg_img_size.items():
            self.dataset_image_size[str(k)] = (int(v[1]), int(v[0]))  # store as (W, H)
        if dataset_name not in self.dataset_image_size:
            raise ValueError(
                f"dataset '{dataset_name}' not in dataset_image_size config. "
                f"Available: {list(self.dataset_image_size.keys())}"
            )
        self.image_target_size = self.dataset_image_size[dataset_name]  # (W, H)
        self.action_dim               = int(dcfg.processor.action_output_dim)
        self.proprio_dim              = int(dcfg.processor.proprio_output_dim)
        self.action_frames            = int(dcfg.action_frames)   # default, overridden per-dataset
        self.max_action_len           = int(dcfg.max_action_len)
        self.context_len              = int(dcfg.context_len)
        self.frames                   = int(dcfg.frames)
        self.action_video_freq_ratio  = int(dcfg.action_video_freq_ratio)

        # Per-dataset FPS: override action_frames for the eval dataset
        if not hasattr(dcfg, 'dataset_fps') or dcfg.dataset_fps is None:
            raise ValueError(
                "config.yaml missing 'dataset_fps'. "
                "This checkpoint was trained with an old config. "
                "Add dataset_fps to data.train in config.yaml."
            )
        self.dataset_fps = {str(k): int(v) for k, v in dcfg.dataset_fps.items()}

        # Task max frames per num_views
        if not hasattr(dcfg, 'task_max_frames') or dcfg.task_max_frames is None:
            raise ValueError(
                "config.yaml missing 'task_max_frames'. "
                "Add task_max_frames to data.train in config.yaml."
            )
        self.task_max_frames = {int(k): list(v) for k, v in dcfg.task_max_frames.items()}

        if dataset_name not in self.dataset_fps:
            raise ValueError(
                f"dataset_name '{dataset_name}' not found in config dataset_fps. "
                f"Available: {list(self.dataset_fps.keys())}"
            )
        self.action_frames = self.dataset_fps[dataset_name]
        # Compute frames_per_seg for VAE-compatible downsampling (mirrors training)
        _raw = max(1, round(self.action_frames / self.action_video_freq_ratio))
        self.n_per_seg = max(4, ((_raw + 3) // 4) * 4) + 1  # ∈ {5, 9, 13, ...}
        self.frames_per_seg = self.n_per_seg - 1  # e.g. 4
        logger.info(
            "[FastWAMEval] action_frames=%d from dataset_fps[%s], n_per_seg=%d",
            self.action_frames, dataset_name, self.n_per_seg,
        )

        # Pre-load action_keys for static frame detection
        if self.remove_static_frames:
            _mp = osp.join(self.joint_action_mapping_dir,
                           f"{dataset_name}_joint_action_mapping.json")
            with open(_mp) as _f:
                self._static_action_keys = next(iter(json.load(_f).values()))["action_keys"]

        logger.info(
            "[FastWAMEval] Config: action_dim=%d, proprio_dim=%d, use_6d_rotation=%s, "
            "action_frames=%d, max_action_len=%d, frames=%d",
            self.action_dim, self.proprio_dim, self.use_6d_rotation,
            self.action_frames, self.max_action_len, self.frames,
        )

        # ---- Build model ----
        # Override load_text_encoder=true for eval (training uses false to save memory)
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        model_cfg["load_text_encoder"] = True
        logger.info("[FastWAMEval] Building model (load_text_encoder=True for eval)...")
        self.model = instantiate(
            model_cfg,
            model_dtype=torch.bfloat16,
            device=device,
        )

        # ---- Load checkpoint weights ----
        ckpt_path = self._find_checkpoint(checkpoint_dir, checkpoint_step)
        logger.info("[FastWAMEval] Loading checkpoint: %s", ckpt_path)
        self.model.load_checkpoint(ckpt_path)
        self.model.eval()

        # ---- Normalization params ----
        self.norm_low, self.norm_high = load_action_norm_bounds(
            dataset_name,
            self.joint_action_mapping_dir,
            use_6d_rotation=self.use_6d_rotation,
            use_relative_action=self.use_relative_action,
        )
        self.j_nm, self.j_nd = load_joint_norm_min_delta(
            dataset_name,
            self.joint_action_mapping_dir,
            use_6d_rotation=self.use_6d_rotation,
        )
        self._joint_keys = load_joint_keys(
            dataset_name, self.joint_action_mapping_dir
        )
        logger.info(
            "[FastWAMEval] Norm loaded: action_dim=%d, joint_dim=%s, joint_keys=%s",
            len(self.norm_low),
            len(self.j_nm) if self.j_nm is not None else "None",
            self._joint_keys,
        )

        # ---- 6D rotation col_map (for denormalization) ----
        self._action_col_map = None
        if self.use_6d_rotation:
            self._action_col_map = build_action_col_map(
                dataset_name, self.joint_action_mapping_dir
            )

        # ---- Task video state ----
        self.task_video_loaded = False
        self._task_frames = None          # list[list[PIL.Image]] per view
        self.task_total_frames = None
        self.predicted_task_frame_idx = None
        self.current_sampled_indices = None

        # ---- Cached context (encode_prompt result) ----
        self._cached_context = None       # (context, context_mask) tuple
        self._cached_prompt = None        # the prompt string that was encoded

        # ---- Async visualization ----
        self._viz_executor = ThreadPoolExecutor(max_workers=2)
        self._viz_futures: list[Future] = []

        logger.info("[FastWAMEval] Initialized successfully.")

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_checkpoint(checkpoint_dir: str, step: Optional[int] = None) -> str:
        """Find checkpoint .pt file in checkpoint_dir/checkpoints/weights/."""
        weights_dir = osp.join(checkpoint_dir, "checkpoints", "weights")
        if not osp.isdir(weights_dir):
            raise FileNotFoundError(
                f"Weights directory not found: {weights_dir}. "
                f"Expected structure: {{checkpoint_dir}}/checkpoints/weights/*.pt"
            )
        if step is not None:
            path = osp.join(weights_dir, f"step_{step:06d}.pt")
            if not osp.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return path

        # Find latest by step number
        pts = sorted(glob.glob(osp.join(weights_dir, "step_*.pt")))
        if not pts:
            raise FileNotFoundError(f"No step_*.pt checkpoints in {weights_dir}")
        return pts[-1]

    # ------------------------------------------------------------------
    # Reset (for new episode)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset task video state for a new episode."""
        self._drain_viz_futures()
        self.task_video_loaded = False
        self._task_frames = None
        self._task_raw_indices = None  # filtered idx → raw frame idx mapping
        self.task_total_frames = None
        self.predicted_task_frame_idx = None
        self.current_sampled_indices = None
        self.current_video_indices = None
        self._cached_context = None
        self._cached_prompt = None

    def _drain_viz_futures(self):
        """Non-blocking cleanup: log exceptions from completed viz futures."""
        remaining = []
        for fut in self._viz_futures:
            if fut.done():
                exc = fut.exception()
                if exc is not None:
                    logger.warning("[viz-async] Visualization task failed: %s", exc)
            else:
                remaining.append(fut)
        self._viz_futures = remaining

    def flush_viz(self):
        """Block until all pending visualization futures complete."""
        for fut in self._viz_futures:
            try:
                fut.result()
            except Exception as e:
                logger.warning("[viz-async] Visualization task failed: %s", e)
        self._viz_futures.clear()

    def __del__(self):
        try:
            self._viz_executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Task video: load + window + tensor
    # ------------------------------------------------------------------

    def set_task_video(self, episode_dir: str):
        """Load all frames of a task video episode. Call once per episode.

        Mirrors EmuVLARealModel.load_task_video_paths (lines 492-588)
        but stores PIL frames instead of VQ tokens.
        """
        self.reset()

        views_frames = []
        ref_count = None
        for view in self.views:
            # Discover frame files (mp4 or image dir)
            mp4_path = osp.join(episode_dir, f"{view}.mp4")
            img_dir = osp.join(episode_dir, view)

            pil_frames = []
            if osp.exists(mp4_path):
                import decord
                decord.bridge.set_bridge("native")
                vr = decord.VideoReader(mp4_path, num_threads=1)
                for i in range(len(vr)):
                    frame = vr[i].asnumpy()  # [H, W, C] uint8
                    pil_frames.append(Image.fromarray(frame))
                del vr
            elif osp.exists(img_dir) and osp.isdir(img_dir):
                import glob as _glob
                files = sorted(
                    _glob.glob(osp.join(img_dir, "*.jpg")) +
                    _glob.glob(osp.join(img_dir, "*.png")),
                    key=lambda f: osp.basename(f),
                )
                for f in files:
                    pil_frames.append(Image.open(f).convert("RGB"))
            else:
                raise FileNotFoundError(
                    f"No video data for view '{view}' in {episode_dir}. "
                    f"Expected {mp4_path} or {img_dir}/"
                )

            if ref_count is None:
                ref_count = len(pil_frames)
            elif len(pil_frames) != ref_count:
                raise ValueError(
                    f"Frame count mismatch: view '{view}' has {len(pil_frames)} "
                    f"frames but expected {ref_count}"
                )

            # Resize to per-dataset target size
            w_t, h_t = self.image_target_size
            pil_frames = [f.resize((w_t, h_t), Image.BILINEAR) for f in pil_frames]
            views_frames.append(pil_frames)

        self._task_frames = views_frames
        self.task_total_frames = ref_count
        self.task_video_loaded = True
        self.predicted_task_frame_idx = None
        logger.info(
            "[set_task_video] Loaded %d frames x %d views from %s",
            ref_count, len(self.views), episode_dir,
        )

        # Filter static frames from task video (matches agent-side filtering)
        if self.remove_static_frames:
            ep_name = osp.basename(episode_dir.rstrip('/'))
            json_path = osp.join(episode_dir, f"{ep_name}.json")
            if osp.exists(json_path):
                with open(json_path) as _f:
                    entries = json.load(_f)["data"]
                raw_actions, col_map, active_keys = Emu3SFTDataset._assemble_raw_actions(
                    entries, self._static_action_keys, self.use_6d_rotation,
                )
                if raw_actions is not None:
                    _r = self.task_video_static_threshold_ratio
                    active_indices = Emu3SFTDataset._get_active_indices(
                        raw_actions, active_keys, col_map,
                        self.static_rot_threshold * _r,
                        self.static_trans_threshold * _r,
                        threshold_gripper=self.static_gripper_threshold * _r,
                    )
                    if len(active_indices) < self.task_total_frames:
                        self._task_frames = [
                            [vf[i] for i in active_indices] for vf in self._task_frames
                        ]
                        self._task_raw_indices = list(active_indices)
                        logger.info(
                            "[set_task_video] remove_static: %d → %d frames",
                            self.task_total_frames, len(active_indices),
                        )
                        self.task_total_frames = len(active_indices)
            else:
                logger.warning(
                    "[set_task_video] remove_static_frames=True but no JSON at %s",
                    json_path,
                )

    def _get_task_video_window(self, max_frames: int) -> Optional[torch.Tensor]:
        """Sample a window from loaded task video and build tensor.

        Window positioning mirrors EmuVLAModel.get_task_video_window (lines 645-703).
        After keyframe sampling, applies _downsample_segments_to_vae_indices to
        produce VAE-compatible video frame indices (matching training pipeline).

        Returns [C, T_task, H_total, W] tensor in [-1, 1], or None.
        """
        if not self.task_video_loaded or self._task_frames is None:
            return None

        total = self.task_total_frames
        step = self.action_frames

        # Determine start frame
        if self.predicted_task_frame_idx is not None:
            start_frame, step = self._frame_to_window_center_impl(
                self.predicted_task_frame_idx, total, max_frames
            )
            self.predicted_task_frame_idx = None
        else:
            start_frame = 0

        # Sample keyframe indices (mirrors get_task_video_window lines 692-693)
        sampled_indices = [
            min(start_frame + i * step, total - 1)
            for i in range(max_frames)
        ]
        self.current_sampled_indices = sampled_indices

        # Downsample keyframes → VAE-compatible video frame indices (mirrors training)
        video_indices = self._downsample_segments_to_vae_indices(
            sampled_indices, self.action_frames, self.action_video_freq_ratio, total
        )
        self.current_video_indices = video_indices

        # Build tensor from PIL frames at video indices
        views_np = []
        for v_frames in self._task_frames:
            frames_np = [np.array(v_frames[idx], dtype=np.uint8) for idx in video_indices]
            views_np.append(frames_np)

        return self._frames_to_video_tensor(views_np)

    def _frame_to_window_center_impl(
        self, absolute_frame: int, total_frames: int, max_frames: int
    ):
        """Instance version of frame_to_window_center."""
        center_pos = max_frames // 2
        step = self.action_frames
        start_frame = absolute_frame - center_pos * step
        start_frame = max(0, start_frame)
        end_frame = start_frame + (max_frames - 1) * step
        if end_frame >= total_frames:
            start_frame = max(0, total_frames - (max_frames - 1) * step - 1)
        return start_frame, step

    @staticmethod
    def _downsample_segments_to_vae_indices(keyframes, fps, ratio, total_available):
        """Downsample keyframes to VAE-compatible video indices.

        Mirrors mydatasets._downsample_segments_to_vae_indices.
        Result: T = 1 + S*(n_per_seg-1), where (T-1)%4==0.
        """
        if len(keyframes) < 2:
            return list(keyframes)
        S = len(keyframes) - 1
        raw = max(1, round(fps / ratio))
        n_per_seg = max(4, ((raw + 3) // 4) * 4) + 1
        indices = [keyframes[0]]
        for i in range(S):
            pts = np.linspace(keyframes[i], keyframes[i + 1], n_per_seg, dtype=int)
            indices.extend(pts[1:].tolist())
        return [min(idx, total_available - 1) for idx in indices]

    @staticmethod
    def _frames_to_video_tensor(views_frames: list) -> torch.Tensor:
        """Convert per-view frame lists to video tensor.

        Mirrors mydatasets._frames_to_video_tensor (lines 2190-2211).

        Args:
            views_frames: list[list[np.ndarray]] — [v_idx][frame_idx], each HWC uint8
        Returns:
            [C, T, H_total, W] tensor in [-1, 1]. Multi-view concat along H.
        """
        view_tensors = []
        for v_frames in views_frames:
            arr = np.stack(v_frames, axis=0)  # [T, H, W, C]
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # [T, C, H, W]
            view_tensors.append(t)

        if len(view_tensors) == 1:
            video = view_tensors[0]
        else:
            video = torch.cat(view_tensors, dim=-2)  # concat along H

        video = video / 255.0 * 2.0 - 1.0    # [0,255] → [-1,1]
        video = video.permute(1, 0, 2, 3)     # [T,C,H,W] → [C,T,H,W]
        return video

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess_image(self, observations: dict) -> torch.Tensor:
        """Convert raw observation images to model input tensor.

        FastWAM expects input_image as [1, 3, H_total, W] in [-1, 1].
        Multi-view images are concatenated along H (same as training).

        Args:
            observations: {view_name: np.ndarray [H, W, 3] uint8}
        Returns:
            [1, 3, H_total, W] tensor in [-1, 1]
        """
        # Validate
        for view in self.views:
            if view not in observations:
                raise ValueError(
                    f"Missing view '{view}' in observations. "
                    f"Expected keys: {self.views}, got: {list(observations.keys())}"
                )
            arr = observations[view]
            if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"View '{view}' must be np.ndarray [H, W, 3], got "
                    f"type={type(arr)}, shape={getattr(arr, 'shape', '?')}"
                )
            if arr.dtype != np.uint8:
                raise ValueError(f"View '{view}' must be uint8, got {arr.dtype}")

        # Resize each view and collect
        view_tensors = []
        for view in self.views:
            arr = observations[view]
            h, w = arr.shape[:2]
            w_t, h_t = self.image_target_size
            pil = Image.fromarray(arr).resize((w_t, h_t), Image.BILINEAR)
            t = torch.from_numpy(np.array(pil, dtype=np.float32))  # [H, W, 3]
            t = t.permute(2, 0, 1)  # [3, H, W]
            view_tensors.append(t)

        # Concat along H for multi-view
        if len(view_tensors) == 1:
            img = view_tensors[0]
        else:
            img = torch.cat(view_tensors, dim=-2)  # [3, H_total, W]

        # Normalize [0, 255] → [-1, 1]
        img = img / 255.0 * 2.0 - 1.0
        return img.unsqueeze(0)  # [1, 3, H_total, W]

    # ------------------------------------------------------------------
    # Joint normalization
    # ------------------------------------------------------------------

    def _normalize_joints(self, joints: dict) -> torch.Tensor:
        """Assemble and normalize raw joints from a dict to [-1, 1].

        Accepts a dict {joint_key: value} keyed by joint field names.
        When use_6d_rotation is enabled, rotation keys (ending in '_rotation')
        are converted from Euler ZYX [roll, pitch, yaw] to 6D representation.

        Args:
            joints: {str: list|float} raw joint values keyed by joint field name
        Returns:
            [1, D_joint] tensor in [-1, 1]
        """
        if not isinstance(joints, dict):
            raise TypeError(
                f"_normalize_joints expects a dict {{joint_key: value}}, "
                f"got {type(joints).__name__}. "
                f"Pass a dict keyed by joint field names (e.g. {self._joint_keys})."
            )
        if self.j_nm is None or self.j_nd is None:
            raise RuntimeError("Joint normalization params not loaded")
        if not self._joint_keys:
            raise RuntimeError("Joint keys not loaded")

        # Assemble flat vector from dict, applying 6D conversion for rotation keys
        vec = []
        for key in self._joint_keys:
            if key not in joints:
                raise ValueError(
                    f"Joint key '{key}' missing from input dict. "
                    f"Expected keys: {self._joint_keys}, got: {list(joints.keys())}"
                )
            if self.use_6d_rotation and key.endswith("_rotation"):
                # [roll, pitch, yaw] ZYX extrinsic → SO(3) → first two columns → 6D
                rpy = joints[key]
                R = _SciRotation.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()
                vec.extend([R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1]])
            else:
                val = joints[key]
                if isinstance(val, (list, tuple)):
                    vec.extend(val)
                else:
                    vec.append(val)

        flat = np.array(vec, dtype=np.float32)  # [D_joint]
        assert flat.shape[0] == len(self.j_nm), (
            f"Assembled joint dim {flat.shape[0]} != norm dim {len(self.j_nm)}. "
            f"Check joint_keys and use_6d_rotation consistency."
        )

        normed = np.clip(
            2.0 * (flat - self.j_nm) / self.j_nd - 1.0,
            -1.0, 1.0,
        )
        return torch.from_numpy(normed).float().unsqueeze(0)  # [1, D_joint]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_visualization(
        self,
        viz_dir: str,
        step_count: int,
        instruction: str,
        task_window_frames: Optional[list],
        predicted_video: Optional[list],
        gt_frames: Optional[list],
        start_progress: float,
        end_progress: float,
        pred_start_frame_idx: int,
        pred_end_frame_idx: int,
        gt_start_frame_idx: Optional[int] = None,
        gt_end_frame_idx: Optional[int] = None,
        sampled_indices_snapshot: Optional[list] = None,
        video_indices_snapshot: Optional[list] = None,
        predicted_task_frame_idx_snapshot: Optional[int] = None,
    ):
        """Save stitched video + meta.json.

        Columns: [task_video | gt_aligned | pred_aligned | predicted | GT]
        gt_aligned is skipped when gt_start_frame_idx is None.

        task_window_frames: list of multi-view PIL frames at video_indices (~21 frames).
        Borders: red = predicted alignment, green = GT alignment, gray = outside.

        The *_snapshot params are snapshotted copies of self state, allowing this
        method to run safely in a background thread without racing with the next
        infer_real() call.
        """
        os.makedirs(viz_dir, exist_ok=True)

        # Determine number of output frames
        n_frames = 0
        if predicted_video is not None:
            n_frames = len(predicted_video)
        elif gt_frames is not None:
            n_frames = len(gt_frames)
        if n_frames == 0:
            return

        def _to_pil(x):
            if isinstance(x, Image.Image):
                return x
            return Image.fromarray(x)

        def _resize(img, tw, th):
            if img.size == (tw, th):
                return img
            return img.resize((tw, th), Image.BILINEAR)

        def _resample_to_n(frames, n):
            """Resample a list of frames to exactly n frames."""
            if not frames:
                return []
            if len(frames) == n:
                return list(frames)
            idx = np.linspace(0, len(frames) - 1, n, dtype=int)
            return [frames[i] for i in idx]

        # ── Build raw columns ──────────────────────────────────────────
        # Column 1: task_video — all video frames with borders
        task_col_raw = []
        if task_window_frames is not None and len(task_window_frames) > 0:
            for fi, pil_frame in enumerate(task_window_frames):
                frame = pil_frame.copy()
                draw = ImageDraw.Draw(frame)
                w, h = frame.size
                if pred_start_frame_idx <= fi <= pred_end_frame_idx:
                    color = (255, 0, 0)  # red — predicted alignment
                elif (gt_start_frame_idx is not None
                      and gt_start_frame_idx <= fi <= gt_end_frame_idx):
                    color = (0, 255, 0)  # green — GT alignment
                else:
                    color = (128, 128, 128)  # gray
                for offset in range(2):
                    draw.rectangle(
                        [offset, offset, w - 1 - offset, h - 1 - offset],
                        outline=color,
                    )
                task_col_raw.append(frame)
            task_col_raw = _resample_to_n(task_col_raw, n_frames)

        # Column 2: gt_aligned — GT progress-matched segment
        gt_aligned_col_raw = []
        if (gt_start_frame_idx is not None
                and task_window_frames is not None
                and len(task_window_frames) > 0):
            gt_subset = task_window_frames[gt_start_frame_idx:gt_end_frame_idx + 1]
            gt_aligned_col_raw = _resample_to_n(gt_subset, n_frames)

        # Column 3: pred_aligned — predicted progress segment
        pred_aligned_col_raw = []
        if task_window_frames is not None and len(task_window_frames) > 0:
            pred_subset = task_window_frames[pred_start_frame_idx:pred_end_frame_idx + 1]
            pred_aligned_col_raw = _resample_to_n(pred_subset, n_frames)

        # Column 4: predicted video
        pred_col_raw = [_to_pil(f) for f in (predicted_video or [])[:n_frames]]

        # Column 5: GT
        gt_col_raw = [_to_pil(f) for f in (gt_frames or [])[:n_frames]]

        # ── Determine uniform column size ──────────────────────────────
        ref_img = None
        for col in [pred_col_raw, gt_col_raw, task_col_raw]:
            if col:
                ref_img = col[0]
                break
        if ref_img is None:
            return
        col_w, col_h = ref_img.size

        def _fill_or_resize(col):
            if not col:
                return [Image.new("RGB", (col_w, col_h), (0, 0, 0))] * n_frames
            return [_resize(img, col_w, col_h) for img in col]

        task_col = _fill_or_resize(task_col_raw)
        pred_aligned_col = _fill_or_resize(pred_aligned_col_raw)
        pred_col = _fill_or_resize(pred_col_raw)
        gt_col = _fill_or_resize(gt_col_raw)

        # ── Assemble columns ───────────────────────────────────────────
        columns = [("task_video", task_col)]
        if gt_start_frame_idx is not None:
            gt_aligned_col = _fill_or_resize(gt_aligned_col_raw)
            columns.append(("gt_aligned", gt_aligned_col))
        columns += [
            ("pred_aligned", pred_aligned_col),
            ("predicted", pred_col),
            ("GT", gt_col),
        ]

        n_cols = len(columns)
        total_w = col_w * n_cols

        info_text = (
            f"step={step_count} | progress={start_progress:.3f}->{end_progress:.3f} "
            f"| pred_fi=[{pred_start_frame_idx},{pred_end_frame_idx}]"
        )
        if gt_start_frame_idx is not None:
            info_text += f" | gt_fi=[{gt_start_frame_idx},{gt_end_frame_idx}]"
        info_text += f" | {instruction[:80]}"

        _line_h = 11
        _wrapped = textwrap.wrap(info_text, width=max(20, total_w // 7)) or [""]
        _text_h = len(_wrapped) * _line_h + 6

        stitched_frames = []
        for t in range(n_frames):
            canvas = Image.new("RGB", (total_w, col_h + _text_h), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            x = 0
            for label, col_imgs in columns:
                canvas.paste(col_imgs[t], (x, 0))
                draw.rectangle([(x, 0), (x + len(label) * 6 + 8, 13)], fill=(0, 0, 0))
                draw.text((x + 4, 2), label, fill=(255, 255, 255))
                x += col_w
            # Frame counter
            _fc = f"{t+1}/{n_frames}"
            _fc_w = len(_fc) * 6 + 8
            draw.rectangle([(total_w - _fc_w, 0), (total_w, 13)], fill=(0, 0, 0))
            draw.text((total_w - _fc_w + 4, 2), _fc, fill=(255, 255, 255))
            # Text strip
            for i, line in enumerate(_wrapped):
                draw.text((4, col_h + 3 + i * _line_h), line, fill=(255, 255, 255))
            stitched_frames.append(canvas)

        # Save MP4
        mp4_path = os.path.join(viz_dir, f"step_{step_count:05d}_stitched.mp4")
        save_mp4(stitched_frames, mp4_path, fps=8)
        _update_latest_symlink(mp4_path, "latest_stitched.mp4")

        # Save meta.json
        meta = {
            "step_count": step_count,
            "sampled_indices": sampled_indices_snapshot,
            "video_indices": video_indices_snapshot,
            "start_progress": start_progress,
            "end_progress": end_progress,
            "pred_start_frame_idx": pred_start_frame_idx,
            "pred_end_frame_idx": pred_end_frame_idx,
            "gt_start_frame_idx": gt_start_frame_idx,
            "gt_end_frame_idx": gt_end_frame_idx,
            "predicted_absolute_frame": predicted_task_frame_idx_snapshot,
            "instruction": instruction,
        }
        meta_path = os.path.join(viz_dir, f"step_{step_count:05d}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        _update_latest_symlink(meta_path, "latest_meta.json")

        logger.info("[viz] Saved %s", mp4_path)

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_real(
        self,
        observations: dict,
        joints: dict,
        instruction: str,
        *,
        episode_dir: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        save_intermediates: bool = False,
        visualize: bool = False,
        viz_dir: Optional[str] = None,
        step_count: int = 0,
        gt_frames: Optional[list] = None,
        gt_progress_list: Optional[dict] = None,
        current_agent_idx: Optional[int] = None,
        gt_action_chunk: Optional[np.ndarray] = None,
    ) -> dict:
        """Main inference entry point. Takes raw sensor data, returns denormalized action.

        Args:
            observations: {view_name: np.ndarray [H, W, 3] uint8}
            joints: {str: list|float} raw joint values keyed by joint field name
            instruction: task description text
            episode_dir: task video episode dir (auto-loads on first call)
            seed: random seed for diffusion sampling
            num_inference_steps: override self.diffusion_steps
            save_intermediates: if True, include predicted video frames in output
            visualize: if True, save stitched video + meta to viz_dir
            viz_dir: directory for visualization output (required if visualize=True)
            step_count: current step index (for file naming)
            gt_frames: list[np.ndarray [H,W,3] uint8] GT frames for visualization
            gt_progress_list: dict {frame_idx_str: progress} from info_dtw.json, or None
            current_agent_idx: current agent frame index for GT alignment
            gt_action_chunk: [K, D] GT action chunk for action plot (optional)

        Returns:
            dict with keys:
                action_raw:    [max_action_len, D_euler] denormalized action
                action_normed: [max_action_len, action_dim] model output in [-1, 1]
                predicted_video: list[PIL.Image] (only if save_intermediates)
                task_video_indices: list[int] sampled keyframe indices
        """
        steps = num_inference_steps or self.diffusion_steps

        # Phase 1: Input validation + preprocessing
        input_image = self._preprocess_image(observations)  # [1, 3, H_total, W]
        proprio = self._normalize_joints(joints)             # [1, D_joint]

        # Phase 2: Task video
        task_video_tensor = None
        task_video_dropped = False
        task_window_pil_frames = None
        if self.use_task_video:
            if episode_dir is not None and not self.task_video_loaded:
                self.set_task_video(episode_dir)
            if not self.task_video_loaded:
                raise RuntimeError(
                    "use_task_video=True but task video not loaded. "
                    "Provide episode_dir or call set_task_video() first."
                )
            n_views = len(self.views)
            max_frames = self.task_max_frames[n_views][1]
            task_video_tensor = self._get_task_video_window(max_frames)
            # Build multi-view PIL frames for visualization (all video_indices frames)
            if visualize and self._task_frames is not None:
                task_window_pil_frames = []
                for idx in (self.current_video_indices or []):
                    view_imgs = [self._task_frames[v][idx]
                                 for v in range(len(self.views))]
                    task_window_pil_frames.append(_vstack_pil(view_imgs))

        if not self.use_task_video:
            # Construct zero task_video to match training's "all dropped" behavior.
            # Training dataset returns zeros [C, T_task, H_total, W] with task_video_dropped=True;
            # this ensures inference walks the same prepend_cross_attn path.
            n_views = len(self.views)
            max_frames = self.task_max_frames[n_views][1]
            S = max_frames - 1
            T_task = 1 + S * (self.n_per_seg - 1)
            _, _, H_total, W = input_image.shape  # [1, 3, H_total, W]
            task_video_tensor = torch.zeros(3, T_task, H_total, W)
            task_video_dropped = True

        # Phase 3: Text encoding
        if self.use_task_video and not self.use_task_description:
            if self._cached_context is None:
                context, context_mask = self.model.encode_prompt(instruction)
                self._cached_context = (torch.zeros_like(context), torch.zeros_like(context_mask))
                self._cached_prompt = None
            context, context_mask = self._cached_context
        else:
            if self._cached_prompt != instruction or self._cached_context is None:
                context, context_mask = self.model.encode_prompt(instruction)
                self._cached_context = (context, context_mask)
                self._cached_prompt = instruction
            else:
                context, context_mask = self._cached_context

        # Phase 4: Model inference
        num_video_frames = max(5, self.frames)
        if num_video_frames % 4 != 1:
            num_video_frames = (num_video_frames // 4) * 4 + 1

        result = self.model.infer_joint(
            prompt=None,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=self.max_action_len,
            context=context.clone(),
            context_mask=context_mask.clone(),
            proprio=proprio,
            task_video=task_video_tensor,
            num_inference_steps=steps,
            seed=seed,
            test_action_with_infer_action=False,
            task_video_dropped=task_video_dropped,
        )

        action_full = result["action"].cpu().numpy()  # [max_action_len, action_dim+2]
        action_normed = action_full[:, :len(self.norm_low)]

        # Phase 5: Denormalize (and convert relative→absolute if needed)
        if self.use_relative_action and joints is not None:
            # Step 1: denormalize without 6D→Euler (relative_to_absolute needs 6D)
            action_raw = denormalize_action(
                action_normed,
                self.norm_low,
                self.norm_high,
                use_6d_rotation=False,
                action_col_map=self._action_col_map,
            )
            # Step 2: relative → absolute using follower state from joints dict
            action_raw = relative_to_absolute(
                action_raw, joints, self._action_col_map,
            )
            # Step 3: 6D → Euler
            if self.use_6d_rotation and self._action_col_map is not None:
                action_raw = decode_6d_to_euler(action_raw, self._action_col_map)
        else:
            action_raw = denormalize_action(
                action_normed,
                self.norm_low,
                self.norm_high,
                use_6d_rotation=self.use_6d_rotation,
                action_col_map=self._action_col_map,
            )

        # Phase 6: Progress → video frame index mapping
        # progress ∈ [-1, 1] → frame index ∈ [0, T_task - 1]
        progress_pred = result.get("progress")
        if progress_pred is not None:
            progress_pred = progress_pred.cpu().numpy() if hasattr(progress_pred, 'cpu') else progress_pred
            _start_progress = float(progress_pred[0])
            _end_progress = float(progress_pred[1])
        else:
            _start_progress = 0.0
            _end_progress = 0.0

        T_task = len(self.current_video_indices) if self.current_video_indices else 1
        _pred_start_fi = int((_start_progress + 1.0) / 2.0 * (T_task - 1))
        _pred_start_fi = max(0, min(_pred_start_fi, T_task - 1))
        _pred_end_fi = int((_end_progress + 1.0) / 2.0 * (T_task - 1))
        _pred_end_fi = max(0, min(_pred_end_fi, T_task - 1))

        # Advance task video window using predicted end progress → keyframe index
        if self.current_video_indices is not None:
            _abs_frame = self.current_video_indices[_pred_end_fi]
            self.predicted_task_frame_idx = _abs_frame

        # Phase 7: GT aligned computation (frame-level in video_indices)
        _gt_start_fi = None
        _gt_end_fi = None
        if (gt_progress_list is not None
                and current_agent_idx is not None
                and self.current_video_indices is not None):
            agent_progress = gt_progress_list.get(str(current_agent_idx))
            if agent_progress is not None:
                video_progresses = []
                for idx in self.current_video_indices:
                    # Map filtered index → raw frame index for progress lookup
                    raw_idx = self._task_raw_indices[idx] if self._task_raw_indices else idx
                    p = gt_progress_list.get(str(raw_idx))
                    video_progresses.append(float(p) if p is not None else float('inf'))
                video_progresses = np.array(video_progresses, dtype=float)
                _gt_start_fi = int(np.argmin(np.abs(video_progresses - float(agent_progress))))
                _gt_end_fi = min(_gt_start_fi + self.frames_per_seg, T_task - 1)

        if visualize and viz_dir is not None:
            # Snapshot mutable self state for thread safety
            _snap_sampled = list(self.current_sampled_indices) if self.current_sampled_indices else None
            _snap_video = list(self.current_video_indices) if self.current_video_indices else None
            _snap_pred_frame = self.predicted_task_frame_idx

            self._drain_viz_futures()
            self._viz_futures.append(self._viz_executor.submit(
                self._save_visualization,
                viz_dir=viz_dir,
                step_count=step_count,
                instruction=instruction,
                task_window_frames=task_window_pil_frames,
                predicted_video=result.get("video"),
                gt_frames=gt_frames,
                start_progress=_start_progress,
                end_progress=_end_progress,
                pred_start_frame_idx=_pred_start_fi,
                pred_end_frame_idx=_pred_end_fi,
                gt_start_frame_idx=_gt_start_fi,
                gt_end_frame_idx=_gt_end_fi,
                sampled_indices_snapshot=_snap_sampled,
                video_indices_snapshot=_snap_video,
                predicted_task_frame_idx_snapshot=_snap_pred_frame,
            ))

            # Action plot (if GT provided)
            if gt_action_chunk is not None:
                _pred_chunk = action_raw[:len(gt_action_chunk)].copy()
                _gt_chunk = gt_action_chunk.copy()
                def _save_action_plot_with_latest():
                    plot_path = os.path.join(viz_dir, f"step_{step_count:05d}_action.png")
                    save_action_plot(
                        plot_path, _pred_chunk, _gt_chunk,
                        title=f"Action chunk @ t={step_count}",
                    )
                    _update_latest_symlink(plot_path, "latest_action.png")
                self._viz_futures.append(
                    self._viz_executor.submit(_save_action_plot_with_latest)
                )

        # Phase 8: Build output
        output = {
            "action_raw": action_raw,
            "action_normed": action_normed,
            "task_video_indices": self.current_sampled_indices,
            "video_indices": self.current_video_indices,
            "predicted_progress": _end_progress,
            "predicted_start_progress": _start_progress,
            "pred_start_frame_idx": _pred_start_fi,
            "pred_end_frame_idx": _pred_end_fi,
        }

        if save_intermediates and "video" in result:
            output["predicted_video"] = result["video"]

        return output
