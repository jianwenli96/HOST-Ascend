# -*- coding: utf-8 -*-
import json
import os
import os.path as osp
import re
import random
import glob
import torch
import numpy as np
import logging
import traceback
import decord
from typing import List, Union
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import sys

# Set decord bridge to native for better performance
decord.bridge.set_bridge('native')

# Import view configuration for JSON mode
from .view_config import DATASET_INSTRUCTION_FILENAME

# Import path transformation configuration for JSON mode
from .path_transforms_config import get_path_transforms

# Import normalization constants for x2robot action data
from .constant import DATA_WARNING_FLAGS
from .joint_action_mapping_norms import (
    build_joint_norm_vectors,
    build_norm_vectors,
    load_joint_action_mapping_cache,
    parse_field_norms,
)

# Lazy import for 6D rotation conversion (only needed when use_6d_rotation=True)
from scipy.spatial.transform import Rotation as _SciRotation

# Lazy import: rel_step_to_abs from convert_abs_to_rel_actions (Euler inverse, matches --verify)
_REL_STEP_TO_ABS = None


# Datasets that require segment path parsing (format: "path:segment_id:start-end")
# Only these datasets will have their paths parsed for segment information
# Other datasets with colons in paths (e.g., timestamps) will be treated as simple paths
SEGMENT_PARSING_DATASETS = {
    'AgiBotWorld',  # Uses segment format for frame ranges
    'AgiBotWorld-Beta-v2',
    'robocoin_v2',
    # Add more datasets here if they use segment format
}

def _build_action_component_slices(col_map: dict) -> dict:
    """Convert col_map {key: (start, dim)} → {short_name: (start, end)}.

    Naming: "left"/"right" → "L"/"R" prefix; "position" → "pos",
    "rotation" → "rot", "gripper" → "grip".
    Falls back to the full key name if pattern not recognized.
    """
    slices = {}
    for key, (start, dim) in col_map.items():
        k = key.lower()
        prefix = "L" if "left" in k else ("R" if "right" in k else "")
        if "position" in k:
            suffix = "pos"
        elif "rotation" in k:
            suffix = "rot"
        elif "gripper" in k:
            suffix = "grip"
        else:
            suffix = key  # fallback: keep full key
        short = f"{prefix}_{suffix}" if prefix else suffix
        slices[short] = (start, start + dim)
    return slices


class SilentFilterError(Exception):
    """Exception raised when a sample should be silently filtered out."""
    pass


class CamMappingError(RuntimeError):
    """Cam mapping missing, task not in index, or disk view mismatch; not swallowed by __getitem__ retry loop."""
    pass


def natural_sort_key(s):
    """
    Natural sort key for strings containing numbers.
    Example: 'frame_10.jpg' -> ['frame_', 10, '.jpg']
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


Interval_times = [1,2,3]
TASK_MASKED_FRAMES = 2
PROGRESS_BINS = 10

# Minimum frames filter (for JSON mode data quality control)
# Samples with fewer frames than this threshold will be filtered out during loading
MIN_PHYSICAL_SECONDS_THRESHOLD = 4  # Must cover at least 6 seconds of physical time

# Image processing configuration (for raw_image mode)
# Default fallback pixel count (128*128); overridden per-type via DataArguments.
IMAGE_MIN_PIXELS = 16384  # 128*128

# Per-dataset temporal stride for action downsampling (factor N keeps every N-th frame).
# Exported as a module-level constant so train_moe.py can persist it in model config
# without instantiating a dataset object.
ACTION_DOWNSAMPLE_FACTOR: dict = {
    'AgiBotWorld-Beta-v2': 1,
    'Robochallenge_x2':    1,
    'robocoin_v2':         1,
    'robomindv2':          1,
    'robomind_v2.0_v2':    1,
    'robomind':            1,
    '10042':               1,
    '10053':               1,
    '10058':               1,
}

# Simplified token types for SelfGroundedPredictor (task video vs agent image distinction)
TOKEN_TYPE_PAD = 0
TOKEN_TYPE_TASK_VIDEO = 1
TOKEN_TYPE_AGENT_IMAGE = 2

class CustomDataset(Dataset):
    debug_count = 0
    _printed_messages = set()

    def __init__(self, args: "DataArguments", tokenizer=None):
        super().__init__()

        self.args = args
        # data args
        _tvdp = getattr(args, 'task_video_drop_prob', None)
        _ttdp = getattr(args, 'task_text_drop_prob', None)
        if _tvdp is None or _ttdp is None:
            raise ValueError(
                "config must include 'task_video_drop_prob' and 'task_text_drop_prob'."
            )
        self.task_video_drop_prob = float(_tvdp)
        self.task_text_drop_prob = float(_ttdp)
        self.task_paths_filename = str(getattr(args, 'task_paths_filename', 'task_paths.json'))
        self.task_masked_frames = TASK_MASKED_FRAMES
        cfg_tmf = getattr(args, 'task_max_frames', None)
        if cfg_tmf is None:
            raise ValueError(
                "config must include 'task_max_frames' dict (e.g. {1: [8,10], 2: [6,6], 3: [4,4]}). "
                "Add task_max_frames to your data config YAML."
            )
        self.task_max_frames = {int(k): list(v) for k, v in cfg_tmf.items()}

        # Sampling interval multipliers for random action frame spacing
        _si_min = getattr(args, 'sampling_interval_min_mult', None)
        _si_max = getattr(args, 'sampling_interval_max_mult', None)
        if _si_min is None or _si_max is None:
            raise ValueError(
                "config must include 'sampling_interval_min_mult' and 'sampling_interval_max_mult'. "
                "Add them to your data config YAML."
            )
        self.sampling_interval_min_mult = float(_si_min)
        self.sampling_interval_max_mult = float(_si_max)

        # Whether to apply random sub-segment offset in task video frame-level shift
        self.task_video_random_offset = bool(getattr(args, 'task_video_random_offset', True))

        self.random_frame_sampling = args.random_frame_sampling
        self.raw_image = args.raw_image

        # JSON loading (lazy - only path list in memory)
        self.data = self._load_from_json(args.data_path)
        print(f"JSON mode: Loaded {len(self.data)} video paths (lazy loading)")

        # Episode-level exclusion: filter by original_path against a provided JSON list.
        exclude_json = getattr(args, 'exclude_episode_json', None)
        if exclude_json:
            excluded_paths = self._load_excluded_episode_paths(exclude_json)
            original_len = len(self.data)
            self.data = [
                item for item in self.data
                if item.get('original_path') not in excluded_paths
            ]
            excluded_count = original_len - len(self.data)
            logging.warning(
                "[CustomDataset] exclude_episode_json: excluded %d / %d episodes from %s",
                excluded_count, original_len, exclude_json
            )

        # weights
        dataset_weights = {
            # Sorted by dataset name (matching train_video_1node.sh order)
            'berkeley_autolab_ur5': 2.0,
            'bridgev2': 1.0,
            'calvin': 1.0,
            'cmu_play_fusion': 1.0,
            'droid': 0.2,
            'fmb': 1.0,
            'language_table': 0.1,
            'libero': 1.0,
            'maniskill': 1.0,
            'robocoin': 1.0,
            'rt1': 0.5,
            'SSv2': 0.5,
            'taco_play': 1.0,
            'toto': 1.0,
            'utaustin_mutex': 1.0,
            'viola': 1.0,
            # New datasets
            'AgiBotWorld': 0.05,
            'AgiBotWorld-Beta-v2': 0.05,
            'bridge_data_v2_img1_v2': 1.0,
            'bridge_data_v2_img3_v2': 1.0,
            'bridge_data_v2_img4_v2': 1.0,
            'droid_success_only_v2': 1.0,
            'fractal20220817_data_v2': 1.0,
            'Robochallenge_x2': 1.0,
            'robocoin_v2': 1.0,
            'robomindv2': 0.1,
            'robomind_v2.0_v2': 0.1,
            'robomind': 0.5,
            'Egodex': 0.1,
            '10042': 0.3,
            '10053': 0.3,
            '10058': 0.3,
        }
        
        # Debug: Count dataset distribution
        from collections import Counter
        dataset_names = [d.get("dataset", "") for d in self.data]
        dataset_counts = Counter(dataset_names)
        print("\n" + "="*80)
        print("📊 Dataset Distribution in Loaded Data:")
        print("="*80)
        for dataset_name, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
            weight = dataset_weights.get(dataset_name, 1.0)
            has_weight = "✓" if dataset_name in dataset_weights else "✗"
            print(f"  [{has_weight}] {dataset_name:40s}: {count:8d} samples (weight: {weight:.2f})")
        print("="*80)
        
        # Check for datasets without defined weights
        undefined_datasets = [name for name in dataset_counts.keys() if name not in dataset_weights and name != ""]
        if undefined_datasets:
            print(f"⚠️  WARNING: {len(undefined_datasets)} dataset(s) without defined weights (will use default 1.0):")
            for name in undefined_datasets:
                print(f"     - '{name}'")
            print("="*80)
        print()
        
        self.datasets_weight = getattr(args, 'datasets_weight', True)
        if self.datasets_weight:
            self.sample_weights = [dataset_weights.get(d.get("dataset", ""), 1.0) for d in self.data]
            print(f"✓ Weighted sampling ENABLED with {len(self.sample_weights)} sample weights")
        else:
            # 即使不启用加权采样，也要设置 sample_weights（全部为 1.0）
            # 这确保了 WeightedSamplerTrainer 能够正常工作
            self.sample_weights = [1.0] * len(self.data)
            print(f"✓ Uniform sampling (all weights = 1.0) with {len(self.sample_weights)} samples")

        self.cfg = False

        # Per-dataset FPS: must be provided in config (dataset_fps field).
        cfg_fps = getattr(args, 'dataset_fps', None)
        if cfg_fps is None:
            raise ValueError(
                "config must include 'dataset_fps' dict (e.g. {'10042': 32, ...}). "
                "Add dataset_fps to your data config YAML."
            )
        self.dataset_fps = {str(k): int(v) for k, v in cfg_fps.items()}

        # Per-dataset uniform action downsampling factor (see module-level ACTION_DOWNSAMPLE_FACTOR).
        self.action_downsample_factor = ACTION_DOWNSAMPLE_FACTOR
        
        # Cam mapping (eagerly loaded; see _init_cam_mapping)
        self.cam_mapping_dir = getattr(args, 'cam_mapping_dir', '/open_data/cgy/cam_mapping')
        self._init_cam_mapping()

        # Joint-action mapping (eagerly loaded; see _init_joint_action_mapping)
        self.joint_action_mapping_dir = getattr(args, 'joint_action_mapping_dir',
                                                '/open_data/cgy/joint_action_mapping')
        self._init_joint_action_mapping()

        self.T = args.frames
        self.action_frames = args.action_frames

        self.actions = args.actions
        self.joints = args.joints
        self.use_6d_rotation = args.use_6d_rotation

        # Online preprocessing flags (JSON mode only)
        self.use_relative_action   = args.use_relative_action
        self.remove_static_frames    = args.remove_static_frames
        self.static_rot_threshold    = args.static_rot_threshold    # radians
        self.static_trans_threshold  = args.static_trans_threshold  # metres
        self.static_gripper_threshold = getattr(args, "static_gripper_threshold", 0.0)

        self.use_indicator_prompt = getattr(args, "use_indicator_prompt", False)
        self.use_indicator_positive = bool(getattr(args, "use_indicator_positive", False))
        if self.use_indicator_prompt:
            raise NotImplementedError(
                "use_indicator_prompt is not yet supported in SelfGroundedPredictor. "
                "Set use_indicator_prompt=False."
            )

        # Parse num_view_probs: JSON string -> {int(n): float(prob)}
        try:
            raw_probs = json.loads(args.num_view_probs)
            self.num_view_probs = {int(k): float(v) for k, v in raw_probs.items()}
            assert self.num_view_probs, "num_view_probs must not be empty"
        except Exception as e:
            logging.warning(f"Failed to parse num_view_probs='{args.num_view_probs}': {e}. Defaulting to {{1: 1.0}}")
            self.num_view_probs = {1: 1.0}
        self.skip_no_action = getattr(args, "skip_no_action", False)

        # Progress alignment args
        self.progress_bins = PROGRESS_BINS
        # Per-dataset image size: {dataset_name: (W, H)} — PIL convention
        cfg_image_size = getattr(args, 'dataset_image_size', None)
        if cfg_image_size is None:
            raise ValueError(
                "config must include 'dataset_image_size' dict "
                "(e.g. {'10042': [224, 224], ...}). "
                "Add dataset_image_size to your data config YAML."
            )
        self.dataset_image_size = {}
        for k, v in cfg_image_size.items():
            h, w = int(v[0]), int(v[1])
            if h % 16 != 0 or w % 16 != 0:
                raise ValueError(
                    f"dataset_image_size['{k}'] = [{h}, {w}] must be divisible by 16 "
                    f"(VAE requirement)"
                )
            self.dataset_image_size[str(k)] = (w, h)  # store as (W, H) for PIL
        self.use_augmentation = getattr(args, 'use_augmentation', False)

        # SelfGroundedPredictor video params
        self.action_video_freq_ratio = getattr(args, 'action_video_freq_ratio', 4)
        self.max_action_len = getattr(args, 'max_action_len', None)
        self.context_len = getattr(args, 'context_len', 128)
        # Expose collate_fn for DataLoader (handles variable-length task_video + str prompts)
        self.collate_fn = CustomCollator(context_len=self.context_len)

    def get_action_component_slices(self) -> dict:
        """Return {short_name: (start, end)} from the first episode's col_map.

        Assumes all episodes share the same action key structure (same col_map layout).
        Caches in self._action_component_slices for consistency with _get_single_item.
        """
        if getattr(self, '_action_component_slices', None):
            return self._action_component_slices
        r = self._load_episode_raw(self.data[0])
        if r is None:
            logging.warning(
                "[get_action_component_slices] _load_episode_raw returned None for data[0]; "
                "per-key action metrics will be disabled."
            )
            return {}
        slices = _build_action_component_slices(r['col_map'])
        self._action_component_slices = slices
        logging.info("[action_component_slices] built from data[0]: %s", list(slices.keys()))
        return slices

    def __len__(self):
        return len(self.data)

    def _data_warn(self, key: str, msg: str) -> None:
        """Emit logging.warning when DATA_WARNING_FLAGS[key] is True (default True if key missing)."""
        if DATA_WARNING_FLAGS.get(key, True):
            logging.warning(msg)

    # ==================== JSON Mode Helper Methods ====================

    def _get_target_size(self, dataset_name):
        """Return (width, height) for this dataset from config.

        Raises ValueError if the dataset is not configured in dataset_image_size.
        """
        if dataset_name not in self.dataset_image_size:
            raise ValueError(
                f"dataset '{dataset_name}' not found in dataset_image_size config. "
                f"Available: {list(self.dataset_image_size.keys())}"
            )
        return self.dataset_image_size[dataset_name]

    def _sample_aug_params(self) -> dict:
        """Sample augmentation parameters once per sample.

        Returns a dict of scalar values (all resolution-independent fractions or
        absolute scalars).  Pass the same dict to _apply_aug() for every
        frame/view that should share identical augmentation within one sample.
        """
        apply_blur = random.random() < 0.5
        crop_scale = random.uniform(0.8, 1.0)
        return {
            'brightness': random.uniform(0.7, 1.3),
            'contrast':   random.uniform(0.7, 1.3),
            'saturation': random.uniform(0.7, 1.3),
            'hue':        random.uniform(-0.05, 0.05),
            # blur_sigma=None means no blur for this sample
            'blur_sigma': random.uniform(0.1, 2.0) if apply_blur else None,
            # Crop: uniform scale applied to both h and w; top/left as fractions of
            # the remaining slack so they are independent of absolute image size.
            'crop_scale': crop_scale,
            'top_frac':   random.uniform(0.0, 1.0),
            'left_frac':  random.uniform(0.0, 1.0),
        }

    def _apply_aug(self, pil_images: list, params: dict) -> list:
        """Apply fixed augmentation params to a list of PIL Images.

        All frames receive *identical* transforms (same brightness factor,
        same crop box as a fraction, etc.) so temporal and view consistency
        is guaranteed when the caller reuses the same `params` dict.

        Args:
            pil_images: List[PIL.Image] — all at the same (W, H).
            params:     Dict returned by _sample_aug_params().

        Returns:
            List[PIL.Image] at the same (W, H) as the inputs.

        Shape contract:
            Input:  [PIL(W,H)] × N
            Output: [PIL(W,H)] × N  (crop+resize preserves spatial dims)
        """
        assert pil_images, "_apply_aug received an empty image list"
        result = []
        for img in pil_images:
            w, h = img.size
            # --- Color jitter (SimCLR order: brightness → contrast → saturation → hue) ---
            img = TF.adjust_brightness(img, params['brightness'])
            img = TF.adjust_contrast(img, params['contrast'])
            img = TF.adjust_saturation(img, params['saturation'])
            img = TF.adjust_hue(img, params['hue'])
            # --- Gaussian blur (optional) ---
            if params['blur_sigma'] is not None:
                img = img.filter(ImageFilter.GaussianBlur(radius=params['blur_sigma']))
            # --- Random crop + resize back to original spatial size ---
            crop_h = max(1, int(h * params['crop_scale']))
            crop_w = max(1, int(w * params['crop_scale']))
            # top/left as fractions of the slack space; clamped implicitly by max(0,…)
            top  = int(params['top_frac']  * max(0, h - crop_h))
            left = int(params['left_frac'] * max(0, w - crop_w))
            img = TF.crop(img, top, left, crop_h, crop_w)
            img = img.resize((w, h), Image.BILINEAR)   # restore original spatial dims
            result.append(img)
        return result

    def _load_frame(self, paths, target_size: tuple):
        """
        Batch loader for images and video frames with direct resize.

        Assumes: All paths in a batch are either all videos (from same video) or all images.

        Args:
            paths: Single path string or list of paths (all same type)
            target_size: (width, height) tuple for resizing

        Returns:
            Single PIL Image or list of PIL Images
        """
        # Handle single vs list
        is_single = isinstance(paths, str)
        path_list = [paths] if is_single else paths

        if not path_list:
            return [] if not is_single else None

        try:
            first_path = path_list[0]
            is_video = isinstance(first_path, str) and first_path.startswith("video://")

            target_width, target_height = target_size
            results = []

            if is_video:
                # All are video frames from the same video
                path_info = first_path[8:]  # Remove "video://"
                video_path = path_info.split('::')[0]

                # Collect all frame indices
                frame_indices = []
                for p in path_list:
                    parts = p[8:].split('::')
                    frame_indices.append(int(parts[1]))

                # Batch read with direct resize
                vr = decord.VideoReader(
                    video_path,
                    num_threads=1,
                    width=target_width,
                    height=target_height
                )
                frames_np = vr.get_batch(frame_indices).asnumpy()
                del vr

                for frame in frames_np:
                    img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
                    results.append(img)

            else:
                # All are image files
                for img_path in path_list:
                    img = Image.open(img_path).convert('RGB')
                    if img.size != (target_width, target_height):
                        img = img.resize((target_width, target_height))
                    results.append(img)

            return results[0] if is_single else results

        except Exception as e:
            logging.warning(f"Failed to load frames from {paths}: {e}")
            placeholder = Image.new('RGB', target_size, (0, 0, 0))
            return placeholder if is_single else [placeholder] * len(path_list)
    
    def _apply_path_transform(self, path, dataset_name):
        """
        Apply path transformation rules based on dataset type.
        
        This enables path prefix remapping for different datasets, useful for
        data migration or reorganization scenarios.
        
        Example:
            path = '/open_data/task1/instruction.txt'
            dataset_name = 'AgiBotWorld'
            transform = {'/open_data': '/open_data/cgy'}
            -> returns '/open_data/cgy/task1/instruction.txt'
        
        Args:
            path: Original file path
            dataset_name: Dataset name (e.g., 'AgiBotWorld', 'BridgeV2')
        
        Returns:
            Transformed path or original path if no transformation applies
        """
        # Skip if no dataset name provided
        if dataset_name is None:
            return path
        
        # Get transformation rules for this dataset
        transforms = get_path_transforms(dataset_name)
        
        if not transforms:
            return path
        
        # Apply first matching prefix transformation
        for old_prefix, new_prefix in transforms.items():
            if path.startswith(old_prefix):
                transformed = path.replace(old_prefix, new_prefix, 1)
                return transformed
        
        return path
    
    def _fs_parent_dir_from_media_path(self, path):
        """Directory containing the media file; matches _load_frame handling of video:// paths."""
        if isinstance(path, str) and path.startswith("video://"):
            fs_file = path[8:].split("::")[0]
            return osp.dirname(fs_file)
        return osp.dirname(path)
    
    def _parse_video_path(self, path_str, dataset_name=None):
        """
        Parse video path to extract segment information.
        
        Only parses segment format for datasets in SEGMENT_PARSING_DATASETS.
        Other datasets (like DROID with timestamps in paths) are treated as simple paths.
        
        Formats:
            - Simple: "/path/to/video"
            - With segment (only for whitelisted datasets): "/path/to/video:segment_id:start-end"
            - Example: "/data/AgiBotWorld/task1:0:100-200"
        
        Args:
            path_str: Path string to parse
            dataset_name: Dataset name (e.g., 'AgiBotWorld', 'DROID')
        
        Returns:
            dict: {
                'video_dir': str,
                'segment_id': int or None,
                'frame_start': int or None,
                'frame_end': int or None,
                'original_path': str
            }
        """
        # Check if this dataset requires segment parsing
        needs_segment_parsing = dataset_name in SEGMENT_PARSING_DATASETS if dataset_name else False
        
        # If no colons or dataset doesn't need segment parsing, treat as simple path
        if ':' not in path_str or not needs_segment_parsing:
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }
        
        # Parse segment format for whitelisted datasets
        parts = path_str.rsplit(':', 2)  # Split from right, max 3 parts
        
        if len(parts) != 3:
            # Format incorrect, treat as simple path
            logging.warning(f"Path format incorrect '{path_str}'. Treating as simple path.")
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }
        
        video_dir, segment_id_str, frame_range_str = parts
        
        try:
            # Parse segment_id
            segment_id = int(segment_id_str)
            
            # Parse frame_range "456-651"
            if '-' in frame_range_str:
                frame_start_str, frame_end_str = frame_range_str.split('-')
                frame_start = int(frame_start_str)
                frame_end = int(frame_end_str)
            else:
                # Format incorrect, treat as simple path
                logging.warning(f"Failed to parse frame range '{frame_range_str}' for dataset '{dataset_name}'. Treating as simple path.")
                return {
                    'video_dir': path_str,
                    'segment_id': None,
                    'frame_start': None,
                    'frame_end': None,
                    'original_path': path_str
                }
            
            return {
                'video_dir': video_dir,
                'segment_id': segment_id,
                'frame_start': frame_start,
                'frame_end': frame_end,
                'original_path': path_str
            }
        except (ValueError, AttributeError) as e:
            # Parse failed, treat as simple path
            logging.warning(f"Failed to parse video path '{path_str}' for dataset '{dataset_name}': {e}. Treating as simple path.")
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }
    
    def _get_available_views(self, video_info):
        """
        Get available views (camera angles) for a video.
        
        Priority:
            1. Check for .mp4 files (e.g., images.mp4, gripper_images.mp4)
            2. Check for subdirectories (e.g., images/, gripper_images/)
        
        Args:
            video_info: dict with 'video_dir' and optional 'segment_id' keys
        
        Returns:
            list[str]: View names (e.g., ['images', 'gripper_images', 'front', 'wrist1'])
        """
        video_dir = video_info['video_dir'] if isinstance(video_info, dict) else video_info
        segment_id = video_info.get('segment_id') if isinstance(video_info, dict) else None
        
        # If segment_id exists, look in segment subdirectory
        if segment_id is not None:
            search_dir = osp.join(video_dir, str(segment_id))
        else:
            search_dir = video_dir
        
        if not osp.exists(search_dir):
            logging.warning(f"Directory does not exist: {search_dir}")
            return []
        
        # Priority 1: Check for mp4 files
        mp4_files = glob.glob(osp.join(search_dir, '*.mp4'))
        if mp4_files:
            # Return mp4 filenames without extension as view names
            views = [osp.splitext(osp.basename(mp4))[0] for mp4 in mp4_files]
            return sorted(views)
        
        # Priority 2: Check for subdirectories
        try:
            subdirs = [d for d in os.listdir(search_dir) 
                      if osp.isdir(osp.join(search_dir, d))]
            if subdirs:
                return sorted(subdirs)
        except Exception as e:
            logging.warning(f"Failed to list subdirectories in {search_dir}: {e}")
        
        return []
    
    def _get_files(self, video_info, view):
        """
        Get list of frame paths/references for a view.
        
        Supports:
            - Image sequences: returns list of .jpg/.png file paths
            - MP4 videos: returns list of "video://<path>::<idx>::<w>x<h>" strings
            - Segmented videos: respects frame_start/frame_end from video_info
        
        Args:
            video_info: dict from _parse_video_path()
            view: str, view name (e.g., 'images', 'gripper_images', 'front', 'wrist1')
        
        Returns:
            list[str]: Frame paths or video:// protocol strings
        """
        video_dir = video_info['video_dir'] if isinstance(video_info, dict) else video_info
        segment_id = video_info.get('segment_id') if isinstance(video_info, dict) else None
        frame_start = video_info.get('frame_start') if isinstance(video_info, dict) else None
        frame_end = video_info.get('frame_end') if isinstance(video_info, dict) else None
        
        # Determine search directory
        if segment_id is not None:
            search_dir = osp.join(video_dir, str(segment_id))
        else:
            search_dir = video_dir
        
        if not osp.exists(search_dir):
            logging.warning(f"Directory does not exist: {search_dir}")
            return []
        
        # Check for mp4 file
        mp4_path = osp.join(search_dir, f"{view}.mp4")
        if osp.exists(mp4_path):
            # MP4 format: generate video:// protocol strings
            try:
                import decord
                decord.bridge.set_bridge('native')
                vr = decord.VideoReader(mp4_path, num_threads=1)
                total_frames = len(vr)
                
                if total_frames == 0:
                    logging.warning(f"MP4 file {mp4_path} has 0 frames")
                    del vr
                    return []
                
                # Determine frame range
                if frame_start is not None and frame_end is not None:
                    # Use specified frame range
                    start = max(0, frame_start)
                    end = min(total_frames - 1, frame_end)
                    frame_indices = range(start, end + 1)
                else:
                    # Use all frames
                    frame_indices = range(total_frames)
                
                # Generate video:// protocol paths (without target size)
                # Target size will be calculated and passed by the caller
                del vr  # Release VideoReader
                return [f"video://{mp4_path}::{idx}" for idx in frame_indices]
            except Exception as e:
                logging.warning(f"Failed to read mp4 file {mp4_path}: {e}")
                return []
        
        # Check for image sequence directory
        img_dir = osp.join(search_dir, view)
        if osp.exists(img_dir) and osp.isdir(img_dir):
            # Image sequence format
            try:
                # Look for .jpg or .png files
                all_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    all_files.extend(glob.glob(osp.join(img_dir, ext)))
                
                # Use natural sort to handle numeric sequences correctly (e.g., 2.jpg < 10.jpg)
                all_files = sorted(all_files, key=natural_sort_key)
                
                if not all_files:
                    logging.warning(f"No image files found in {img_dir}")
                    return []
                
                # Apply frame range if specified
                if frame_start is not None and frame_end is not None:
                    start = max(0, frame_start)
                    end = min(len(all_files) - 1, frame_end)
                    selected_files = all_files[start:end + 1]
                else:
                    selected_files = all_files
                
                return selected_files
            except Exception as e:
                logging.warning(f"Failed to list image files in {img_dir}: {e}")
                return []
        
        logging.warning(f"No valid data found for view '{view}' in {search_dir}")
        return []
    
    def _instruction_filenames_for_dataset(self, dataset_name):
        """Ordered basenames: dataset-specific first (if configured), then instruction.txt."""
        default = "instruction.txt"
        if dataset_name is None:
            return [default]
        alt = DATASET_INSTRUCTION_FILENAME.get(str(dataset_name))
        if not alt or alt == default:
            return [default]
        return [alt, default]

    def _get_task_description(self, video_info):
        """
        Task text from instruction file(s). Per directory (segment first, then episode root),
        tries DATASET_INSTRUCTION_FILENAME[dataset] if set, else instruction.txt.

        Args:
            video_info: dict with 'video_dir', optional 'segment_id', optional 'dataset'
        """
        video_dir = video_info['video_dir'] if isinstance(video_info, dict) else video_info
        segment_id = video_info.get('segment_id') if isinstance(video_info, dict) else None
        dataset_name = video_info.get('dataset') if isinstance(video_info, dict) else None

        search_bases = []
        if segment_id is not None:
            search_bases.append(osp.join(video_dir, str(segment_id)))
        search_bases.append(video_dir)

        for base in search_bases:
            for fn in self._instruction_filenames_for_dataset(dataset_name):
                instruction_path = self._apply_path_transform(osp.join(base, fn), dataset_name)
                if not osp.exists(instruction_path):
                    continue
                try:
                    with open(instruction_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    logging.warning(f"Failed to read {instruction_path}: {e}")

        self._data_warn(
            "missing_instruction_file",
            f"No instruction file found for {video_dir}, skipping sample (resample in __getitem__)",
        )
        raise SilentFilterError(f"missing instruction file under {video_dir}")

    def _load_text_embedding(self, video_info):
        """Load pre-computed T5 embedding from instruction.pt (same dir as instruction.txt)."""
        video_dir = video_info['video_dir'] if isinstance(video_info, dict) else video_info
        segment_id = video_info.get('segment_id') if isinstance(video_info, dict) else None
        dataset_name = video_info.get('dataset') if isinstance(video_info, dict) else None

        search_bases = []
        if segment_id is not None:
            search_bases.append(osp.join(video_dir, str(segment_id)))
        search_bases.append(video_dir)

        for base in search_bases:
            candidate = self._apply_path_transform(osp.join(base, "instruction.pt"), dataset_name)
            if osp.exists(candidate):
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                context = payload["context"]        # [L, D]
                context_mask = payload["mask"].bool()  # [L]
                return context, context_mask  # variable length; collator handles padding

        raise FileNotFoundError(f"instruction.pt not found in {search_bases}")

    # ── Mapping init helpers (called once from __init__) ──────────────────────

    def _init_cam_mapping(self) -> None:
        """Eagerly load all *_cam_mapping.json files into self._cam_mapping_cache.

        Cache: {dataset_name: {normpath(task_path): cam_list}}
        task_path = dirname(episode_dir) using the pre-transform raw path.
        """
        self._cam_mapping_cache: dict = {}
        if not osp.isdir(self.cam_mapping_dir):
            logging.warning("[_init_cam_mapping] cam_mapping_dir not found: %s", self.cam_mapping_dir)
            return
        for fname in os.listdir(self.cam_mapping_dir):
            if not fname.endswith('_cam_mapping.json'):
                continue
            ds    = fname[:-len('_cam_mapping.json')]
            fpath = osp.join(self.cam_mapping_dir, fname)
            try:
                with open(fpath, 'r') as f:
                    raw = json.load(f)
                # Pre-build normpath index for O(1) task-level lookup
                self._cam_mapping_cache[ds] = {osp.normpath(k): v for k, v in raw.items()}
                logging.info("[_init_cam_mapping] Loaded cam_mapping for '%s' (%d tasks)", ds, len(raw))
            except Exception as e:
                logging.warning("[_init_cam_mapping] Failed to load %s: %s", fpath, e)

    def _init_joint_action_mapping(self) -> None:
        """Eagerly load all *_joint_action_mapping.json files into self._joint_action_mapping_cache.

        Cache: {dataset_name: mapping_dict}
        """
        self._joint_action_mapping_cache = load_joint_action_mapping_cache(
            self.joint_action_mapping_dir
        )

    def _get_cam_list_from_mapping(self, video_info, dataset_name: str):
        """Return the ordered cam list for this episode from the cam_mapping JSON, or None.

        Lookup key = normpath(dirname(video_dir)) using the pre-transform raw path.
        Returns None if no mapping file exists for this dataset or if the task path
        is not found in the index. Caller logs and raises CamMappingError.
        """
        cache = self._cam_mapping_cache.get(dataset_name)
        if cache is None:
            return None
        video_dir = video_info['video_dir'] if isinstance(video_info, dict) else video_info
        task_path = osp.normpath(osp.dirname(video_dir.rstrip('/')))
        cam_list = cache.get(task_path)
        return cam_list   # list[str] or None

    def _select_views_from_cam_list(self, cam_list: list) -> list:
        """Sample n_views camera names from cam_list using self.num_view_probs.

        Rules:
          - Eligible n_views = keys of num_view_probs that are ≤ len(cam_list).
          - If no eligible count exists, clamp to len(cam_list) with a warning.
          - n_views == len(cam_list) → return all cameras.
          - n_views <  len(cam_list) → random subset of indices, output in cam_list order.
        """
        max_views = len(cam_list)
        eligible  = [n for n in self.num_view_probs if n <= max_views]
        if not eligible:
            logging.warning(
                "[_select_views_from_cam_list] num_view_probs keys %s all exceed "
                "cam_list length %d — using all %d cameras.",
                list(self.num_view_probs.keys()), max_views, max_views,
            )
            return list(cam_list)
        weights = [self.num_view_probs[n] for n in eligible]
        n_views = random.choices(eligible, weights=weights, k=1)[0]
        if n_views >= max_views:
            return list(cam_list)
        idx = sorted(random.sample(range(max_views), n_views))
        return [cam_list[i] for i in idx]

    def _select_view_combination(self, dataset_name, available_views, video_info=None):
        """
        Select camera views from cam_mapping JSON and num_view_probs subsampling.

        Args:
            dataset_name: str, dataset identifier
            available_views: list[str], available view names on disk
            video_info: dict with 'video_dir'; required for mapping lookup

        Returns:
            list[str]: Ordered view names.
                views[0] is the primary view (frame-count reference, task/progress lookup).
                views[1:] are extra views (additional cameras beyond the primary).

        Raises:
            CamMappingError: no mapping file, task not in mapping, empty cam_list, or disk missing views.
        """
        if video_info is None:
            logging.warning(
                "[_select_view_combination] video_info is None; "
                "cam_mapping requires video_info (dataset=%s).",
                dataset_name,
            )
            raise CamMappingError(
                "video_info is required for cam_mapping-based view selection"
            )

        cam_list = self._get_cam_list_from_mapping(video_info, dataset_name)
        if cam_list is None:
            if dataset_name not in self._cam_mapping_cache:
                logging.warning(
                    "[_select_view_combination] No cam_mapping file for dataset %r "
                    "(expected %s_cam_mapping.json under cam_mapping_dir).",
                    dataset_name,
                    dataset_name,
                )
                raise CamMappingError(
                    f"No cam_mapping file for dataset {dataset_name!r}"
                )
            video_dir = video_info["video_dir"] if isinstance(video_info, dict) else video_info
            task_path = osp.normpath(osp.dirname(video_dir.rstrip("/")))
            logging.warning(
                "[_select_view_combination] task_path %r not in cam_mapping for dataset %r",
                task_path,
                dataset_name,
            )
            raise CamMappingError(
                f"task_path {task_path!r} not in cam_mapping for dataset {dataset_name!r}"
            )

        if not cam_list:
            video_dir = video_info["video_dir"] if isinstance(video_info, dict) else video_info
            task_path = osp.normpath(osp.dirname(video_dir.rstrip("/")))
            raise CamMappingError(f"Empty cam_list in mapping for task {task_path!r}")

        video_dir = video_info["video_dir"] if isinstance(video_info, dict) else video_info
        missing = [v for v in cam_list if v not in available_views]
        if missing:
            logging.warning(
                "[_select_view_combination] Disk missing views required by cam_mapping: %s "
                "(available_views=%s, video_dir=%s)",
                missing,
                available_views,
                video_dir,
            )
            raise CamMappingError(
                f"cam_mapping requires views {missing} not on disk (available={available_views!r})"
            )

        return self._select_views_from_cam_list(cam_list)
    
    @staticmethod
    def _load_excluded_episode_paths(exclude_json_path: str) -> set:
        """Load a set of episode original_paths to exclude from training.

        Args:
            exclude_json_path: Comma-separated path(s) to JSON file(s), each containing
                a list of episode path strings (same format as the training data JSON).
                Multiple files are unioned together.

        Returns:
            set of path strings; matched against item['original_path'] in self.data.

        Raises:
            ValueError: if any JSON file does not contain a list.
        """
        excluded: set = set()
        for single_path in [p.strip() for p in exclude_json_path.split(',') if p.strip()]:
            with open(single_path) as f:
                paths = json.load(f)
            if not isinstance(paths, list):
                raise ValueError(
                    f"exclude_episode_json must contain a JSON list, got {type(paths).__name__} "
                    f"in {single_path}"
                )
            excluded.update(paths)
        return excluded

    def _load_from_json(self, json_path_arg):
        """
        Lazy JSON loading: only load video path list and metadata.
        Item structure is constructed dynamically in _get_single_item().
        
        Supports multiple input formats:
            - Single JSON file path: "/path/to/dataset1.json"
            - Comma-separated paths: "/path/to/dataset1.json,/path/to/dataset2.json"
            - List of paths: ["/path/to/dataset1.json", "/path/to/dataset2.json"]
        
        Args:
            json_path_arg: str or list, JSON file path(s)
        
        Returns:
            list: List of parsed video path dicts with metadata
                [{
                    'video_dir': str,
                    'segment_id': int or None,
                    'frame_start': int or None,
                    'frame_end': int or None,
                    'dataset': str,  # extracted from path or filename
                    'original_path': str
                }, ...]
        """
        # Parse input to list of JSON file paths
        if isinstance(json_path_arg, str):
            json_paths = [p.strip() for p in json_path_arg.split(',')]
        elif isinstance(json_path_arg, list):
            json_paths = json_path_arg
        else:
            raise ValueError(f"Invalid json_path type: {type(json_path_arg)}. Expected str or list.")
        
        # Helper to extract dataset name from JSON filename
        def get_dataset_name_from_file(json_path):
            """
            Extract dataset name from JSON filename.
            Examples:
                - /path/to/berkeley_autolab_ur5_video_paths.json -> berkeley_autolab_ur5
                - /path/to/agibot_data.json -> agibot_data
            """
            base = osp.basename(json_path)
            name = base.replace('_video_paths.json', '').replace('.json', '')
            return name
        
        # Load and parse paths from each JSON file
        all_parsed_data = []
        dataset_stats = {}  # Track counts per dataset
        
        for json_path in json_paths:
            if not osp.exists(json_path):
                logging.warning(f"JSON file not found: {json_path}, skipping")
                continue
            
            print(f"📂 Loading video paths from: {json_path}")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    path_list = json.load(f)
                
                if not isinstance(path_list, list):
                    logging.warning(f"JSON file {json_path} content is not a list, skipping")
                    continue
                
                # Extract dataset name from filename (can be overridden by path parsing)
                file_dataset_name = get_dataset_name_from_file(json_path)
                
                # Parse each path and attach metadata
                parsed_batch = []
                for path_str in path_list:
                    # Pass dataset name to enable segment parsing for whitelisted datasets
                    parsed = self._parse_video_path(path_str, dataset_name=file_dataset_name)
                    # Use dataset name extracted from JSON filename
                    parsed['dataset'] = file_dataset_name
                    parsed_batch.append(parsed)
                
                # Update statistics
                dataset_name = parsed_batch[0]['dataset'] if parsed_batch else file_dataset_name
                dataset_stats[dataset_name] = dataset_stats.get(dataset_name, 0) + len(parsed_batch)
                
                all_parsed_data.extend(parsed_batch)
                print(f"  ✓ Loaded {len(parsed_batch)} paths from {osp.basename(json_path)}")
                
            except Exception as e:
                logging.error(f"Failed to load JSON file {json_path}: {e}")
                continue
        
        if not all_parsed_data:
            raise ValueError("No valid video paths loaded from JSON files")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("📊 JSON Data Loading Summary")
        print("="*60)
        print(f"Total JSON files loaded: {len([p for p in json_paths if osp.exists(p)])}")
        print(f"Total video paths: {len(all_parsed_data)}")
        print("\nDataset Distribution:")
        for dataset_name, count in sorted(dataset_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_parsed_data)) * 100
            print(f"  • {dataset_name:30s}: {count:6d} paths ({percentage:5.1f}%)")
        print("="*60 + "\n")
        
        return all_parsed_data
    
    def _construct_json_item(self, index, forced_views=None):
        """
        Dynamically construct item structure from JSON path data.
        Uses cam_mapping + num_view_probs to select camera views.

        Args:
            index: int, index into self.data
            forced_views: list[str] or None.  When provided, steps 1-2 (view
                discovery and selection) are skipped and these views are used
                directly.  Useful for evaluation where the view set is fixed.

        Returns:
            dict: Compatible with PKL format
                {
                    'text': str,          # task description
                    'views_images': list[list[str]],  # [view][frame]
                    'action': np.ndarray or None,
                    'joint_all': np.ndarray or None,
                    'dataset': str,
                    '_selected_views': list[str],     # view names used
                    '_progress_list': list[float] or None,
                    '_indicator_list': list[str] or None,  # when use_indicator_prompt
                    '_task_video_available': bool,
                    '_ep_raw': dict or None,          # raw episode dict (for eval)
                }
        """
        video_info = self.data[index]
        dataset_name = video_info['dataset']

        if forced_views is not None:
            # Eval path: skip view discovery and selection.
            views = list(forced_views)
        else:
            # 1. Get available views
            available_views = self._get_available_views(video_info)

            if not available_views:
                logging.error(f"No views found for index {index}, path: {video_info['original_path']}")
                return {
                    'text': 'error',
                    'views_images': [[]],
                    'action': None,
                    'dataset': dataset_name,
                    '_selected_views': [],
                    '_ep_raw': None,
                }

            # 2. Select view combination using cam_mapping
            # views[0] = primary view (frame-count reference); views[1:] = extra views
            views = self._select_view_combination(dataset_name, available_views, video_info=video_info)  # list[str]

        # 3. Load frame paths for all selected views
        views_images = [self._get_files(video_info, view=v) for v in views]
        primary_paths = views_images[0]

        # Filter: Check if sample has enough frames based on physical time
        # Get dataset FPS, default to 10 if not found
        dataset_fps = self.dataset_fps.get(dataset_name, 10)
        required_frames = dataset_fps * MIN_PHYSICAL_SECONDS_THRESHOLD

        if len(primary_paths) < required_frames:
            raise SilentFilterError(f"Insufficient frames: {len(primary_paths)} < {required_frames}")

        # 4. Get task description from instruction.txt
        task_text = self._get_task_description(video_info)

        # 4b. Load pre-computed T5 text embedding
        try:
            text_context, text_context_mask = self._load_text_embedding(video_info)
        except FileNotFoundError:
            logging.warning(f"instruction.pt not found for {video_info.get('video_dir')}, using dummy")
            text_context = torch.zeros(0, 4096)
            text_context_mask = torch.zeros(0, dtype=torch.bool)

        # 5. Load action and joint data.
        # _load_action_and_joint returns normalised arrays + the raw episode dict.
        # The raw dict is used by steps 5c/5d when relative-action features are enabled;
        # it is otherwise ignored and will be GC'd.
        action_data = None
        joint_all   = None   # float32 [T, D_joint] — all frames; indexed after sampling
        _ep_raw     = None   # raw episode dict (raw_actions, col_map, action_keys, nmd, …)
        if self.actions:
            action_data, joint_all, _ep_raw = self._load_action_and_joint(video_info)

            # action, joint, and image frame counts must all agree — no silent truncation.
            if action_data is not None:
                if len(action_data) != len(primary_paths):
                    raise SilentFilterError(
                        f"action length {len(action_data)} != image frames {len(primary_paths)} "
                        f"for {video_info['video_dir']}"
                    )
                if joint_all is not None and len(joint_all) != len(action_data):
                    raise SilentFilterError(
                        f"joint length {len(joint_all)} != action length {len(action_data)} "
                        f"for {video_info['video_dir']}"
                    )

            if self.skip_no_action and action_data is None:
                raise SilentFilterError("missing action data")

        # 5b-pre. Load progress info and task-video flag against the full (pre-downsample) primary_paths.
        # load_progress_info only needs primary_paths[0] for directory lookup; the returned dict
        # covers all original frames, so we must build progress_list here and filter it in sync
        # with every subsequent index-selection step (5b downsampling, 5d static-frame removal).
        agent_progress_info  = self.load_progress_info(primary_paths, dataset_name)
        task_video_available = self.check_task_video_available(primary_paths, dataset_name)

        if agent_progress_info is not None:
            sorted_keys   = sorted([int(k) for k in agent_progress_info.keys()])
            progress_list = [agent_progress_info[str(k)] for k in sorted_keys]
            if len(progress_list) != len(primary_paths):
                raise SilentFilterError(
                    f"progress_list length {len(progress_list)} != "
                    f"primary_paths length {len(primary_paths)}"
                )
        else:
            progress_list = None

        indicator_list = None
        if self.use_indicator_prompt and not self.use_indicator_positive:
            agent_indicator_info = self.load_indicator_info(primary_paths, dataset_name)
            if agent_indicator_info is not None:
                sorted_ind_keys = sorted([int(k) for k in agent_indicator_info.keys()])
                indicator_list = [agent_indicator_info[str(k)] for k in sorted_ind_keys]
                if len(indicator_list) != len(primary_paths):
                    raise SilentFilterError(
                        f"indicator_list length {len(indicator_list)} != "
                        f"primary_paths length {len(primary_paths)}"
                    )

        # 5c. Compute relative actions for filtering / round-trip verification.
        # 5d. Static-frame removal.
        # Greedy forward scan: from the current anchor frame, advances to the first
        # subsequent frame whose displacement exceeds threshold, then repeats.
        # Filtering is applied to raw_actions (and to normalised action_data when abs
        # actions are used directly).  Both branches also filter views_images, joint_all,
        # and progress_list.
        _total_frames_before = len(primary_paths)   # snapshot before any filtering
        if self.remove_static_frames \
                and _ep_raw is not None and _ep_raw.get('raw_actions') is not None:
            active_indices = self._get_active_indices(
                _ep_raw['raw_actions'],
                _ep_raw['action_keys'],
                _ep_raw['col_map'],
                self.static_rot_threshold,
                self.static_trans_threshold,
                threshold_gripper=self.static_gripper_threshold,
            )
            if len(active_indices) < len(primary_paths):
                views_images, action_data, joint_all, progress_list, indicator_list = self._apply_indices(
                    active_indices, views_images, action_data, joint_all, progress_list, indicator_list
                )
                primary_paths = views_images[0]
                # raw_actions must be filtered so step 5e sees the correct sequence.
                _ep_raw['raw_actions'] = _ep_raw['raw_actions'][active_indices]
                if _ep_raw.get('follower_actions') is not None:
                    _ep_raw['follower_actions'] = _ep_raw['follower_actions'][active_indices]
                # raw_joint_entries must also be filtered to stay in sync.
                if _ep_raw.get('raw_joint_entries'):
                    _ep_raw['raw_joint_entries'] = [
                        _ep_raw['raw_joint_entries'][i] for i in active_indices
                    ]
            static_keep_ratio = len(active_indices) / _total_frames_before
        else:
            static_keep_ratio = 1.0

        # 5e. Relative action conversion is deferred to _get_single_item()
        # after chunk selection, where follower_ref at the chunk start is known.

        # 7. Construct unified structure
        return {
            'text': task_text,
            'views_images': views_images,         # list[list[str]]: [v_idx][frame_idx]
            'action': action_data,                # np.float32 [T, D_action] or None
            'joint_all': joint_all,               # np.float32 [T, D_joint] or None
            'dataset': dataset_name,
            '_selected_views': views,             # list[str]: stored for task-video view alignment
            '_progress_list': progress_list,      # list[float] sorted by frame index, or None
            '_indicator_list': indicator_list,    # list[str] per frame, or None
            '_task_video_available': task_video_available, # bool
            '_ep_raw': _ep_raw,                   # raw episode dict; None if actions not loaded
            'static_keep_ratio': static_keep_ratio,  # float in (0, 1]: frames retained after static removal
            'context': text_context,
            'context_mask': text_context_mask,
        }
    
    def _infer_view_from_paths(self, paths):
        """
        Infer view name from frame paths (fallback when agent_selected_views not available).
        
        Examples:
            "/data/task1/images/frame_0.jpg" -> "images"
            "/data/task1/front/0001.jpg" -> "front"
            "video:///data/task1/images.mp4::0::256x256" -> "images"
            "video:///data/task1/wrist1.mp4::0::128x128" -> "wrist1"
        
        Args:
            paths: list[str], frame paths
        
        Returns:
            str: View name (default to 'images')
        """
        if not paths or len(paths) == 0:
            return 'images'
        
        first_path = paths[0]
        
        # Handle video:// protocol
        if isinstance(first_path, str) and first_path.startswith("video://"):
            # Parse: "video:///path/to/view.mp4::0::256x256"
            path_part = first_path[8:].split('::')[0]  # Get video file path
            # Extract view from filename (e.g., "images.mp4" -> "images")
            view_name = osp.splitext(osp.basename(path_part))[0]
            return view_name
        
        # Handle regular file path
        if isinstance(first_path, str):
            # Get parent directory name (e.g., "/data/task1/images/frame_0.jpg" -> "images")
            parent_dir = osp.basename(osp.dirname(first_path))
            return parent_dir
        
        return 'images'  # Default fallback
    
    # ── Joint-action mapping helpers ──────────────────────────────────────────

    def _load_joint_action_mapping(self, dataset_name: str):
        """Look up the pre-loaded joint_action_mapping for a dataset.

        All mapping JSONs are loaded at __init__ time; this method is a
        pure cache lookup. Returns None (with a one-time warning) if no
        mapping file was found for the given dataset.
        """
        if dataset_name in self._joint_action_mapping_cache:
            return self._joint_action_mapping_cache[dataset_name]
        # Dataset not present in any pre-loaded mapping file
        logging.warning(
            "[_load_joint_action_mapping] No mapping for dataset '%s' — "
            "action and joint_data will be None.", dataset_name,
        )
        self._joint_action_mapping_cache[dataset_name] = None  # cache the miss
        return None

    def _resolve_mapping_entry(self, mapping: dict, video_dir: str):
        """Select the matching entry from a joint_action_mapping dict.

        Single-key dicts: return the unique value unconditionally.
        Multi-key dicts: derive sub_data_path = dirname(dirname(video_dir)) and
                         find the first key that is a prefix of sub_data_path.
        """
        if len(mapping) == 1:
            return next(iter(mapping.values()))
        sub_data_path = osp.dirname(osp.dirname(video_dir.rstrip('/')))
        for key, value in mapping.items():
            if sub_data_path.startswith(key):
                return value
        logging.warning(
            "[_resolve_mapping_entry] No matching key for sub_data_path=%r in mapping keys=%s",
            sub_data_path, list(mapping.keys()),
        )
        return None

    def _parse_field_norms(self, field_key: str, nmd: dict):
        """Extract (min_vals, delta_vals) for a field from norm_min_delta.

        Raises RuntimeError if the field is absent or any value is 'nan'.
        """
        return parse_field_norms(field_key, nmd)

    def _build_norm_vectors(self, action_keys, nmd, ds_factor=1,
                            use_relative=False, use_6d_rotation=False):
        """Build flat (nm, nd) float32 norm arrays for a list of action keys.

        Handles three modes uniformly so that both training normalisation and
        eval unnormalisation use identical logic:

          use_relative=False, use_6d_rotation=False  →  absolute, Euler
          use_relative=False, use_6d_rotation=True   →  absolute, 6D rotation
          use_relative=True,  use_6d_rotation=False  →  relative, position norms from _relative suffix
          use_relative=True,  use_6d_rotation=True   →  relative, rotation uses 6D identity norm

        Args:
            action_keys    : list[str]  active absolute action field names
            nmd            : dict       norm_min_delta dict from mapping entry
            ds_factor      : int        action_downsample_factor (relative only)
            use_relative   : bool       look up key+"_relative" and scale pos/rot
            use_6d_rotation: bool       expand rotation fields to 6D identity norm

        Returns:
            (nm, nd): float32 [D] arrays; nm = norm_min, nd = norm_delta
        """
        return build_norm_vectors(
            action_keys,
            nmd,
            ds_factor=ds_factor,
            use_relative=use_relative,
            use_6d_rotation=use_6d_rotation,
        )

    def _load_episode_raw(self, video_info):
        """Open the episode JSON once and return raw (un-normalised) arrays + norm parameters.

        All JSON-open, key-resolution, and array-assembly logic lives here.
        Normalization is a single one-liner that callers apply themselves:
            np.clip(2*(raw - nm) / nd - 1, -1, 1)

        Returns a dict on success, None on failure:
            raw_actions : float32 [T, D_action]  assembled (incl. 6D if use_6d_rotation)
            raw_joints  : float32 [T, D_joint]   assembled
            col_map     : dict[str, (int, int)]  key -> (start_col, dim) in raw_actions
            a_nm, a_nd  : float32 [D_action]  action norm min / delta
            j_nm, j_nd  : float32 [D_joint]   joint  norm min / delta
            action_keys : list[str]  active action keys (for _relative suffix lookup)
            nmd         : dict  norm_min_delta from mapping entry
        """
        video_dir    = video_info['video_dir'] if isinstance(video_info, dict) else video_info
        dataset_name = video_info.get('dataset') if isinstance(video_info, dict) else None

        # A. Resolve mapping entry
        mapping = self._load_joint_action_mapping(dataset_name)
        if mapping is None:
            return None   # warning already emitted
        entry = self._resolve_mapping_entry(mapping, video_dir)
        if entry is None:
            return None

        action_keys = entry.get('action_keys', [])
        joint_keys  = entry.get('joint_keys',  [])
        nmd         = entry['norm_min_delta']

        # B. Locate episode JSON — raw video_dir, no path transform
        video_dir     = video_dir.rstrip('/')
        ep_name       = osp.basename(video_dir)
        json_path     = osp.join(video_dir, f"{ep_name}.json")
        bad_json_path = osp.join(video_dir, "bad_action.json")

        if not osp.exists(json_path):
            if osp.exists(bad_json_path):
                return None
            self._data_warn(
                "action_json_not_found",
                f"Action JSON not found at: {json_path}",
            )
            return None

        try:
            with open(json_path, 'r') as f:
                raw = json.load(f)
            entries     = raw['data']
            first_entry = entries[0]

            # C. Action: build active key list + flat norm arrays + assemble raw matrix
            # When use_6d_rotation=True, keys ending in "_rotation" are converted from
            # ZYX Euler [roll, pitch, yaw] → 6D (first two columns of SO(3) matrix).
            # 6D values are in [-1, 1], so we set min=-1, delta=2 → identity passthrough.
            # col_map records (start_offset, dim) for each active key so that relative-
            # action computation can locate fields without hardcoding key names.
            # Pass 1: build col_map (key → (start_col, dim)) and filter missing keys.
            # Norm vectors are built separately via _build_norm_vectors to share logic
            # with _normalize_relative_actions and RealEpisodeDataset._build_action_norms.
            # Assemble raw actions via shared static method (reused by eval pipeline).
            raw_actions, col_map, active_action_keys = self._assemble_raw_actions(
                entries, action_keys, self.use_6d_rotation
            )

            # Build flat norm arrays via the unified helper.
            a_nm = a_nd = None
            if active_action_keys:
                a_nm, a_nd = self._build_norm_vectors(
                    active_action_keys, nmd,
                    use_relative=False, use_6d_rotation=self.use_6d_rotation
                )
            else:
                self._data_warn("no_active_action_fields",
                    f"No active action fields in {json_path}")

            # D. Joint: same pattern, assemble raw matrix
            raw_joints = None
            j_nm = j_nd = None
            raw_joint_entries = []
            if joint_keys:
                active_joint_keys = []
                for key in joint_keys:
                    if key not in first_entry:
                        self._data_warn("missing_joint_key",
                            f"Joint key '{key}' missing in {json_path}, skipping.")
                        continue
                    active_joint_keys.append(key)

                if active_joint_keys:
                    j_nm, j_nd = build_joint_norm_vectors(
                        active_joint_keys, nmd,
                        use_6d_rotation=self.use_6d_rotation,
                    )
                    # Identify joint rotation keys for 6D conversion
                    joint_rot6d_key_set = set()
                    if self.use_6d_rotation:
                        for key in active_joint_keys:
                            if key.endswith("_rotation"):
                                joint_rot6d_key_set.add(key)
                    rows = []
                    # Store raw joint entries (pre-conversion) for eval dict reconstruction
                    raw_joint_entries = []
                    for e in entries:
                        vec = []
                        joint_entry = {key: e[key] for key in active_joint_keys}
                        raw_joint_entries.append(joint_entry)
                        for key in active_joint_keys:
                            if key in joint_rot6d_key_set:
                                # [roll, pitch, yaw] ZYX extrinsic → SO(3) → first two columns → 6D
                                rpy = e[key]
                                R = _SciRotation.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()
                                vec.extend([R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1]])
                            else:
                                val = e[key]
                                vec.extend(val) if isinstance(val, (list, tuple)) else vec.append(val)
                        rows.append(vec)
                    raw_joints = np.array(rows, dtype=np.float32)   # [T, D_joint]

            # Follower data for relative action computation (master → follow key mapping)
            follower_actions = None
            if self.use_relative_action and active_action_keys:
                follow_keys = [k.replace('master', 'follow') for k in active_action_keys]
                follower_actions, _, _ = self._assemble_raw_actions(
                    entries, follow_keys, self.use_6d_rotation
                )
                if follower_actions is None:
                    logging.warning(
                        "[_load_episode_raw] Failed to assemble follower actions "
                        "for relative mode; follow_keys=%s", follow_keys,
                    )

            return {
                'raw_actions':  raw_actions,
                'follower_actions': follower_actions,
                'raw_joints':   raw_joints,
                'raw_joint_entries': raw_joint_entries,
                'col_map':      col_map,
                'a_nm': a_nm,  'a_nd': a_nd,
                'j_nm': j_nm,  'j_nd': j_nd,
                'action_keys':  active_action_keys,
                'joint_keys':   active_joint_keys if joint_keys else [],
                'nmd':          nmd,
            }

        except Exception as e:
            logging.warning(f"[_load_episode_raw] Failed for {json_path}: {e}")
            return None

    def _load_action_and_joint(self, video_info):
        """Load episode actions and joints, return raw (unnormalized) arrays.

        Action normalization is deferred to _get_single_item() so that
        relative-action conversion can be applied to the selected chunk first.

        Returns:
            action_data: float32 [T, D_action] **unnormalized** raw actions, or None.
            joint_data:  float32 [T, D_joint]  normalised to [-1, 1], or None.
            ep_raw:      dict returned by _load_episode_raw (contains raw_actions, col_map,
                         a_nm, a_nd, action_keys, nmd, etc.); None if loading failed.
        """
        r = self._load_episode_raw(video_info)
        if r is None:
            return None, None, None
        # Actions stay raw — normalization happens in _get_single_item after chunk selection.
        action_data = r['raw_actions']
        joint_data  = np.clip(2.0 * (r['raw_joints']  - r['j_nm']) / r['j_nd'] - 1.0, -1.0, 1.0) \
                      if r['raw_joints']  is not None else None
        return action_data, joint_data, r

    # ── Preprocessing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _apply_indices(indices, views_images, action_data, joint_all, progress_list=None,
                       indicator_list=None):
        """Apply an index subset to all episode arrays simultaneously.

        Called in _construct_json_item for static-frame removal (and may be used for striding).

        Args:
            indices      : list[int] or np.ndarray — selected frame positions
            views_images : list[list[str]] — [v_idx][frame_idx]
            action_data  : np.ndarray [T, D] or None
            joint_all    : np.ndarray [T, D] or None
            progress_list: list[float] or None
            indicator_list: list[str] or None — per-frame positive/negative labels

        Returns:
            (views_images, action_data, joint_all, progress_list, indicator_list) after selection
        """
        views_images = [[vp[i] for i in indices] for vp in views_images]
        if action_data    is not None: action_data    = action_data[indices]
        if joint_all      is not None: joint_all      = joint_all[indices]
        if progress_list  is not None: progress_list  = [progress_list[i] for i in indices]
        if indicator_list is not None:
            indicator_list = [indicator_list[i] for i in indices]
        return views_images, action_data, joint_all, progress_list, indicator_list

    @staticmethod
    def _compute_relative_actions(action_chunk, action_keys, col_map, follower_ref):
        """Compute master actions relative to follower reference state.

        First-frame reference: all frames relative to follower at chunk start.
        Aligned with x2robot_dataset (fix commit 1b21021).

        Field type detected from key name:
          *_position → world-frame subtraction: pos_master[t] - pos_follower_ref
          *_rotation → R_master[t] @ R_follower_ref^T → 6D (first two columns)
          other      → copied as-is (e.g. gripper)

        Rotation input may be 3D Euler ZYX or 6D matrix columns; both converted
        to SO(3). Output rotation is always 6D (first two columns of R_rel).

        Args:
            action_chunk: float [T, D_action] — unnormalized absolute master actions
            action_keys:  list[str]           — active field names (in column order)
            col_map:      dict[str, (int,int)] — key → (start_col, dim) in action_chunk
            follower_ref: float [D_action]    — follower state at observation time

        Returns:
            (float64 [T, D_rel], dict[str, int]):
              *_position → 3 cols (world-frame delta)
              *_rotation → 6 cols (6D rotation)
              other      → original dim cols
        """
        T   = len(action_chunk)
        raw = action_chunk.astype(np.float64)
        ref = follower_ref.astype(np.float64)

        # Helper: recover [N, 3, 3] rotation matrix from data array columns
        def _R_from_col(data, start, dim):
            if dim == 6:
                a = data[:, start:start+3]
                b = data[:, start+3:start+6]
                c = np.cross(a, b)                             # [N, 3]
                return np.stack([a, b, c], axis=-1)            # [N, 3, 3]
            else:
                ypr = data[:, start:start+3][:, ::-1]         # [roll,pitch,yaw] → [yaw,pitch,roll]
                return _SciRotation.from_euler(
                    "ZYX", ypr.reshape(-1, 3)).as_matrix()    # [N, 3, 3]

        # Output dimension: position→3, rotation→6, other→original
        out_dim = {}
        for key in action_keys:
            if '_rotation' in key:
                out_dim[key] = 6
            elif '_position' in key:
                out_dim[key] = 3
            else:
                out_dim[key] = col_map[key][1]

        out_result = np.zeros((T, sum(out_dim[k] for k in action_keys)), dtype=np.float64)
        out_off = 0
        out_start = {}
        for key in action_keys:
            out_start[key] = out_off
            out_off += out_dim[key]

        ref_2d = ref[np.newaxis, :]  # [1, D] for _R_from_col

        for key in action_keys:
            src_s, src_d = col_map[key]
            dst_s = out_start[key]

            if '_position' in key:
                # World-frame subtraction: pos_master[t] - pos_follower_ref
                pos_master = raw[:, src_s:src_s+3]             # [T, 3]
                pos_follow = ref[src_s:src_s+3]                # [3]
                out_result[:, dst_s:dst_s+3] = pos_master - pos_follow

            elif '_rotation' in key:
                # R_rel = R_master[t] @ R_follower_ref^T → 6D (first two columns)
                R_master = _R_from_col(raw, src_s, src_d)      # [T, 3, 3]
                R_follow = _R_from_col(ref_2d, src_s, src_d)[0]  # [3, 3]
                R_rel = R_master @ R_follow.T                  # [T, 3, 3]
                out_result[:, dst_s:dst_s+3]   = R_rel[:, :, 0]  # 6D col 1
                out_result[:, dst_s+3:dst_s+6] = R_rel[:, :, 1]  # 6D col 2

            else:
                out_result[:, dst_s:dst_s+src_d] = raw[:, src_s:src_s+src_d]

        return out_result, out_start

    def _normalize_relative_actions(self, rel_raw, action_keys, nmd, ds_factor: int = 1):
        """Normalise relative actions using per-key norms with a '_relative' suffix.

        Delegates norm-vector construction to _build_norm_vectors(use_relative=True)
        so that training normalisation and eval unnormalisation share identical logic.

        Args:
            rel_raw    : float64 [T, D_rel]  raw relative actions
            action_keys: list[str]           active abs action keys (same order / dim)
            nmd        : dict                norm_min_delta from mapping entry
            ds_factor  : int                 action_downsample_factor for this dataset

        Returns:
            float32 [T, D_rel] normalised to [-1, 1]
        """
        nm, nd = self._build_norm_vectors(
            action_keys, nmd, ds_factor=ds_factor,
            use_relative=True, use_6d_rotation=self.use_6d_rotation,
        )
        return np.clip(2.0 * (rel_raw.astype(np.float32) - nm) / nd - 1.0, -1.0, 1.0)

    @staticmethod
    def _assemble_raw_actions(entries, action_keys, use_6d_rotation):
        """Build raw action array and column map from episode JSON entries.

        Extracted from _load_episode_raw for reuse by eval pipeline.
        Determines field dimensions by inspecting the first entry (no nmd needed).

        Args:
            entries: list[dict] — episode JSON ``data`` entries.
            action_keys: list[str] — field names to extract.
            use_6d_rotation: bool — convert Euler → 6D rotation columns.

        Returns:
            (raw_actions [T, D] float32, col_map {key: (start, dim)}, active_keys list[str])
            or (None, None, None) if no active keys found.
        """
        first_entry = entries[0]
        active_keys = []
        col_map = {}
        col_offset = 0
        rot6d_key_set = set()
        for key in action_keys:
            if key not in first_entry:
                logging.warning("[_assemble_raw_actions] key '%s' missing, skipping.", key)
                continue
            active_keys.append(key)
            if use_6d_rotation and key.endswith("_rotation"):
                rot6d_key_set.add(key)
                dim = 6
            else:
                val = first_entry[key]
                dim = len(val) if isinstance(val, (list, tuple)) else 1
            col_map[key] = (col_offset, dim)
            col_offset += dim

        if not active_keys:
            return None, None, None

        rows = []
        for e in entries:
            vec = []
            for key in active_keys:
                if key in rot6d_key_set:
                    rpy = e[key]
                    R = _SciRotation.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()
                    vec.extend([R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1]])
                else:
                    val = e[key]
                    vec.extend(val) if isinstance(val, (list, tuple)) else vec.append(val)
            rows.append(vec)
        raw_actions = np.array(rows, dtype=np.float32)  # [T, D_action]
        return raw_actions, col_map, active_keys

    @staticmethod
    def _get_active_indices(
        raw_actions, action_keys, col_map,
        threshold_rot, threshold_trans, threshold_gripper=0.0,
    ):
        """Return indices of frames with sufficient motion in any arm/field.

        Uses a greedy forward scan: starting from frame 0 (anchor), advances to the
        first subsequent frame whose displacement from the anchor exceeds any threshold,
        makes that frame the new anchor, and repeats.  This guarantees roughly uniform
        motion spacing between kept frames and avoids bulk-removing continuous segments.

        Displacement metrics (anchor → candidate frame t):
          *_position : L2 distance  ||pos[t] - pos[anchor]||₂  > threshold_trans
          *_rotation : geodesic angle  arccos(clip((tr(R_anchor^T @ R[t]) - 1) / 2, -1, 1))
                       > threshold_rot  (rotation matrices precomputed in batch)
          *gripper*  : max absolute change  max|raw[t, cols] - raw[anchor, cols]|
                       > threshold_gripper  (only when threshold_gripper > 0)

        Frame 0 is always kept so the episode always has at least one frame.

        Args:
            raw_actions      : [T, D_abs]          raw absolute actions
            action_keys      : list[str]            active field names
            col_map          : dict[str, (int,int)] key → (start_col, dim)
            threshold_rot    : float                radians — geodesic rotation distance threshold
            threshold_trans  : float                metres  — L2 position distance threshold
            threshold_gripper: float                absolute change threshold for *gripper* fields.
                               0.0 (default) disables gripper check.

        Returns:
            list[int] sorted indices of kept frames
        """
        T = len(raw_actions)
        if T == 0:
            return []

        # --- Precompute rotation matrices [T, 3, 3] for each *_rotation key (batch) ---
        raw = raw_actions.astype(np.float64)
        R_all = {}
        for key in action_keys:
            if '_rotation' in key:
                s, d = col_map[key]
                if d == 6:
                    a = raw[:, s:s+3]; b = raw[:, s+3:s+6]
                    R_all[key] = np.stack([a, b, np.cross(a, b)], axis=-1)  # [T, 3, 3]
                else:  # dim == 3: Euler ZYX stored as [roll, pitch, yaw]
                    ypr = raw[:, s:s+3][:, ::-1]                            # flip → [yaw, pitch, roll]
                    R_all[key] = _SciRotation.from_euler("ZYX", ypr).as_matrix()  # [T, 3, 3]

        # --- Greedy forward scan ---
        kept   = [0]
        anchor = 0

        while anchor < T - 1:
            N    = T - anchor - 1                      # number of remaining candidate frames
            mask = np.zeros(N, dtype=bool)             # [N]: True → exceeds threshold from anchor

            for key in action_keys:
                s, d = col_map[key]

                if '_position' in key:
                    delta = raw[anchor+1:, s:s+3] - raw[anchor, s:s+3]      # [N, 3]
                    mask |= np.linalg.norm(delta, axis=1) > threshold_trans

                elif '_rotation' in key:
                    R_anchor_T  = R_all[key][anchor].T                       # [3, 3]
                    R_remaining = R_all[key][anchor+1:]                      # [N, 3, 3]
                    R_rel       = R_anchor_T @ R_remaining                   # [N, 3, 3] broadcast
                    traces      = np.einsum('nii->n', R_rel)                 # [N]
                    mask |= np.arccos(np.clip((traces - 1) / 2, -1., 1.)) > threshold_rot

                elif 'gripper' in key:
                    if threshold_gripper > 0.0:
                        delta = np.abs(raw[anchor+1:, s:s+d] - raw[anchor, s:s+d])  # [N, d]
                        mask |= np.any(delta > threshold_gripper, axis=1)

                else:
                    logging.warning(
                        f"[_get_active_indices] unrecognized action key '{key}' "
                        "(not *_position, *_rotation, or *gripper*) — skipped."
                    )

            hits = np.nonzero(mask)[0]
            if len(hits) == 0:
                break
            anchor = anchor + 1 + int(hits[0])
            kept.append(anchor)

        return kept

    # ==================== End of JSON Mode Helper Methods ====================

    def random_frames_to_tensor(self, views_paths, T, action_prompt=None, action_frames_list=None, dataset_name=None):
        """
        Sample frames from all camera views simultaneously using shared frame indices.

        Args:
            views_paths: list[list[str]] — ordered view paths; views_paths[0] is the primary view
                (used for frame-count validation and start_idx sampling).
            T: int — number of agent steps to sample.
            action_prompt: Optional action array (np.ndarray or similar).
            action_frames_list: Optional list of per-step frame intervals.
            dataset_name: str — dataset identifier for per-dataset image size lookup.

        Returns:
            If action_prompt is not None:
                (selected_actions, start_idx, frame_indices, resized_views, agent_unified_size)
            Else:
                (None, start_idx, frame_indices, resized_views, agent_unified_size)

            agent_unified_size: (width, height) from dataset_image_size config.
            resized_views: list[list[np.ndarray]] — [v_idx][frame_idx], primary view is [0].
                Frames are loaded at video_frame_indices (downsampled by action_video_freq_ratio).
        """
        # Frame count check on primary view (views_paths[0])
        primary_paths = views_paths[0]
        if len(primary_paths) < T:
            logging.error(f"❌ Sample has insufficient frames!")
            logging.error(f"   Required frames (T): {T}")
            logging.error(f"   Available frames: {len(primary_paths)}")
            first_path = str(primary_paths[0]) if primary_paths else 'EMPTY LIST'
            logging.error(f"   First image path: {first_path}")
            if primary_paths:
                if 'video://' in first_path:
                    logging.error(f"   Video source: {first_path.split('::')[0].replace('video://', '')}")
                else:
                    logging.error(f"   Parent directory: {osp.dirname(first_path)}")
            raise ValueError(f"Insufficient frames: need {T}, got {len(primary_paths)}")

        start_idx = random.randint(0, len(primary_paths) - T)

        # Build frame indices from action_frames_list or continuous fallback
        if action_frames_list is not None:
            frame_indices = []
            cumulative = start_idx
            for af in action_frames_list:
                frame_indices.append(cumulative)
                cumulative += af
        else:
            print('warning: continuous sampling')
            frame_indices = list(range(start_idx, start_idx + T))

        # Two-stage downsampling (no padding):
        # Stage 1: keyframes (random intervals) -> fixed-fps action indices at self.action_frames
        # Stage 2: action indices -> VAE-compatible video indices via segment-aware downsample
        video_frame_indices = self._downsample_segments_to_vae_indices(
            frame_indices, self.action_frames, self.action_video_freq_ratio, len(primary_paths)
        )

        agent_unified_size = self._get_target_size(dataset_name)

        resized_views = []
        agent_aug_params = self._sample_aug_params() if self.use_augmentation else None
        for vp in views_paths:
            v_frame_paths = [vp[min(idx, len(vp) - 1)] for idx in video_frame_indices]
            v_images = self._load_frame(v_frame_paths, target_size=agent_unified_size)
            if agent_aug_params is not None:
                v_images = self._apply_aug(v_images, agent_aug_params)
            resized_views.append([np.array(img, dtype=np.uint8) for img in v_images])

        # Return with or without actions
        if action_prompt is not None:
            selected_actions = action_prompt[frame_indices[0]:frame_indices[-1]]
            return (selected_actions, start_idx, frame_indices, resized_views, agent_unified_size)
        else:
            return (None, start_idx, frame_indices, resized_views, agent_unified_size)
    
    def pad_tensor(self, tensor, max_length, pad_value):
        """Pads a tensor to a specified maximum length."""
        current_length = tensor.shape[-1]
        if current_length < max_length:
            pad_length = max_length - current_length
            padding = torch.full((pad_length,), fill_value=pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=-1)
        return tensor

    def _build_action_tensor(self, action_raw, progress_values, max_action_len):
        """Build action tensor with progress and mask dimensions.

        Args:
            action_raw: np.ndarray [action_len, action_dim], normalized to [-1, 1]
            progress_values: list[float], len=action_len, range [0, 1]
            max_action_len: int, target length for padding (None = no padding)

        Returns:
            action_final: Tensor [L, action_dim + 2] where last 2 dims are [progress, mask]
            action_is_pad: BoolTensor [L]
        """
        action_len, action_dim = action_raw.shape
        target_len = max_action_len if max_action_len is not None else action_len

        # Pad action with last-frame value
        padded = np.zeros((target_len, action_dim), dtype=np.float32)
        valid_len = min(action_len, target_len)
        padded[:valid_len] = action_raw[:valid_len]
        if valid_len < target_len:
            padded[valid_len:] = action_raw[valid_len - 1]

        # Progress: normalize [0,1] -> [-1,1], pad with last value
        progress = np.zeros((target_len, 1), dtype=np.float32)
        for i in range(valid_len):
            progress[i, 0] = progress_values[i] * 2.0 - 1.0
        if valid_len < target_len:
            progress[valid_len:] = progress[valid_len - 1]

        # Mask: 1=valid, 0=padding
        mask = np.zeros((target_len, 1), dtype=np.float32)
        mask[:valid_len] = 1.0

        action_final = torch.from_numpy(
            np.concatenate([padded, progress, mask], axis=-1)
        ).float()

        action_is_pad = torch.zeros(target_len, dtype=torch.bool)
        action_is_pad[valid_len:] = True

        return action_final, action_is_pad

    @staticmethod
    def _downsample_to_vae_indices(start, end, ratio, total_available=None):
        """Compute VAE-compatible downsampled frame indices from a range.

        1. Determine raw frame count in [start, end]: raw_len = end - start + 1
        2. Downsample: n_down = max(2, raw_len // ratio)
        3. Adjust to VAE constraint (T-1) % 4 == 0: target_T = ((n_down-1+3)//4)*4 + 1
        4. Clamp target_T >= 5 (minimum useful video length)
        5. Uniformly sample target_T indices from [start, end]

        Args:
            start: first frame index (inclusive)
            end:   last frame index (inclusive)
            ratio: downsampling ratio (action_video_freq_ratio)
            total_available: max valid index + 1 (for clamping); None = no clamp

        Returns:
            list[int] of length target_T, sorted, all in [start, end]
        """
        raw_len = end - start + 1
        n_down = max(2, raw_len // ratio)
        # VAE constraint: (T - 1) must be divisible by 4
        target_T = ((n_down - 1 + 3) // 4) * 4 + 1
        target_T = max(5, target_T)  # at least 5 frames
        # Uniform sampling (no dedup — repeated frames are OK, preserves fixed target_T)
        indices = np.linspace(start, end, target_T, dtype=int).tolist()
        if total_available is not None:
            indices = [min(idx, total_available - 1) for idx in indices]
        return indices

    @staticmethod
    def _downsample_segments_to_vae_indices(keyframes, fps, ratio, total_available):
        """Downsample multi-segment keyframes to VAE-compatible video indices.

        Each segment is sampled with n points where (n-1)%4==0. Drop each
        segment's first point (natural dedup of shared boundaries), then
        prepend the first keyframe. Result: T = 1 + S*4k, (T-1)%4==0.

        All keyframes appear in the output. First/last frames guaranteed.

        Args:
            keyframes: list[int], sorted keyframe indices (len >= 2)
            fps: int, dataset fps (frames per keyframe interval)
            ratio: int, downsampling ratio (action_video_freq_ratio)
            total_available: int, max valid frame index + 1
        Returns:
            list[int], VAE-compatible frame indices
        """
        if len(keyframes) < 2:
            return list(keyframes)

        S = len(keyframes) - 1

        # n per segment based on dataset_fps / ratio, rounded to (n-1)%4==0
        raw = max(1, round(fps / ratio))
        n_per_seg = max(4, ((raw + 3) // 4) * 4) + 1  # n ∈ {5, 9, 13, ...}

        # Each segment: linspace → drop first point (dedup + VAE alignment)
        indices = [keyframes[0]]
        for i in range(S):
            pts = np.linspace(keyframes[i], keyframes[i + 1], n_per_seg, dtype=int)
            indices.extend(pts[1:].tolist())

        return [min(idx, total_available - 1) for idx in indices]

    def _frames_to_video_tensor(self, views_frames):
        """Convert per-view frame lists to video tensor with vertical camera concat.

        Args:
            views_frames: list[list[np.ndarray]] -- [v_idx][frame_idx], each HWC uint8
        Returns:
            video: Tensor [C, T, H, W], range [-1, 1]. Multi-view: concat along H.
        """
        view_tensors = []
        for v_frames in views_frames:
            arr = np.stack(v_frames, axis=0)  # [T, H, W, C]
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # [T, C, H, W]
            view_tensors.append(t)

        if len(view_tensors) == 1:
            video = view_tensors[0]
        else:
            video = torch.cat(view_tensors, dim=-2)  # vertical concat along H

        video = video / 255.0 * 2.0 - 1.0  # [0,255] -> [-1,1]
        video = video.permute(1, 0, 2, 3)   # [T,C,H,W] -> [C,T,H,W]
        return video

    def load_progress_info(self, image_paths_list, dataset_name=None):
        """
        从 info_dtw.json（若不存在则 info.json）加载进度信息
        
        Args:
            image_paths_list: 图像路径列表，如 scene["image"]
            dataset_name: JSON 模式下路径前缀变换（与 _apply_path_transform 一致）
        
        Returns:
            dict: {"frame_idx": progress_value} 或 None（如果文件不存在）
        """
        if not image_paths_list or len(image_paths_list) == 0:
            logging.warning("Empty image paths list, cannot load progress info")
            return None
        
        parent_dir = self._fs_parent_dir_from_media_path(image_paths_list[0])
        
        # Try candidate dirs: current parent, then one level up.
        # The two-level search is needed for JSON mode where paths include a
        # view subfolder (e.g., /video_dir/images/frame.png → parent is /video_dir/images).
        candidate_dirs = [parent_dir, osp.dirname(parent_dir)]
        
        for search_dir in candidate_dirs:
            search_dir = self._apply_path_transform(search_dir, dataset_name)
            # If bad_info_dtw.json exists, this is an expected "bad" episode, return None silently
            bad_info_path = osp.join(search_dir, "bad_info_dtw.json")
            if osp.exists(bad_info_path):
                return None
            
            info_path_dtw  = osp.join(search_dir, "info_dtw.json")
            info_path_info = osp.join(search_dir, "info.json")
            if osp.exists(info_path_dtw):
                info_path = info_path_dtw
            elif osp.exists(info_path_info):
                info_path = info_path_info
            else:
                continue
            
            try:
                with open(info_path, 'r') as f:
                    data = json.load(f)
                if "aligned_progress" not in data:
                    logging.warning(f"'aligned_progress' key not found in {info_path}")
                    continue
                return data["aligned_progress"]
            except Exception as e:
                continue
        
        msg = f"[DEBUG FALLBACK] No info_dtw.json, info.json, or bad_info_dtw.json found in {candidate_dirs}"
        if msg not in CustomDataset._printed_messages:
            print(msg)
            CustomDataset._printed_messages.add(msg)
        return None

    def check_task_video_available(self, image_tokens_path, dataset_name=None):
        """
        Check if task video is available without actually loading it.
        Returns False if task_paths.json is missing/empty or other issues.
        
        Args:
            image_tokens_path: List of image paths
            dataset_name: Optional dataset name for path transformation
            
        Returns:
            bool: True if task video can potentially be loaded, False otherwise
        """
        if not image_tokens_path or len(image_tokens_path) == 0:
            return False
        
        # Derive task_paths.json path from first image path
        parent_dir = self._fs_parent_dir_from_media_path(image_tokens_path[0])
        
        # In JSON mode, image paths include view folder (e.g., /video_dir/images/frame.png)
        # Try to find task_paths.json: first in parent_dir, then in parent's parent
        json_path = osp.join(parent_dir, self.task_paths_filename)
        
        # Apply path transformation before checking existence
        json_path = self._apply_path_transform(json_path, dataset_name)
        
        if not osp.exists(json_path):
            # Try one level up (in case parent_dir is a view folder)
            parent_dir = osp.dirname(parent_dir)
            json_path = osp.join(parent_dir, self.task_paths_filename)
            json_path = self._apply_path_transform(json_path, dataset_name)
        
        # Check if file exists
        if not osp.exists(json_path):
            print(f"[DEBUG FALLBACK] task_paths.json not found at {json_path}")
            logging.warning(f"task_paths.json not found at {json_path}")
            return False
        
        # Check if it has valid candidates
        try:
            with open(json_path, 'r') as f:
                task_data = json.load(f)
            candidates = task_data.get("same", [])
            if not candidates:
                return False
            return True
        except Exception:
            return False

    def load_indicator_info(self, image_paths_list, dataset_name=None):
        """
        Load indicator.json: per-frame strings (e.g. 'positive' / 'negative') under key 'indicator'.

        Same directory search as load_progress_info; bad_info_dtw.json suppresses load (returns None).
        Missing file returns None (no console spam).
        """
        if not image_paths_list or len(image_paths_list) == 0:
            return None

        parent_dir = self._fs_parent_dir_from_media_path(image_paths_list[0])
        candidate_dirs = [parent_dir, osp.dirname(parent_dir)]

        for search_dir in candidate_dirs:
            search_dir = self._apply_path_transform(search_dir, dataset_name)
            bad_info_path = osp.join(search_dir, "bad_info_dtw.json")
            if osp.exists(bad_info_path):
                return None

            indicator_path = osp.join(search_dir, "indicator.json")
            if not osp.exists(indicator_path):
                continue
            try:
                with open(indicator_path, "r") as f:
                    data = json.load(f)
                if "indicator" not in data or not isinstance(data["indicator"], dict):
                    logging.warning(f"'indicator' dict key missing or invalid in {indicator_path}")
                    continue
                return data["indicator"]
            except Exception:
                continue
        return None

    def _preselect_task_video_peer(self, sample_paths_list, dataset_name):
        """
        Pick a random valid peer episode from task_paths.json.

        Candidates are shuffled and checked in order; peers without info_dtw.json
        (or with bad_info_dtw.json) are skipped. Returns the first valid candidate.

        Returns:
            (selected_idx, True)  — a valid peer exists; use this index with load_task_video_frames
            (None, False)         — no valid candidates found; caller sets will_drop_task_video=True
        """
        if not sample_paths_list:
            return None, False

        # Locate task_paths.json (mirrors load_task_video_frames path logic)
        parent_dir = self._fs_parent_dir_from_media_path(sample_paths_list[0])
        json_path  = self._apply_path_transform(osp.join(parent_dir, self.task_paths_filename), dataset_name)
        if not osp.exists(json_path):
            parent_dir = osp.dirname(parent_dir)
            json_path  = self._apply_path_transform(osp.join(parent_dir, self.task_paths_filename), dataset_name)
        if not osp.exists(json_path):
            return None, False

        try:
            with open(json_path, 'r') as f:
                candidates = json.load(f).get("same", [])
        except Exception:
            return None, False

        if not candidates:
            return None, False

        # Shuffle indices for random selection, then scan for first valid peer.
        shuffled_indices = list(range(len(candidates)))
        random.shuffle(shuffled_indices)

        for idx in shuffled_indices:
            selected_path = candidates[idx]
            # selected_path is an episode directory; append dummy filename so
            # _fs_parent_dir_from_media_path correctly extracts episode dir via dirname.
            peer_progress_info = self.load_progress_info([osp.join(selected_path, "_")], dataset_name)
            if peer_progress_info is not None:
                return idx, True

        # All candidates are bad or missing progress info
        return None, False

    def _resolve_task_episode(self, sample_paths_list, dataset_name=None, selected_idx=None, view_name=None):
        """Find peer task episode and get its file list.

        Args:
            sample_paths_list: Agent primary-view image paths (used to locate task_paths.json).
            dataset_name: Optional dataset name for path transformation.
            selected_idx: Optional pre-selected task video index.
            view_name: str or None — camera view to load. If None, inferred from sample_paths_list.

        Returns:
            (files, selected_idx, selected_path, video_info) or (None, selected_idx, None, None) on failure.
        """
        _ltv = "[_resolve_task_episode]"
        _wk = "load_task_video_frames_detail"

        if not sample_paths_list:
            self._data_warn(_wk, f"{_ltv} fail_reason=empty_sample_paths_list")
            return None, None, None, None

        parent_dir = self._fs_parent_dir_from_media_path(sample_paths_list[0])

        # In JSON mode, image paths include view folder (e.g., /video_dir/images/frame.png)
        # Try to find task_paths.json: first in parent_dir, then in parent's parent
        json_path = osp.join(parent_dir, self.task_paths_filename)

        # Apply path transformation before checking existence
        json_path = self._apply_path_transform(json_path, dataset_name)

        if not osp.exists(json_path):
            # Try one level up (in case parent_dir is a view folder)
            parent_dir = osp.dirname(parent_dir)
            json_path = osp.join(parent_dir, self.task_paths_filename)
            json_path = self._apply_path_transform(json_path, dataset_name)

        if not osp.exists(json_path):
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=task_paths_json_not_found media_path={sample_paths_list[0]!r} "
                f"tried_json_path={json_path!r} parent_dir={parent_dir!r}",
            )
            return None, None, None, None

        try:
            with open(json_path, 'r') as f:
                task_data = json.load(f)
        except Exception as e:
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=task_paths_json_read_error path={json_path!r} err={e!r}",
            )
            return None, None, None, None

        candidates = task_data.get("same", [])
        if not candidates:
            keys = list(task_data.keys()) if isinstance(task_data, dict) else type(task_data)
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=empty_same_list json_path={json_path!r} keys={keys}",
            )
            return None, None, None, None

        if selected_idx is None:
            selected_idx = random.randint(0, len(candidates) - 1)

        selected_path = candidates[selected_idx]

        video_info = self._parse_video_path(selected_path, dataset_name=dataset_name)

        # Determine which view to load
        if view_name is not None:
            main_view = view_name
        else:
            # Fallback: infer from sample_paths_list
            main_view = self._infer_view_from_paths(sample_paths_list)

        # Get frame paths using helper
        task_files = self._get_files(video_info, view=main_view)

        if not task_files:
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=no_task_files_for_view selected_idx={selected_idx} "
                f"selected_path={selected_path!r} view={main_view!r} video_info={video_info!r}",
            )
            return None, selected_idx, None, None

        return task_files, selected_idx, selected_path, video_info

    def _align_task_keyframes(
        self,
        files,
        selected_path,
        agent_progresses_list,
        max_frames,
        anchor_frame=None,
        dataset_name=None,
    ):
        """Compute keyframe-to-raw-frame alignment for the task video.

        Runs the anchor-based progress alignment algorithm: places aligned frames at anchor,
        fills backward/forward with random intervals.

        Args:
            files: list[str], all frame paths in the task episode.
            selected_path: str, task episode path (for logging).
            agent_progresses_list: list[float], progress values for each agent keyframe.
            max_frames: int, number of keyframes to sample.
            anchor_frame: Optional int, keyframe anchor position.
            dataset_name: Optional str for path transformation.

        Returns:
            (sampled_indices, task_center_progress, task_sampled_progresses) or (None, None, None) on failure.
        """
        _ltv = "[_align_task_keyframes]"
        _wk = "load_task_video_frames_detail"

        task_total_frames = len(files)

        if agent_progresses_list is None:
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=missing_agent_progresses "
                f"selected_path={selected_path!r}",
            )
            return None, None, None

        # Load task video progress info. MP4 mode: files[0] is already "video://...::idx" from _get_files;
        # do not osp.join(selected_path, ...) — that embeds video:// under episode_dir and breaks dirname.
        ref0 = files[0]
        if not (isinstance(ref0, str) and ref0.startswith("video://")):
            ref0 = osp.join(selected_path, ref0)
        task_progress_info = self.load_progress_info([ref0], dataset_name)
        if task_progress_info is None:
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=task_progress_info_none ref0={ref0!r} "
                f"selected_path={selected_path!r}",
            )
            return None, None, None

        # Build progress array
        # Convert progress dict to sorted list (handles index offset like 1-indexed files)
        sorted_keys = sorted([int(k) for k in task_progress_info.keys()])
        task_progresses_array = np.array([task_progress_info[str(k)] for k in sorted_keys])

        # Verify we have expected number of frames
        if len(task_progresses_array) != task_total_frames:
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=progress_len_mismatch n_progress={len(task_progresses_array)} "
                f"n_task_frames={task_total_frames} selected_path={selected_path!r} ref0={ref0!r}",
            )
            return None, None, None

        # Step 1: Select anchor frame x in range [0, max_frames - self.T]
        if anchor_frame is None:
            if max_frames < self.T:
                self._data_warn(
                    _wk,
                    f"{_ltv} fail_reason=max_frames_lt_T max_frames={max_frames} T={self.T} "
                    f"selected_path={selected_path!r}",
                )
                return None, None, None
            anchor_frame = random.randint(0, max_frames - self.T)

        # Step 2: Core alignment - find task frames for anchor positions (x, x+1, x+2, ...)
        # that best match the first self.T agent progresses
        aligned_task_indices = []
        for i in range(min(self.T, len(agent_progresses_list))):
            agent_prog = agent_progresses_list[i]
            # Find task frame with closest progress
            distances = np.abs(task_progresses_array - agent_prog)
            closest_idx = int(np.argmin(distances))
            aligned_task_indices.append(closest_idx)

        # Verify we have enough aligned indices
        if len(aligned_task_indices) < min(self.T, len(agent_progresses_list)):
            need = min(self.T, len(agent_progresses_list))
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=align_indices_short aligned={len(aligned_task_indices)} "
                f"need={need} agent_prog_len={len(agent_progresses_list)} "
                f"selected_path={selected_path!r}",
            )
            return None, None, None

        # Step 3: Initialize sampled_indices with anchor_frame at the anchor position
        sampled_indices = [None] * max_frames
        # Place the aligned frames at positions starting from anchor_frame
        for i, task_idx in enumerate(aligned_task_indices):
            if anchor_frame + i < max_frames:
                sampled_indices[anchor_frame + i] = task_idx

        # Step 4: Backward sampling - fill frames before anchor with random intervals
        current_idx = aligned_task_indices[0]  # Start from first aligned frame
        for pos in range(anchor_frame - 1, -1, -1):
            # Random interval based on multipliers
            interval = random.randint(int(self.action_frames * self.sampling_interval_min_mult),
                                      int(self.action_frames * self.sampling_interval_max_mult))
            current_idx = max(0, current_idx - interval)
            sampled_indices[pos] = current_idx

        # Step 5: Forward sampling - fill frames after the aligned region with random intervals
        last_aligned_pos = anchor_frame + len(aligned_task_indices) - 1
        if last_aligned_pos < max_frames - 1:
            current_idx = aligned_task_indices[-1]  # Start from last aligned frame
            for pos in range(last_aligned_pos + 1, max_frames):
                # Random interval based on multipliers
                interval = random.randint(int(self.action_frames * self.sampling_interval_min_mult),
                                          int(self.action_frames * self.sampling_interval_max_mult))
                current_idx = min(task_total_frames - 1, current_idx + interval)
                sampled_indices[pos] = current_idx

        # Step 6: Validation - ensure all indices are valid
        if any(idx is None or idx < 0 or idx >= task_total_frames for idx in sampled_indices):
            bad = [
                (i, idx) for i, idx in enumerate(sampled_indices)
                if idx is None or idx < 0 or idx >= task_total_frames
            ]
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=invalid_sampled_indices task_total_frames={task_total_frames} "
                f"max_frames={max_frames} anchor_frame={anchor_frame} "
                f"bad_entries(first_8)={bad[:8]!r} selected_path={selected_path!r}",
            )
            return None, None, None

        # Get progress for sampled frames
        sampled_indices_array = np.array(sampled_indices)
        task_sampled_progresses = task_progresses_array[sampled_indices_array]

        # Calculate center progress
        num_sampled = len(sampled_indices)
        if num_sampled % 2 == 1:
            task_center_progress = float(task_sampled_progresses[num_sampled // 2])
        else:
            task_center_progress = float((task_sampled_progresses[num_sampled // 2 - 1] +
                                          task_sampled_progresses[num_sampled // 2]) / 2)

        return sampled_indices, task_center_progress, task_sampled_progresses

    def _load_task_frames(self, files, video_indices, image_target_size=None, aug_params=None, dataset_name=None):
        """Load, resize, and augment task video frames at given video-frame indices.

        Args:
            files: list[str], all frame paths in the task episode.
            video_indices: list[int], frame indices to load.
            image_target_size: Optional (width, height) for resizing.
            aug_params: Optional dict from _sample_aug_params().
            dataset_name: str, dataset identifier for per-dataset image size lookup.

        Returns:
            (resized_images_list, task_unified_size) or (None, None) on failure.
        """
        _ltv = "[_load_task_frames]"
        _wk = "load_task_video_frames_detail"

        # Load frames at video indices
        selected_files = [files[i] for i in video_indices]

        frame_paths = list(selected_files)

        if not frame_paths:
            nf = len(files) if files is not None else None
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=empty_frame_paths "
                f"video_indices={video_indices!r} n_files={nf}",
            )
            return None, None

        # Determine task_unified_size — multi-view secondary calls pass image_target_size
        # (already computed by primary view call) to ensure all views are resized identically.
        if image_target_size is not None:
            task_unified_size = image_target_size
        else:
            task_unified_size = self._get_target_size(dataset_name)

        # Load all frames at the determined size
        images = self._load_frame(frame_paths, target_size=task_unified_size)

        # Apply consistent augmentation (same params across all views)
        if aug_params is not None and images:
            images = self._apply_aug(images, aug_params)

        if images is None or (isinstance(images, list) and len(images) == 0):
            fp0 = frame_paths[0] if frame_paths else None
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=_load_frame_empty n_paths={len(frame_paths)} first_path={fp0!r} "
                f"task_unified_size={task_unified_size!r}",
            )
            return None, None

        # Convert loaded images to numpy arrays
        resized_images_list = []
        for img in images:
            img_array = np.array(img, dtype=np.uint8)
            resized_images_list.append(img_array)

        if not resized_images_list:
            ni = len(images) if images is not None else -1
            fp0 = frame_paths[0] if frame_paths else None
            self._data_warn(
                _wk,
                f"{_ltv} fail_reason=no_frames_after_decode n_images={ni} n_paths={len(frame_paths)} "
                f"first_path={fp0!r}",
            )
            return None, None

        return resized_images_list, task_unified_size

    def _compute_n_per_seg(self):
        """Compute the number of video frames per keyframe segment.

        Returns:
            n_per_seg: int, total points per segment including endpoints (e.g., 5).
                       frames_per_seg = n_per_seg - 1 (e.g., 4 inter-keyframe frames).
        """
        raw = max(1, round(self.action_frames / self.action_video_freq_ratio))
        return max(4, ((raw + 3) // 4) * 4) + 1

    def _apply_frame_level_shift(self, video_indices_ext, max_frames, anchor_frame):
        """Apply random sub-segment shift to extended video indices and compute frame-level progress.

        Given video_indices from (max_frames + 1) keyframes, randomly shifts the window
        by sub_offset frames within one segment, then crops back to the original T_task length.
        Computes frame-level start/end progress from the shifted anchor position.

        Args:
            video_indices_ext: list[int], video frame indices from extended alignment
                (max_frames + 1 keyframes → T_task + frames_per_seg frames).
            max_frames: int, ORIGINAL keyframe count (before +1 extension).
            anchor_frame: int, keyframe-level anchor position.

        Returns:
            video_indices_shifted: list[int], cropped to T_task length.
            start_progress: float in [0, 1].
            end_progress: float in [0, 1].
        """
        n_per_seg = self._compute_n_per_seg()
        frames_per_seg = n_per_seg - 1
        T_task = 1 + (max_frames - 1) * frames_per_seg       # original task video frame count
        T_agent = 1 + (self.T - 1) * frames_per_seg          # agent video frame count

        # Random sub-segment offset (disabled when task_video_random_offset=False)
        sub_offset = random.randint(0, frames_per_seg - 1) if self.task_video_random_offset else 0

        # Crop shifted window from extended indices
        video_indices_shifted = video_indices_ext[sub_offset : sub_offset + T_task]

        # Frame-level progress: the shift moves the window forward by sub_offset,
        # so the anchor's position in the shifted video moves BACK by sub_offset.
        # anchor_frame >= 1 is guaranteed by the caller, so anchor_in_frames >= 1.
        anchor_in_frames = anchor_frame * frames_per_seg - sub_offset
        assert anchor_in_frames >= 0, (
            f"anchor_in_frames={anchor_in_frames} < 0: anchor_frame={anchor_frame} "
            f"frames_per_seg={frames_per_seg} sub_offset={sub_offset}"
        )
        anchor_in_frames = min(anchor_in_frames, max(T_task - T_agent, 0))
        end_in_frames = min(anchor_in_frames + T_agent - 1, T_task - 1)
        denom = max(T_task - 1, 1)
        start_progress = anchor_in_frames / denom
        end_progress = end_in_frames / denom

        logging.debug(
            "[_apply_frame_level_shift] anchor=%d sub_offset=%d/%d "
            "anchor_in_frames=%d T_task=%d T_agent=%d "
            "start_progress=%.4f end_progress=%.4f",
            anchor_frame, sub_offset, frames_per_seg,
            anchor_in_frames, T_task, T_agent,
            start_progress, end_progress,
        )
        return video_indices_shifted, start_progress, end_progress

    def load_task_video_frames(
        self,
        sample_paths_list,
        agent_views,
        views_images,
        agent_progresses_list=None,
        max_frames=None,
        anchor_frame=None,
        selected_idx=None,
        dataset_name=None,
        aug_params=None,
        video_frame_indices=None,
    ):
        """
        Load task video frames for ALL camera views in one call.

        Orchestrates: _resolve_task_episode (once) → _align_task_keyframes (once)
        → _load_task_frames (per view).

        Args:
            sample_paths_list: Primary-view image paths (used to locate task_paths.json).
            agent_views: list[str], all view names [primary, extra1, ...].
            views_images: list[list[str]], per-view path lists (fallback for file resolution).
            agent_progresses_list: list[float], progress values for each agent keyframe.
            max_frames: int or list[int,int], number of keyframes to sample.
            anchor_frame: Optional int, keyframe anchor position.
            selected_idx: Optional int, pre-selected peer episode index.
            dataset_name: Optional str for path transformation.
            aug_params: Optional dict from _sample_aug_params(). Same augmentation
                applied to every frame in every view for consistency.
            video_frame_indices: Optional list[int], pre-computed video-frame-level indices.
                When provided, skips alignment and downsampling — loads frames directly
                at these indices. Used for frame-level shift.

        Returns:
            (resized_task_views, selected_idx, selected_path, sampled_indices,
             task_center_progress, task_sampled_progresses, task_unified_size,
             video_indices)
            where resized_task_views is list[list[np.ndarray]], one list per view.
            Returns (None, selected_idx, None, ...) on failure.
        """
        _FAIL = (None, None, None, None, None, None, None, None, None, None)

        if max_frames is None:
            max_frames = self.task_max_frames
        if isinstance(max_frames, list):
            max_frames = random.randint(max_frames[0], max_frames[1])

        primary_view = agent_views[0] if agent_views else None

        # Step 1: Resolve task episode (once — determines peer episode for all views)
        files_primary, selected_idx, selected_path, video_info = self._resolve_task_episode(
            sample_paths_list, dataset_name=dataset_name,
            selected_idx=selected_idx, view_name=primary_view,
        )
        if files_primary is None:
            return _FAIL

        task_total_frames = len(files_primary)

        # Step 2: Alignment + downsample + frame-level shift
        sampled_indices = None
        task_center_progress = None
        task_sampled_progresses = None
        start_progress = 0.0
        end_progress = 0.0

        if video_frame_indices is not None:
            # Pre-computed frame-level indices — skip alignment/shift
            video_indices = video_frame_indices
        elif anchor_frame == 0 and max_frames == self.T:
            # Exact alignment mode: task keyframes == agent keyframes, no extension/shift needed
            align_result = self._align_task_keyframes(
                files_primary, selected_path, agent_progresses_list,
                max_frames, anchor_frame=0, dataset_name=dataset_name,
            )
            if align_result[0] is None:
                return _FAIL
            sampled_indices, task_center_progress, task_sampled_progresses = align_result

            video_indices = self._downsample_segments_to_vae_indices(
                sampled_indices, self.action_frames, self.action_video_freq_ratio, task_total_frames,
            )
            start_progress = 0.0
            end_progress = 1.0
        else:
            # Align with extended max_frames (+1 segment buffer for shift)
            extended_max_frames = max_frames + 1
            align_result = self._align_task_keyframes(
                files_primary, selected_path, agent_progresses_list,
                extended_max_frames, anchor_frame=anchor_frame, dataset_name=dataset_name,
            )
            if align_result[0] is None:
                return _FAIL
            sampled_indices, task_center_progress, task_sampled_progresses = align_result

            # Downsample extended keyframes to video indices
            video_indices_ext = self._downsample_segments_to_vae_indices(
                sampled_indices, self.action_frames, self.action_video_freq_ratio, task_total_frames,
            )

            # Apply frame-level shift: crop shifted window + compute progress
            video_indices, start_progress, end_progress = self._apply_frame_level_shift(
                video_indices_ext, max_frames, anchor_frame,
            )

        # Step 3: Load frames for ALL views
        resized_task_views = []
        task_unified_size = None

        for v_idx, v_name in enumerate(agent_views):
            # Get file list for this view
            if v_idx == 0:
                files_v = files_primary
            else:
                files_v = self._get_files(video_info, view=v_name)
                if not files_v:
                    self._data_warn(
                        "load_task_video_frames_detail",
                        f"[load_task_video_frames] fail_reason=no_files_for_extra_view "
                        f"view={v_name!r} selected_path={selected_path!r}",
                    )
                    return _FAIL

            # Load frames — primary computes task_unified_size, extras reuse it
            load_result = self._load_task_frames(
                files_v, video_indices,
                image_target_size=task_unified_size,
                aug_params=aug_params,
                dataset_name=dataset_name,
            )
            if load_result[0] is None:
                self._data_warn(
                    "load_task_video_frames_detail",
                    f"[load_task_video_frames] fail_reason=load_frames_failed "
                    f"view={v_name!r} v_idx={v_idx} selected_path={selected_path!r}",
                )
                return _FAIL
            resized_images, task_unified_size = load_result
            resized_task_views.append(resized_images)

        return (resized_task_views, selected_idx, selected_path, sampled_indices,
                task_center_progress, task_sampled_progresses, task_unified_size,
                video_indices, start_progress, end_progress)

    def __getitem__(self, index: int):
        """
        Get a single training sample. Automatically retries with a different index if the sample is invalid.
        
        Args:
            index: Initial index to try
            
        Returns:
            Valid sample dict (guaranteed not None)
        """
        max_retries = 100  # Prevent infinite loops
        for attempt in range(max_retries):
            try:
                sample = self._get_single_item(index)
                if sample is not None:
                    return sample
                # Sample was None, try a different random index
                new_index = random.randint(0, len(self.data) - 1)
                logging.warning(f"Sample at index {index} returned None (attempt {attempt+1}/{max_retries}), retrying with index {new_index}")
                index = new_index
            except SilentFilterError:
                # Silent retry for expected filtering (e.g., insufficient frames)
                index = random.randint(0, len(self.data) - 1)
            except CamMappingError:
                raise
            except ValueError as e:
                new_index = random.randint(0, len(self.data) - 1)
                logging.warning(
                    f"ValueError in _get_single_item at index {index} (attempt {attempt+1}/{max_retries}): {e}, "
                    f"retrying with index {new_index}"
                )
                index = new_index
            except Exception as e:
                # Unexpected exception: log with full traceback and try a different index
                tb_str = traceback.format_exc()
                new_index = random.randint(0, len(self.data) - 1)
                logging.warning(
                    f"Exception in _get_single_item at index {index}: {type(e).__name__}: {e}\n"
                    f"Attempt {attempt+1}/{max_retries}, retrying with index {new_index}\n"
                    f"Full Traceback:\n{tb_str}"
                )
                index = new_index
        
        # If we exhausted all retries, raise an error
        raise RuntimeError(f"Failed to get valid sample after {max_retries} attempts")

    def _get_single_item(self, index: int):
        """
        Internal method to retrieve a single item. May return None if data is invalid.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Sample dict or None if the sample should be skipped
        """
        # JSON mode only: construct item dynamically from path data.
        scene = self._construct_json_item(index)
        # Lazily derive action component slices from col_map (same for all episodes,
        # cached on self so it is only computed once per DataLoader worker).
        if not hasattr(self, '_action_component_slices') or self._action_component_slices is None:
            _col_map = (scene.get('_ep_raw') or {}).get('col_map')
            if _col_map:
                self._action_component_slices = _build_action_component_slices(_col_map)
                logging.info("[action_component_slices] built: %s", list(self._action_component_slices.keys()))
            else:
                self._action_component_slices = {}
        # Extract selected views for task video alignment
        agent_views = scene.pop('_selected_views', None) or []  # list[str]
        if not agent_views:
            logging.warning(f"_selected_views missing or empty for index {index}, no multi-view alignment will be applied")
        # Extract joint data arrays loaded by _construct_json_item
        joint_all        = scene.pop('joint_all', None)  # np.float32 [T, D_joint] or None
        joint_data_frame = None                          # np.float32 [D_joint], set after sampling
        # Extract progress list and task-video availability pre-loaded by _construct_json_item
        progress_list        = scene.pop('_progress_list', None)     # list[float] or None
        indicator_list       = scene.pop('_indicator_list', None)   # list[str] or None
        task_video_available = scene.pop('_task_video_available', False)
        static_keep_ratio    = scene.pop('static_keep_ratio', 1.0)  # float in (0, 1]
        context = scene.pop('context', None)
        context_mask = scene.pop('context_mask', None)
        # Resolve views and dataset info
        views_images      = scene["views_images"]    # list[list[str]]
        image_tokens_path = views_images[0]          # primary view paths
        dataset_name      = scene.get("dataset", None)
        if dataset_name and dataset_name in self.dataset_fps:
            self.action_frames = self.dataset_fps[dataset_name]
        else:
            print(f"Warning: No dataset fps found for {dataset_name}")

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["text"]
        else:
            prompt = scene["text"]

        # 0. Early decision on task video drop - check all conditions before sampling
        # This allows us to choose the right sampling strategy upfront
        # (agent_progress_info and task_video_available are pre-loaded above for both modes)

        # Check 1: info.json exists (progress_list is None when info.json was missing or mismatched)
        forced_drop_no_progress   = (progress_list is None)
        # Check 2: task_paths.json valid
        forced_drop_no_task_video = not task_video_available
        # Check 3: Pre-select a peer episode and verify it has progress info.
        # Only runs when checks 1-2 all pass (lazy: avoids I/O when already dropping).
        pre_selected_task_idx        = None
        forced_drop_no_peer_progress = False
        if not (forced_drop_no_progress or forced_drop_no_task_video):
            pre_selected_task_idx, peer_has_progress = self._preselect_task_video_peer(
                image_tokens_path, dataset_name
            )
            if not peer_has_progress:
                forced_drop_no_peer_progress = True
        will_drop_task_video = (forced_drop_no_progress or
                                forced_drop_no_task_video or forced_drop_no_peer_progress)

        # Probabilistic task_video drop (per-sample, moved from model-side batch-level drop).
        # This ensures frame sampling strategy is consistent with the drop decision.
        if not will_drop_task_video and random.random() < self.task_video_drop_prob:
            will_drop_task_video = True

        # 1. Generate variable action_frames per agent frame based on drop decision
        if will_drop_task_video:
            # No task video alignment needed, use fixed interval
            action_frames_per_step = [self.action_frames] * self.T
        else:
            # Task video alignment needed, use variable intervals
            action_frames_per_step = [random.randint(int(self.action_frames * self.sampling_interval_min_mult),
                                                     int(self.action_frames * self.sampling_interval_max_mult))
                                      for _ in range(self.T)]

        frames_num = sum(action_frames_per_step)

        # Ensure frames_num doesn't exceed available data
        if frames_num > len(image_tokens_path):
            # Scale down proportionally
            scale_factor = len(image_tokens_path) / frames_num
            action_frames_per_step = [max(1, int(af * scale_factor)) for af in action_frames_per_step]
            frames_num = sum(action_frames_per_step)

        # 2. Sample Robot Frames & Get Progress
        start_idx = 0
        agent_frame_indices = None
        resized_views = []      # list[list[np.ndarray]]: [v_idx][frame_idx]

        if self.actions and scene.get("action") is not None:
            action = scene["action"]
            result = self.random_frames_to_tensor(
                views_images, frames_num,
                action_prompt=action,
                action_frames_list=action_frames_per_step,
                dataset_name=dataset_name,
            )
            (
                action_tokens,
                start_idx,
                agent_frame_indices,
                resized_views,
                _,
            ) = result
        else:
            result = self.random_frames_to_tensor(
                views_images, frames_num,
                action_frames_list=action_frames_per_step,
                dataset_name=dataset_name,
            )
            (
                _,
                start_idx,
                agent_frame_indices,
                resized_views,
                _,
            ) = result

        # Validation: Check consistency of sampled data
        assert len(agent_frame_indices) == len(action_frames_per_step), \
            f"Agent frame indices length ({len(agent_frame_indices)}) must equal action_frames_per_step length ({len(action_frames_per_step)})"

        # Get agent images' progress using the actual sampled frame indices.
        if progress_list is None:
            agent_progresses = [0.5] * len(agent_frame_indices)
            if not will_drop_task_video:
                logging.warning(
                    "[_get_single_item] progress_list is None for index=%d — "
                    "falling back to uniform progress=0.5 for %d agent frames; "
                    "task video alignment will be incorrect",
                    index, len(agent_frame_indices),
                )
        else:
            agent_progresses = [progress_list[idx] for idx in agent_frame_indices]

        # 4. Load Task Video (Aligned) — unified multi-view loop
        # drop_task_video was already decided in step 0
        drop_task_video = will_drop_task_video

        task_idx = None
        task_image_path = None
        sampled_indices = None
        task_center_progress = None
        task_sampled_progresses = None
        anchor_frame = None
        video_indices = None
        task_unified_size = None
        frame_start_progress = None
        frame_end_progress = None
        # resized_task_views[v_idx] = list of np.ndarray frames for view v_idx
        resized_task_views = []

        if not drop_task_video:
            n_views = len(views_images)
            max_frames_range = self.task_max_frames[n_views]
            current_max_frames = random.randint(max_frames_range[0], max_frames_range[1])

            # anchor_frame is shared across all views for temporal alignment.
            # When max_frames == T, all keyframes align 1:1 with agent frames → anchor must be 0.
            # When max_frames > T, start from 1 (not 0) so frame-level shift has buffer.
            if current_max_frames == self.T:
                anchor_frame = 0
            elif current_max_frames >= self.T + 1:
                anchor_frame = random.randint(1, current_max_frames - self.T)
            else:
                anchor_frame = 1

            # Sample task-video augmentation params once — shared across all views so every
            # camera angle in the task video receives identical transforms.  These params are
            # independent of the agent augmentation sampled in random_frames_to_tensor().
            task_aug_params = self._sample_aug_params() if self.use_augmentation else None

            # Load task video for ALL views in one call
            tv_result = self.load_task_video_frames(
                image_tokens_path,
                agent_views=agent_views,
                views_images=views_images,
                agent_progresses_list=agent_progresses,
                max_frames=current_max_frames,
                anchor_frame=anchor_frame,
                selected_idx=pre_selected_task_idx,
                dataset_name=dataset_name,
                aug_params=task_aug_params,
            )
            (resized_task_views_result, task_idx, task_image_path, sampled_indices,
             task_center_progress, task_sampled_progresses, task_unified_size,
             video_indices, frame_start_progress, frame_end_progress) = tv_result

            if resized_task_views_result is None:
                if not will_drop_task_video:
                    self._data_warn(
                        "load_task_video_frames_detail",
                        "Task video loading failed but was needed for alignment, skipping sample",
                    )
                    raise SilentFilterError("Task video loading failed but was needed for alignment, skipping sample")
                drop_task_video = True
            else:
                resized_task_views = resized_task_views_result

            # Compute aligned region indices in the task_video tensor (visualization only)
            # Directly map frame-level progress to task video frame indices
            T_task = len(video_indices) if video_indices is not None else 0
            if T_task > 0 and frame_start_progress is not None:
                task_video_start = int(round(frame_start_progress * max(T_task - 1, 1)))
                task_video_end = int(round(frame_end_progress * max(T_task - 1, 1))) + 1
            else:
                task_video_start = 0
                task_video_end = T_task

            # Final cleanup if dropped during loading
            if drop_task_video:
                task_idx = None
                task_image_path = None
                sampled_indices = None
                task_center_progress = None
                task_sampled_progresses = None
                video_indices = None
                frame_start_progress = None
                frame_end_progress = None
                resized_task_views = []

        # ===== Build SelfGroundedPredictor output =====

        # Agent video tensor
        agent_video = self._frames_to_video_tensor(resized_views)  # [C, T_video, H, W]

        # Task video tensor
        task_video = None
        task_video_dropped = False
        if not drop_task_video and resized_task_views:
            task_video = self._frames_to_video_tensor(resized_task_views)  # [C, T_task, H, W]

        if task_video is None and not drop_task_video:
            # Data-availability issue (not probabilistic drop) — retry with different sample
            raise SilentFilterError(
                f"task_video unavailable for index {index} "
                f"(drop={drop_task_video}, n_task_views={len(resized_task_views)})"
            )

        if task_video is None and drop_task_video:
            # Probabilistic drop: construct zero tensor with task video shape
            C, _, H_total, W = agent_video.shape
            n_views = len(agent_views)
            max_frames = self.task_max_frames[n_views][1]
            S = max_frames - 1
            n_per_seg = self._compute_n_per_seg()
            T_task = 1 + S * (n_per_seg - 1)
            task_video = torch.zeros(C, T_task, H_total, W)
            task_video_dropped = True

        # Action + progress + mask
        action_final = None
        action_is_pad = None
        if self.actions and scene.get("action") is not None and agent_frame_indices is not None:
            # Stage 1: resample from random span to fixed-fps action indices (dataset_fps)
            target_action_len = (self.T - 1) * self.action_frames
            assert target_action_len > 0, (
                f"target_action_len must be > 0, got T={self.T}, action_frames={self.action_frames}"
            )
            raw_action_len = len(scene["action"])
            action_sample_indices = (
                np.round(np.linspace(agent_frame_indices[0], agent_frame_indices[-1], target_action_len))
                .astype(int)
                .clip(0, raw_action_len - 1)
                .tolist()
            )
            action_chunk = scene["action"][action_sample_indices]  # unnormalized raw

            # Normalize (and optionally convert to relative) after chunk selection.
            _ep_raw = scene.get('_ep_raw')
            if self.use_relative_action and _ep_raw is not None \
                    and _ep_raw.get('follower_actions') is not None:
                follower_ref = _ep_raw['follower_actions'][agent_frame_indices[0]]
                rel_chunk, _ = self._compute_relative_actions(
                    action_chunk, _ep_raw['action_keys'], _ep_raw['col_map'],
                    follower_ref,
                )
                action_raw = self._normalize_relative_actions(
                    rel_chunk, _ep_raw['action_keys'], _ep_raw['nmd'],
                    ds_factor=1,
                )
            else:
                # Absolute normalization
                a_nm, a_nd = _ep_raw['a_nm'], _ep_raw['a_nd']
                action_raw = np.clip(
                    2.0 * (action_chunk.astype(np.float32) - a_nm) / a_nd - 1.0,
                    -1.0, 1.0,
                )

            # Progress: frame-level task video position from load_task_video_frames.
            # At eval, model-predicted progress is used to advance the task video window.
            if not drop_task_video and frame_start_progress is not None:
                action_progress = np.linspace(
                    frame_start_progress, frame_end_progress, target_action_len
                ).tolist()
            else:
                # task_video dropped — progress is meaningless, use 0.5 as neutral placeholder
                action_progress = [0.5] * target_action_len

            action_final, action_is_pad = self._build_action_tensor(
                action_raw, action_progress, None  # None: no padding, exact target_action_len
            )

        # Proprio: only current frame (first keyframe)
        proprio = None
        if joint_all is not None and agent_frame_indices is not None:
            first_idx = agent_frame_indices[0]
            if first_idx < len(joint_all):
                proprio = torch.from_numpy(joint_all[first_idx]).float().unsqueeze(0)  # [1, joint_dim]

        if proprio is None and self.joints:
            logging.warning(
                f"[_get_single_item] joint_all unavailable for index {index}, "
                f"using zero fallback for proprio"
            )
            proprio = torch.zeros(1, 14, dtype=torch.float32)  # [1, joint_dim]

        # Per-sample text drop (only when task_video is present, same logic as model-side)
        if not task_video_dropped and random.random() < self.task_text_drop_prob:
            context = torch.zeros_like(context)
            context_mask = torch.zeros_like(context_mask)

        # Padding masks
        T_video = agent_video.shape[1]
        image_is_pad = torch.zeros(T_video, dtype=torch.bool)

        sample_dict = {
            "video": agent_video,                    # [C, T_video, H, W], [-1, 1]
            "task_video": task_video,                 # [C, T_task, H, W] (zeros when dropped)
            "task_video_dropped": task_video_dropped,  # bool
            "action": action_final,                   # [L, action_dim+2] or None
            "proprio": proprio,                       # [joint_dim] or None
            "prompt": prompt,                         # str
            "context": context,                       # [L, D]
            "context_mask": context_mask,             # [L]
            "image_is_pad": image_is_pad,             # [T_video]
            "action_is_pad": action_is_pad,           # [L] or None
            "agent_episode_dir": os.path.dirname(image_tokens_path[0]) if image_tokens_path else None,
            "task_episode_dir": task_image_path if task_image_path is not None else None,
            "progress_gt": torch.tensor(
                [frame_start_progress or 0.0, frame_end_progress or 0.0],
                dtype=torch.float32,
            ),  # [2]: [start, end] in [0, 1]
        }
        if not drop_task_video and task_video is not None:
            sample_dict["task_video_start"] = task_video_start   # int, visualization only
            sample_dict["task_video_end"]   = task_video_end     # int (exclusive), visualization only
        return sample_dict

def pad_context_sequence(context, context_mask, target_len):
    """Pad or truncate a single context tensor pair to target_len.

    Args:
        context:      [L, D] float tensor
        context_mask: [L]    bool  tensor
        target_len:   int, target sequence length

    Returns:
        context:      [target_len, D]
        context_mask: [target_len]
    """
    L = context.shape[0]
    if L > target_len:
        return context[:target_len], context_mask[:target_len]
    if L < target_len:
        pad = target_len - L
        context = torch.cat([context, torch.zeros(pad, context.shape[1])], dim=0)
        context_mask = torch.cat([context_mask, torch.zeros(pad, dtype=torch.bool)], dim=0)
    return context, context_mask


class CustomCollator:
    """Collator that dynamically pads context to the longest sequence in the batch.

    Asserts shape consistency for video and task_video to catch logic errors early.

    Args:
        context_len: Hard truncation cap applied before computing max_L.
                     Protects against unexpectedly long T5 outputs.
    """

    def __init__(self, context_len: int):
        self.context_len = context_len

    def __call__(self, batch):
        from torch.utils.data._utils.collate import default_collate

        batch = [s for s in batch if s is not None]
        if not batch:
            return None

        # Extract non-tensor / variable-length fields before default_collate
        prompts            = [s.pop("prompt") for s in batch]
        task_video_starts  = [s.pop("task_video_start",  0)     for s in batch]
        task_video_ends    = [s.pop("task_video_end",    None)  for s in batch]
        task_video_dropped = [s.pop("task_video_dropped", False) for s in batch]
        agent_episode_dirs = [s.pop("agent_episode_dir", None)  for s in batch]
        task_episode_dirs  = [s.pop("task_episode_dir",  None)  for s in batch]

        # Pop variable-length context from each sample
        raw_contexts = [s.pop("context") for s in batch]       # list of [L_i, D]
        raw_masks    = [s.pop("context_mask") for s in batch]  # list of [L_i]

        # Hard truncation cap, then find batch-max length for dynamic padding
        if self.context_len is not None:
            raw_contexts = [c[:self.context_len] for c in raw_contexts]
            raw_masks    = [m[:self.context_len] for m in raw_masks]

        max_L = max(c.shape[0] for c in raw_contexts)

        # Pad each sample to max_L using the shared helper
        contexts, context_masks = [], []
        for ctx, msk in zip(raw_contexts, raw_masks):
            ctx, msk = pad_context_sequence(ctx, msk, max_L)
            contexts.append(ctx)
            context_masks.append(msk)

        # Assert shape consistency for video tensors
        video_shapes = set(tuple(s["video"].shape) for s in batch)
        assert len(video_shapes) == 1, f"Agent video shapes inconsistent: {video_shapes}"
        tv_shapes = set(tuple(s["task_video"].shape) for s in batch)
        assert len(tv_shapes) == 1, f"Task video shapes inconsistent: {tv_shapes}"

        result = default_collate(batch)
        result["context"]            = torch.stack(contexts, dim=0)       # [B, max_L, D]
        result["context_mask"]       = torch.stack(context_masks, dim=0)  # [B, max_L]
        result["prompt"]             = prompts
        result["task_video_start"]   = task_video_starts
        result["task_video_end"]     = task_video_ends
        result["task_video_dropped"] = task_video_dropped
        result["agent_episode_dir"]  = agent_episode_dirs
        result["task_episode_dir"]   = task_episode_dirs
        return result


class _HydraArgs:
    """Minimal args object that bridges Hydra config kwargs to attribute access."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_custom_dataset(**kwargs) -> CustomDataset:
    """Factory function for Hydra instantiate().

    Converts keyword arguments from the YAML config into an args object
    that CustomDataset.__init__ expects.

    Usage in YAML:
        _target_: self_grounded_prediction.datasets.custom.mydatasets.build_custom_dataset
        data_path: ...
        frames: 2
        ...
    """
    args = _HydraArgs(**kwargs)
    return CustomDataset(args)