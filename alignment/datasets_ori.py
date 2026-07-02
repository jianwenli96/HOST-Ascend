# coding=utf-8
"""Datasets."""

import os
import json
import glob
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from config import CONFIG

from data_utils import get_image_info, get_video_info

from torch.nn.utils.rnn import pad_sequence

# Decord for efficient video reading
import decord
from decord import VideoReader, cpu

# Qwen VL utils for smart resize
from qwen_vl_utils.vision_process import smart_resize

# From 3Hz to 5Hz control frequency
# 低频控制数据集 (通常 <= 5Hz)
datasets_with_lower_frequency = [
    'rt1', 
    'berkeley_autolab_ur5', 
    'bridgev2'  # Bridge V2 通常是 5Hz
    'SSv2'
]

# From 15Hz to 30 Hz control frequency
# 高频控制数据集 (通常 >= 15Hz)
datasets_with_higher_frequency = [
    'utaustin_mutex', 
    'droid', 
    'droid_fast', 
    'viola', 
    'toto',       # TOTO 原生通常是 30Hz
    'taco_play',  # Taco Play 通常是较高频率 (Rosbag),
    'Egodex',
    'robocoin',
]

# Per-dataset allowed view (subdirectory) names for sampling. Only these views are used when present.
# Add a key per dataset (matched by substring in video path, e.g. 'robocoin' in path) and list allowed dir names.
# Datasets not listed here keep the default behavior: keyword filter ('image', 'gripper', 'rgb').
DATASET_VIEW_CONFIG = {
    'bridge': ['images'],
    'robocoin': [
        'observation.images.cam_chest_rgb',
        'observation.images.cam_front_rgb',
        'observation.images.cam_head_rgb',
        'observation.images.cam_high_left_rgb',
        'observation.images.cam_high_rgb',
        'observation.images.cam_high_right_rgb',
        'observation.images.camera_head_rgb',
        'observation.images.cam_third_view',
    ],
}

class TCCCollator:
    def __init__(self, processor=None, mode='train'):
        self.processor = processor
        self.mode = mode
        if self.processor and hasattr(self.processor, 'tokenizer'):
            self.pad_token_id = self.processor.tokenizer.pad_token_id
        else:
            self.pad_token_id = 0
            
        print('self.pad_token_id:', self.pad_token_id)
            
        # Standard Visual Augmentation
        # Configurable Parameters
        self.do_color_jitter = CONFIG.AUGMENTATION.get('BRIGHTNESS', True) or \
                               CONFIG.AUGMENTATION.get('CONTRAST', True) or \
                               CONFIG.AUGMENTATION.get('SATURATION', False) or \
                               CONFIG.AUGMENTATION.get('HUE', False)
        
        self.color_jitter = transforms.ColorJitter(
            brightness=CONFIG.AUGMENTATION.get('BRIGHTNESS_MAX_DELTA', 32.0 / 255) if CONFIG.AUGMENTATION.get('BRIGHTNESS', True) else 0, 
            contrast=(CONFIG.AUGMENTATION.get('CONTRAST_LOWER', 0.5), CONFIG.AUGMENTATION.get('CONTRAST_UPPER', 1.5)) if CONFIG.AUGMENTATION.get('CONTRAST', True) else 0,
            saturation=(CONFIG.AUGMENTATION.get('SATURATION_LOWER', 0.5), CONFIG.AUGMENTATION.get('SATURATION_UPPER', 1.5)) if CONFIG.AUGMENTATION.get('SATURATION', True) else 0, 
            hue=CONFIG.AUGMENTATION.get('HUE_MAX_DELTA', 0.2) if CONFIG.AUGMENTATION.get('HUE', True) else 0
        )
        self.do_random_flip = CONFIG.AUGMENTATION.RANDOM_FLIP
        
        # Random Crop Parameters
        self.do_random_crop = CONFIG.AUGMENTATION.RANDOM_CROP
        self.crop_min_scale = CONFIG.AUGMENTATION.get('CROP_MIN_SCALE', 0.8)
        self.crop_max_scale = CONFIG.AUGMENTATION.get('CROP_MAX_SCALE', 1.0)
        self.crop_ratio = (3.0/4.0, 4.0/3.0) 

        # Disable augmentation in non-train mode
        if self.mode != 'train':
            self.do_color_jitter = False
            self.do_random_flip = False
            self.do_random_crop = False

    def augment_sequence(self, images, jitter_transform, random_flip, random_crop):
        """
        Applies consistent geometric and independent photometric augmentation to a sequence of images.
        """
        augmented = []
        do_flip = False
        if random_flip and random.random() < 0.5:
            do_flip = True
            
        crop_params = None
        if random_crop and len(images) > 0:
            img0 = images[0]
            crop_params = transforms.RandomResizedCrop.get_params(
                img0, scale=(self.crop_min_scale, self.crop_max_scale), ratio=self.crop_ratio
            )
            
        jitter_params_list = []
        
        for img in images:
            # Geometric: Consistent
            if do_flip:
                img = TF.hflip(img)
            
            if crop_params is not None:
                i, j, h, w = crop_params
                img = TF.resized_crop(img, i, j, h, w, size=img.size)
            
            # Photometric: Independent (Manual to capture params)
            img_jitter_params = None
            if jitter_transform is not None:
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    transforms.ColorJitter.get_params(
                        jitter_transform.brightness, 
                        jitter_transform.contrast, 
                        jitter_transform.saturation, 
                        jitter_transform.hue
                    )
                
                # Apply params
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = TF.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = TF.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = TF.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = TF.adjust_hue(img, hue_factor)
                
                # Capture params
                img_jitter_params = {
                    'fn_idx': fn_idx.tolist() if hasattr(fn_idx, 'tolist') else fn_idx,
                    'b': float(brightness_factor) if brightness_factor is not None else None,
                    'c': float(contrast_factor) if contrast_factor is not None else None,
                    's': float(saturation_factor) if saturation_factor is not None else None,
                    'h': float(hue_factor) if hue_factor is not None else None
                }
            
            jitter_params_list.append(img_jitter_params)
            augmented.append(img)
            
        params = {
            'flip': do_flip,
            'crop': crop_params,
            'jitter': jitter_params_list
        }
            
        return augmented, params

    def load_imgs(self, paths):
        """
        Load images from various sources (file paths, arrays, video frames).
        Uses batch reading for video frames to maximize decord performance.
        """
        # Phase 1: Scan and group video frames
        video_frame_groups = {}  # {(video_path, width, height): [(idx_in_paths, frame_idx), ...]}
        
        for idx, p in enumerate(paths):
            if isinstance(p, str) and p.startswith("video://"):
                # Parse video protocol: "video://<video_path>::<frame_idx>::<width>x<height>"
                try:
                    path_and_info = p[8:]  # Remove "video://" prefix
                    parts = path_and_info.rsplit('::', 2)  # Split by last 2 "::"
                    
                    if len(parts) == 3:
                        # New protocol with size
                        video_path, frame_idx_str, size_str = parts
                        frame_idx = int(frame_idx_str)
                        width, height = map(int, size_str.split('x'))
                    else:
                        # Fallback for old protocol without size
                        video_path, frame_idx_str = path_and_info.rsplit('::', 1)
                        frame_idx = int(frame_idx_str)
                        width, height = None, None  # Will use original size
                    
                    key = (video_path, width, height)
                    if key not in video_frame_groups:
                        video_frame_groups[key] = []
                    video_frame_groups[key].append((idx, frame_idx))
                except Exception as e:
                    print(f"Error parsing video path {p}: {e}")
        
        # Phase 2: Batch read video frames with target size
        video_frames_map = {}  # {(video_path, frame_idx): PIL.Image}
        
        for (video_path, target_width, target_height), frame_list in video_frame_groups.items():
            try:
                # Create VideoReader with target dimensions (always use CPU)
                if target_width is not None and target_height is not None:
                    # Decord will resize during decoding
                    try:
                        vr = VideoReader(
                            video_path, 
                            ctx=cpu(0),
                            width=target_width,   # Target width
                            height=target_height  # Target height
                        )
                    except TypeError:
                        # Fallback if VideoReader doesn't support width/height (old decord version)
                        print(f"Warning: VideoReader doesn't support resize, using original size")
                        vr = VideoReader(video_path, ctx=cpu(0))
                else:
                    # Fallback: use original size (for old protocol)
                    vr = VideoReader(video_path, ctx=cpu(0))
                
                # Extract frame indices
                indices = [frame_idx for _, frame_idx in frame_list]
                
                # Batch read using decord (already at target size if width/height specified!)
                frames_batch = vr.get_batch(indices)  # Returns (N, H, W, C) NDArray
                
                # Convert NDArray to numpy array first (required for indexing)
                frames_np = frames_batch.asnumpy()  # Now it's a numpy array: (N, H, W, C)
                
                # Convert to PIL Images and store in map
                for i, (path_idx, frame_idx) in enumerate(frame_list):
                    frame = frames_np[i]  # Index into numpy array, already at target size!
                    img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
                    video_frames_map[(video_path, frame_idx)] = img
                
                # Release VideoReader immediately after use
                del vr
                    
            except Exception as e:
                print(f"Error batch reading video {video_path}: {e}")
                # Fallback: create black images for failed frames
                for path_idx, frame_idx in frame_list:
                    video_frames_map[(video_path, frame_idx)] = Image.new(
                        'RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0)
                    )
        
        # Phase 3: Assemble results in original order
        imgs = []
        for idx, p in enumerate(paths):
            if isinstance(p, str) and p == "DUSTBIN":
                img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
            
            elif isinstance(p, str) and p.startswith("video://"):
                # Retrieve from pre-loaded batch
                try:
                    path_and_info = p[8:]
                    parts = path_and_info.rsplit('::', 2)
                    
                    if len(parts) == 3:
                        # New protocol with size: ignore size in retrieval
                        video_path, frame_idx_str, _ = parts  # size_str not needed here
                        frame_idx = int(frame_idx_str)
                    else:
                        # Old protocol without size
                        video_path, frame_idx_str = path_and_info.rsplit('::', 1)
                        frame_idx = int(frame_idx_str)
                    
                    img = video_frames_map.get((video_path, frame_idx))
                    
                    if img is None:
                        # Fallback if not found
                        img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
                except Exception as e:
                    print(f"Error retrieving video frame {p}: {e}")
                    img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
            
            elif isinstance(p, Image.Image):
                img = p.convert('RGB')
            
            elif isinstance(p, np.ndarray):
                img = Image.fromarray(p).convert('RGB')
            
            elif torch.is_tensor(p):
                # Handle torch tensor (C, H, W) or (H, W, C)
                if p.dim() == 3:
                    if p.shape[0] == 3:  # C, H, W
                        p = p.permute(1, 2, 0)
                    img = Image.fromarray(p.cpu().numpy().astype(np.uint8)).convert('RGB')
                else:
                    img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
            
            else:
                # Regular file path
                img = Image.open(p).convert('RGB')
            
            imgs.append(img)
        
        return imgs

    def __call__(self, batch):
        if self.mode == 'train':
            # Sort batch by frame count (descending) to maximize high-multiplier opportunities
            # Each 'frame_paths' list length is (sampled_frames * context_size)
            batch = sorted(batch, key=lambda d: len(d.get('frame_paths', [])), reverse=True)

        # Standard Collation for ResNet path
        collated_batch = {}
        
        def simple_collate(data_list):
            if not data_list: return {}
            elem = data_list[0]
            res = {}
            for key in elem:
                if key.startswith('_'): continue # Skip private keys
                # Keys to keep as lists
                list_keys = ['frame_paths', 'ref_frame_paths', 'name', 'ref_name', 
                             'align_frame_paths', 'main_align_frame_paths',
                             'frame_paths_str', 'ref_frame_paths_str', 
                             'align_frame_paths_str', 'main_align_frame_paths_str']
                if key in list_keys:
                    res[key] = [d[key] for d in data_list]
                elif isinstance(elem[key], torch.Tensor):
                    try:
                        res[key] = torch.stack([d[key] for d in data_list])
                    except RuntimeError:
                        res[key] = [d[key] for d in data_list]
                else:
                    res[key] = [d[key] for d in data_list]
            return res

        collated_batch = simple_collate(batch)

        # Qwen Processing
        if self.processor is not None:
            # --- Implementation of Chunked Batching ---
            batch_size = len(batch)
            c_size = CONFIG.DATA.NUM_STEPS
            chunk_len = getattr(CONFIG.TRAIN, 'NUM_FRAMES', 24)
            
            # Get multiplier probabilities based on mode
            if self.mode == 'train':
                multiplier_probs = getattr(CONFIG.TRAIN, 'CHUNK_PROBS', [0.5, 0.25, 0.25])
                multipliers = [1, 2, 4]
                
                # Filter valid multipliers based on batch size AND available frames in the items
                actually_valid_mults = []
                for m in multipliers:
                    if batch_size % m != 0:
                        continue
                    # Check if the first num_unique unique videos have enough frames to support M
                    num_u = batch_size // m
                    can_do_m = True
                    for i in range(num_u):
                        # len(batch[i]['frame_paths']) is (num_sampled * c_size)
                        num_sampled_frames = len(batch[i]['frame_paths']) // c_size
                        if num_sampled_frames < m * chunk_len:
                            can_do_m = False
                            break
                    if can_do_m:
                        actually_valid_mults.append(m)
                
                if not actually_valid_mults:
                    actually_valid_mults = [1]
                
                valid_probs = [multiplier_probs[multipliers.index(m)] for m in actually_valid_mults]
                p_sum = sum(valid_probs)
                
                if p_sum == 0:
                    # Fallback to uniform distribution if all probabilities are zero
                    valid_probs = [1.0 / len(actually_valid_mults)] * len(actually_valid_mults)
                else:
                    valid_probs = [p / p_sum for p in valid_probs]
                
                M = random.choices(actually_valid_mults, weights=valid_probs)[0]
            else:  # eval mode
                # In eval mode, directly use the multiplier specified by CHUNK_PROBS
                # No validation needed - we trust the config (batch_size=1, dataset samples 96 frames)
                multiplier_probs = getattr(CONFIG.EVAL, 'CHUNK_PROBS', [0.0, 0.0, 1.0])
                multipliers = [1, 2, 4]
                
                # Find the multiplier with probability 1.0 (or highest)
                M = 1
                max_prob = 0
                for i, prob in enumerate(multiplier_probs):
                    if prob > max_prob:
                        max_prob = prob
                        M = multipliers[i]
            
            # if M > 1:
            #     print(f"[TCCCollator Multiplier] Using M={M} for batch of size {batch_size} (Selected from {actually_valid_mults})")

            # In eval mode with batch_size=1 and M=4, ensure we process at least 1 sample
            if self.mode == 'eval':
                num_unique = 1
            else:
                num_unique = batch_size // M
            new_batch = []
            
            # Use chunks of size chunk_len (default 24)
            for i in range(num_unique):
                orig_data = batch[i]
                main_paths_all = orig_data['frame_paths']
                ref_paths_all = orig_data['ref_frame_paths']
                
                # S is the number of frames actually sampled by the dataset (e.g. 96, 48, 24)
                S = len(main_paths_all) // c_size
                is_train = (self.mode == 'train')
                
                # Always want to output exactly M chunks of chunk_len
                # If available S > M*chunk_len, we downsample uniformly.
                target_total = M * chunk_len
                # Indices mapping from [0, S-1] to [0, target_total-1]
                indices = np.linspace(0, S - 1, target_total, dtype=int)
                
                # --- Pre-compute Cut Params for Consistency across Chunks ---
                # This ensures that all chunks from the same video see the SAME cut in the reference.
                consistent_cut_params = None
                if is_train:
                    align_paths_for_cut = orig_data.get('align_frame_paths', [])
                    full_len_cut = len(align_paths_for_cut)
                    if full_len_cut > 0:
                         cut_pct = random.uniform(CONFIG.TRAIN.CUT_RANGE[0], CONFIG.TRAIN.CUT_RANGE[1])
                         cut_len = int(full_len_cut * cut_pct)
                         if cut_len > 0:
                             max_start = full_len_cut - cut_len
                             start_idx = random.randint(0, max_start)
                             end_idx = start_idx + cut_len
                             consistent_cut_params = (start_idx, end_idx)
                
                for m in range(M):
                    chunk_data = orig_data.copy()
                    if consistent_cut_params:
                        chunk_data['_precomputed_cut'] = consistent_cut_params

                    # Pick indices for this specific chunk
                    chunk_indices = indices[m * chunk_len : (m + 1) * chunk_len]
                    
                    # 1. Update paths
                    new_frame_paths = []
                    new_ref_frame_paths = []
                    for idx in chunk_indices:
                        new_frame_paths.extend(main_paths_all[idx * c_size : (idx + 1) * c_size])
                        new_ref_frame_paths.extend(ref_paths_all[idx * c_size : (idx + 1) * c_size])
                    chunk_data['frame_paths'] = new_frame_paths
                    chunk_data['ref_frame_paths'] = new_ref_frame_paths
                    
                    # Also update logical paths if they exist
                    if 'frame_paths_str' in orig_data and orig_data['frame_paths_str'] is not None:
                        main_logical_all = orig_data['frame_paths_str']
                        new_logical_frame_paths = []
                        for idx in chunk_indices:
                            new_logical_frame_paths.extend(main_logical_all[idx * c_size : (idx + 1) * c_size])
                        chunk_data['frame_paths_str'] = new_logical_frame_paths
                    
                    if 'ref_frame_paths_str' in orig_data and orig_data['ref_frame_paths_str'] is not None:
                        ref_logical_all = orig_data['ref_frame_paths_str']
                        new_logical_ref_frame_paths = []
                        for idx in chunk_indices:
                            new_logical_ref_frame_paths.extend(ref_logical_all[idx * c_size : (idx + 1) * c_size])
                        chunk_data['ref_frame_paths_str'] = new_logical_ref_frame_paths
                    
                    # 2. Update chosen steps (tensors)
                    if 'chosen_steps' in orig_data:
                        chunk_data['chosen_steps'] = orig_data['chosen_steps'][chunk_indices]
                    if 'ref_chosen_steps' in orig_data:
                        chunk_data['ref_chosen_steps'] = orig_data['ref_chosen_steps'][chunk_indices]

                    # Track grouping
                    chunk_data['group_id'] = i
                    chunk_data['chunk_id'] = m
                    chunk_data['multiplier'] = M
                    new_batch.append(chunk_data)
            
            # If we had extra items in the batch that weren't used (e.g. batch_size=4, M=4, only batch[0] used)
            # they are naturally dropped by the num_unique loop.
            
            batch = new_batch
            # ------------------------------------------

            # --- Re-run Standard Collation to update top-level keys (like chosen_steps) ---
            collated_batch = simple_collate(batch)

            context_size = CONFIG.DATA.NUM_STEPS
            
            batch_input_ids = []
            batch_pixel_values = []
            batch_image_grid_thw = []
            batch_pixel_values_videos = []
            batch_video_grid_thw = []
            
            batch_num_mains = []
            batch_num_refs = []
            
            batch_group_ids = []
            batch_chunk_ids = []
            
            # Row-level metadata to keep it consistent with the final Qwen rows (2N entries)
            batch_row_frame_paths = []
            batch_row_ref_frame_paths = []
            batch_row_names = []
            batch_row_ref_names = []
            batch_row_dataset_names = []
            batch_row_aug_params = []
            batch_row_ref_aug_params = []
            
            # Metadata Collections (Flattened)
            main_seq_lens_list = []
            ref_seq_lens_list = []

            # --- UPDATED: Collect Masks ---
            batch_is_cut_masks = []
            batch_has_dustbin = []
            
            def process_packed_sequence(video_imgs, chunk_imgs, is_ref_row, seq_len_val, group_id, chunk_id, 
                                      frame_paths, ref_frame_paths, name, ref_name, dataset_name,
                                      aug_params, ref_aug_params,
                                      is_cut_mask=None, has_dustbin=False):
                # --- Build Content ---
                content = [
                    {"type": "video", "video": video_imgs},
                    {"type": "text", "text": "<|fim_pad|>"}  # Align video 结束标记
                ]
                
                # Each chunk in chunk_imgs is a group of frames (NUM_STEPS=4)
                # We process each group as a video to leverage Qwen's temporal compression
                for chunk in chunk_imgs:
                    content.append({"type": "video", "video": chunk})
                    # Add a special token after each group video as a learnable CLS token
                    content.append({"type": "text", "text": "<|file_sep|>"})
                
                # Update Num Metadata
                num_c = len(chunk_imgs)
                if is_ref_row:
                    batch_num_mains.append(0)
                    batch_num_refs.append(num_c)
                    ref_seq_lens_list.extend([seq_len_val] * num_c)
                    
                    # Store Masks for Ref Row
                    if is_cut_mask is not None and is_cut_mask.numel() > 0:
                        batch_is_cut_masks.append(is_cut_mask)
                    else:
                        batch_is_cut_masks.append(torch.zeros(num_c, dtype=torch.float))
                    
                    batch_has_dustbin.append(has_dustbin)

                else:
                    batch_num_mains.append(num_c)
                    batch_num_refs.append(0)
                    main_seq_lens_list.extend([seq_len_val] * num_c)

                    # Store Masks for Main Row (Dummy)
                    batch_is_cut_masks.append(torch.zeros(num_c, dtype=torch.float))
                    batch_has_dustbin.append(False)

                
                batch_group_ids.append(group_id)
                batch_chunk_ids.append(chunk_id)
                
                # Update row-level metadata
                batch_row_frame_paths.append(frame_paths)
                batch_row_ref_frame_paths.append(ref_frame_paths)
                batch_row_names.append(name)
                batch_row_ref_names.append(ref_name)
                batch_row_dataset_names.append(dataset_name)
                batch_row_aug_params.append(aug_params)
                batch_row_ref_aug_params.append(ref_aug_params)

                # --- Processing ---
                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Extract Video Info (Pass PIL Images)
                # 1. Main Align/Ref Align Video
                video_input_pair, video_kwargs = get_video_info(
                    video_imgs,
                    min_pixels=224 * 224,
                    max_pixels=256 * 256,
                    width=None,
                    height=None,
                    fps=2.0,
                    image_patch_size=16,
                    return_video_metadata=True
                )
                video_tensor, video_metadata = video_input_pair
                
                # 2. Group Videos
                group_video_tensors = []
                group_video_metadatas = []
                for chunk in chunk_imgs:
                    gv_input_pair, gv_kwargs = get_video_info(
                        chunk,
                        min_pixels=224 * 224,
                        max_pixels=256 * 256,
                        width=None,
                        height=None,
                        fps=2.0,
                        image_patch_size=16,
                        return_video_metadata=True
                    )
                    gv_tensor, gv_metadata = gv_input_pair
                    group_video_tensors.append(gv_tensor)
                    group_video_metadatas.append(gv_metadata)
                
                # Run Processor
                inputs = self.processor(
                    text=[text],
                    images=None, # No individual images anymore
                    videos=[video_tensor] + group_video_tensors, 
                    padding=False,
                    return_tensors="pt",
                    video_metadata=[video_metadata] + group_video_metadatas,
                    **video_kwargs # Assuming video_kwargs are consistent
                )
                
                batch_input_ids.append(inputs['input_ids'][0])
                if 'pixel_values' in inputs:
                    batch_pixel_values.append(inputs['pixel_values'])
                if 'image_grid_thw' in inputs:
                    batch_image_grid_thw.append(inputs['image_grid_thw'])
                if 'pixel_values_videos' in inputs:
                    batch_pixel_values_videos.append(inputs['pixel_values_videos'])
                if 'video_grid_thw' in inputs:
                    batch_video_grid_thw.append(inputs['video_grid_thw'])

            # Multi-Pass Logic to maintain symmetric computes
            for i in range(len(batch)):
                # --- 1. Load and Augment for this sample ---
                # Gather all Main paths
                main_paths = batch[i]['frame_paths']
                
                align_paths = batch[i].get('align_frame_paths', [])
                
                # --- Cut Logic ---
                full_align_len = len(align_paths)
                
                if self.mode == 'train' and 'ref_chosen_steps' in batch[i]:
                    n_steps = len(batch[i]['ref_chosen_steps'])
                    is_cut_mask = torch.zeros(n_steps, dtype=torch.float)
                else:
                    is_cut_mask = None
                
                has_dustbin = False
                if 'ref_chosen_steps' in batch[i] and len(batch[i]['ref_chosen_steps']) > 0:
                    if batch[i]['ref_chosen_steps'][0] == -1:
                        has_dustbin = True
                        
                if self.mode == 'train' and full_align_len > 0:
                     start_idx = -1
                     end_idx = -1
                     
                     # Check for precomputed consistent cut
                     if '_precomputed_cut' in batch[i]:
                         start_idx, end_idx = batch[i]['_precomputed_cut']
                     else:
                         cut_pct = random.uniform(CONFIG.TRAIN.CUT_RANGE[0], CONFIG.TRAIN.CUT_RANGE[1])
                         cut_len = int(full_align_len * cut_pct)
                         
                         if cut_len > 0:
                             max_start = full_align_len - cut_len
                             start_idx = random.randint(0, max_start)
                             end_idx = start_idx + cut_len
                     
                     if start_idx >= 0 and end_idx >= 0:
                         # Apply Cut to Paths
                         align_paths = align_paths[:start_idx] + align_paths[end_idx:]
                         
                         # Create Mask
                         if 'ref_chosen_steps' in batch[i]:
                             r_steps = batch[i]['ref_chosen_steps']
                             # Dustbin -1 is never cut
                             mask = (r_steps >= start_idx) & (r_steps < end_idx)
                             is_cut_mask = mask.float()
                
                # Downsample align_paths
                target_num = getattr(CONFIG.TRAIN, 'NUM_ALIGN_FRAMES', 24)
                if len(align_paths) > target_num:
                    indices = np.linspace(0, len(align_paths)-1, target_num, dtype=int)
                    align_paths = [align_paths[k] for k in indices]
                
                # Symmetric: Main uses same align paths (Ref abstraction)
                main_align_paths = align_paths
                
                # Gather Ref paths
                ref_paths = batch[i].get('ref_frame_paths', [])
                
                # Symmetrice: Ref uses same align paths (Main abstraction -> Ref abstraction per Fixed Grounding)
                ref_align_paths = align_paths

                main_imgs_loaded = self.load_imgs(main_align_paths + main_paths)
                ref_imgs_loaded = self.load_imgs(ref_align_paths + ref_paths)
                
                # Augment
                aug_main, params_main = self.augment_sequence(main_imgs_loaded, self.color_jitter if self.do_color_jitter else None, self.do_random_flip, self.do_random_crop)
                aug_ref, params_ref = self.augment_sequence(ref_imgs_loaded, self.color_jitter if self.do_color_jitter else None, self.do_random_flip, self.do_random_crop)
                
                # Split back
                main_align_imgs = aug_main[:len(main_align_paths)]
                main_group_imgs_flat = aug_main[len(main_align_paths):]
                
                ref_align_imgs = aug_ref[:len(align_paths)]
                ref_group_imgs_flat = aug_ref[len(align_paths):]
                
                # Chunk grouped images
                main_chunks = [main_group_imgs_flat[j : j + context_size] for j in range(0, len(main_group_imgs_flat), context_size)]
                ref_chunks = [ref_group_imgs_flat[j : j + context_size] for j in range(0, len(ref_group_imgs_flat), context_size)]
                
                # Store for processing
                batch[i]['_aug_main_align'] = main_align_imgs
                batch[i]['_aug_main_chunks'] = main_chunks
                batch[i]['_aug_ref_align'] = ref_align_imgs
                batch[i]['_aug_ref_chunks'] = ref_chunks
                batch[i]['_aug_params'] = params_main
                batch[i]['_aug_ref_params'] = params_ref
                batch[i]['_is_cut_mask'] = is_cut_mask
                batch[i]['_has_dustbin'] = has_dustbin

            # Pass 1: Handle all MAIN sequences (Seq 1: Align_r + Main_Groups)
            for i in range(len(batch)):
                sl = batch[i].get('seq_lens', torch.tensor(0))
                process_packed_sequence(
                    batch[i]['_aug_ref_align'], 
                    batch[i]['_aug_main_chunks'], 
                    is_ref_row=False, 
                    seq_len_val=sl,
                    group_id=batch[i].get('group_id', i),
                    chunk_id=batch[i].get('chunk_id', 0),
                    frame_paths=batch[i]['frame_paths'],
                    ref_frame_paths=batch[i].get('ref_frame_paths', []),
                    name=batch[i].get('name', 'unknown'),
                    ref_name=batch[i].get('ref_name', 'unknown'),
                    dataset_name=batch[i].get('dataset_name', 'unknown'),
                    aug_params=batch[i]['_aug_params'],
                    ref_aug_params=batch[i]['_aug_ref_params'],
                    is_cut_mask=None,
                    has_dustbin=False
                )

            # Pass 2: Handle all REF sequences (Seq 2: Align_c + Ref_Groups)
            for i in range(len(batch)):
                if 'ref_frame_paths' in batch[i]:
                    rsl = batch[i].get('ref_seq_lens', batch[i].get('candidate_seq_lens', torch.tensor(0)))
                    process_packed_sequence(
                        batch[i]['_aug_main_align'], 
                        batch[i]['_aug_ref_chunks'], 
                        is_ref_row=True, 
                        seq_len_val=rsl,
                        group_id=batch[i].get('group_id', i),
                        chunk_id=batch[i].get('chunk_id', 0),
                        frame_paths=batch[i]['frame_paths'],
                        ref_frame_paths=batch[i].get('ref_frame_paths', []),
                        name=batch[i].get('name', 'unknown'),
                        ref_name=batch[i].get('ref_name', 'unknown'),
                        dataset_name=batch[i].get('dataset_name', 'unknown'),
                        aug_params=batch[i]['_aug_params'],
                        ref_aug_params=batch[i]['_aug_ref_params'],
                        is_cut_mask=batch[i]['_is_cut_mask'],
                        has_dustbin=batch[i]['_has_dustbin']
                    )

            # Collate (Pad & Cat)
            input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.pad_token_id)
            attention_mask = (input_ids != self.pad_token_id)
            
            pixel_values = torch.cat(batch_pixel_values, dim=0) if batch_pixel_values else None
            image_grid_thw = torch.cat(batch_image_grid_thw, dim=0) if batch_image_grid_thw else None
            
            pixel_values_videos = torch.cat(batch_pixel_values_videos, dim=0) if batch_pixel_values_videos else None
            video_grid_thw = torch.cat(batch_video_grid_thw, dim=0) if batch_video_grid_thw else None
            
            # Combine Metadata in [All_Mains, All_Refs] order
            all_seq_lens_list = main_seq_lens_list + ref_seq_lens_list
            if all_seq_lens_list:
                seq_lens_tensor = torch.stack(all_seq_lens_list)
            else:
                seq_lens_tensor = None

            # Add to collated batch
            # Stack masks
            if batch_is_cut_masks:
                is_cut_masks_tensor = torch.stack(batch_is_cut_masks)
            else:
                is_cut_masks_tensor = None
            
            has_dustbin_tensor = torch.tensor(batch_has_dustbin, dtype=torch.bool)

            # Get special token IDs from CONFIG
            cls_token_id = CONFIG.SPECIAL_TOKENS.CLS_TOKEN_ID
            align_end_token_id = CONFIG.SPECIAL_TOKENS.ALIGN_END_TOKEN_ID

            collated_batch['qwen_input'] = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
                'pixel_values_videos': pixel_values_videos,
                'video_grid_thw': video_grid_thw,
                'seq_lens': seq_lens_tensor,
                # Info for model to split embeddings
                'num_mains': torch.tensor(batch_num_mains, dtype=torch.long),
                'num_refs': torch.tensor(batch_num_refs, dtype=torch.long),
                'group_ids': torch.tensor(batch_group_ids, dtype=torch.long),
                'chunk_ids': torch.tensor(batch_chunk_ids, dtype=torch.long),
                'is_cut_masks': is_cut_masks_tensor,
                'has_dustbin': has_dustbin_tensor,
                'cls_token_id': torch.tensor(cls_token_id, dtype=torch.long),
                'align_end_token_id': torch.tensor(align_end_token_id, dtype=torch.long)
            }
            
            # Pass augmentation params and paths for logging (aligned with Qwen rows)
            collated_batch['aug_params'] = batch_row_aug_params
            collated_batch['ref_aug_params'] = batch_row_ref_aug_params
            collated_batch['frame_paths'] = batch_row_frame_paths
            collated_batch['ref_frame_paths'] = batch_row_ref_frame_paths
            collated_batch['name'] = batch_row_names
            collated_batch['ref_name'] = batch_row_ref_names
            collated_batch['dataset_name'] = batch_row_dataset_names

        # Debug: Print shapes
        # print("\n--- TCCCollator Debug ---")
        # for k, v in collated_batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape}")
        #     elif isinstance(v, list):
        #         print(f"{k}: list len={len(v)}")
        #     elif isinstance(v, dict):
        #         print(f"{k}: dict keys={list(v.keys())}")
        #         for sub_k, sub_v in v.items():
        #             if isinstance(sub_v, torch.Tensor):
        #                 print(f"  {sub_k}: {sub_v.shape}")
        #             else:
        #                 print(f"  {sub_k}: {type(sub_v)}")
        # print("-------------------------\n")

        return collated_batch

class LiberoDataset(Dataset):
    def __init__(self, mode='train', transform=None, video_paths_json=None, processor=None):
        self.mode = mode
        self.transform = transform
        self.processor = processor
        self.debug_step = 0
        
        self.video_paths = []
        self.video_dataset_names = []
        self.video_weights = []
        self.task_paths_cache = {}
        
        # Pickle data support
        self.pickle_data = None
        self.expert_data = None
        self.pickle_path = None
        self.expert_path = None
        self.is_rollout_mode = False
        self.rollout_images = None
        self.rollout_wrist_images = None
        self.task_descriptions = None
        
        # Check if input is pickle file
        is_pickle = isinstance(video_paths_json, str) and video_paths_json.endswith('.pkl')
        
        if is_pickle:
            # Load pickle data
            import pickle
            print(f"Loading video paths from pickle: {video_paths_json}")
            self.pickle_path = video_paths_json  # Save for traceability
            
            with open(video_paths_json, 'rb') as f:
                self.pickle_data = pickle.load(f)
            
            # Check if this is rollout mode (has 'rollout_batch' key)
            self.is_rollout_mode = 'rollout_batch' in self.pickle_data
            
            if self.is_rollout_mode:
                print(f"  Detected rollout mode pickle")
                rb = self.pickle_data['rollout_batch']
                self.task_descriptions = rb['task_descriptions']
                num_videos = len(self.task_descriptions)
                
                # Reorganize images: (N_chunks, B, Chunk_size, H, W, 3) -> (B, Total_frames, H, W, 3)
                full_imgs = rb['observation/full_image_list']
                n_chunks, b_size, c_size = full_imgs.shape[:3]
                self.rollout_images = full_imgs.permute(1, 0, 2, 3, 4, 5).reshape(b_size, n_chunks * c_size, *full_imgs.shape[3:])
                
                if 'observation/wrist_image' in rb:
                    wrist_imgs = rb['observation/wrist_image']
                    self.rollout_wrist_images = wrist_imgs.permute(1, 0, 2, 3, 4)
                else:
                    self.rollout_wrist_images = None
                
                # video_paths are task indices
                self.video_paths = list(range(num_videos))
                self.video_dataset_names = ["rollout_dataset"] * num_videos
                self.video_weights = [1.0] * num_videos
            else:
                # Standard expert pickle mode: {task_name: {view: array}}
                self.video_paths = sorted(list(self.pickle_data.keys()))
                self.video_dataset_names = ["pickle_dataset"] * len(self.video_paths)
                self.video_weights = [1.0] * len(self.video_paths)
            
            # Try to load corresponding expert data
            pkl_dir = os.path.dirname(video_paths_json)
            expert_candidates = [
                os.path.join(pkl_dir, 'libero_10-expert-video.pkl'),
                os.path.join(pkl_dir, os.path.basename(video_paths_json).replace('rollout', 'expert')),
                os.path.join(pkl_dir, 'expert.pkl'),
            ]
            
            for expert_path in expert_candidates:
                if os.path.exists(expert_path):
                    print(f"  Loading expert data from: {expert_path}")
                    self.expert_path = expert_path
                    with open(expert_path, 'rb') as f:
                        self.expert_data = pickle.load(f)
                    break
            
            if self.expert_data is None:
                print(f"  Warning: No expert data found. Will use rollout data as both query and reference.")
            
            print(f"  Loaded {len(self.video_paths)} videos from pickle file")
        
        # 1. Try loading from provided argument (supporting list/comma-separated)
        elif video_paths_json:
            if isinstance(video_paths_json, str):
                paths = [p.strip() for p in video_paths_json.split(',')]
            elif isinstance(video_paths_json, list):
                paths = video_paths_json
            else:
                paths = []
            
            # Helper to extract dataset name
            def get_dataset_name(json_path):
                # e.g. /path/to/berkeley_autolab_ur5_video_paths.json -> berkeley_autolab_ur5
                base = os.path.basename(json_path)
                name = base.replace('_video_paths.json', '').replace('.json', '')
                return name

            # First pass: Count totals to calculate probabilities
            dataset_counts = {}
            dataset_weights = {} 
            temp_paths = {} # name -> list of paths

            for path in paths:
                if os.path.exists(path):
                    print(f"Loading video paths from: {path}")
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            ds_name = get_dataset_name(path)
                            temp_paths[ds_name] = data
                            dataset_counts[ds_name] = len(data)
                            # Get weight from config, default to 1.0
                            ds_weight = CONFIG.DATA.DATASET_WEIGHTS.get(ds_name, 1.0)
                            dataset_weights[ds_name] = ds_weight
                        else:
                            print(f"Warning: {path} content is not a list.")
                else:
                    print(f"Warning: Provided video_paths file not found: {path}")
            
            # Report Probabilities
            total_weighted_count = sum(c * w for k, c in dataset_counts.items() for w in [dataset_weights[k]])
            print("\n--- Dataset Sampling Probabilities ---")
            for name, count in dataset_counts.items():
                w = dataset_weights[name]
                prob = (count * w) / total_weighted_count if total_weighted_count > 0 else 0
                print(f"Dataset: {name}, Count: {count}, Weight: {w}, Sampling Prob: {prob:.4f}")
            print("--------------------------------------\n")

            # Flatten into lists
            for name, paths_list in temp_paths.items():
                self.video_paths.extend(paths_list)
                self.video_dataset_names.extend([name] * len(paths_list))
                self.video_weights.extend([dataset_weights[name]] * len(paths_list))

        
        if not self.video_paths:
            print(f"WARNING: No video paths loaded.")
        
        # Skip path validation for all modes
        print(f"Total video paths loaded: {len(self.video_paths)}")
        
        # Views to sample from
        self.views = ['images', 'gripper_images']
        
        # Prepare weights for WeightedRandomSampler (per item in __len__)
        # Each video has len(self.views) items.
        # weight list must match __len__ size which is len(videos) * len(views)
        self.weights = []
        for w in self.video_weights:
            self.weights.extend([w] * len(self.views))
        
        dataset_type = "pickle" if is_pickle else "json"
        print(f"LiberoDataset initialized with {len(self.video_paths)} videos and views: {self.views} (source: {dataset_type})")
        
    def __len__(self):
        return len(self.video_paths) * len(self.views)
    
    def _get_files(self, path):
        """
        Get list of frame paths from a directory.
        Supports both image files (jpg/png) and video files (mp4).
        For videos, returns a list of virtual paths in the format:
        "video://<video_path>::<frame_index>::<width>x<height>"
        """
        # Check for mp4 files first
        mp4_files = glob.glob(os.path.join(path, '*.mp4'))
        
        if mp4_files:
            # Use the first mp4 file found
            video_path = mp4_files[0]
            
            # Get video length and dimensions using decord
            # Note: VideoReader only reads metadata here, not the entire video
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                num_frames = len(vr)
                
                # Get original video dimensions from first frame
                sample_frame = vr[0]
                orig_height, orig_width = sample_frame.shape[:2]
                
                del vr  # Immediately release resources after getting dimensions
                
                # Calculate target size using smart_resize
                # Use same parameters as get_video_info in data_utils.py
                try:
                    target_height, target_width = smart_resize(
                        height=orig_height,
                        width=orig_width,
                        factor=32,  # image_patch_size (from Qwen VL config)
                        min_pixels=224 * 224,
                        max_pixels=256 * 256
                    )
                except ValueError as e:
                    print(f"Warning: smart_resize failed for {video_path}: {e}")
                    # Fallback to fixed size
                    target_width, target_height = 224, 224
                
                # Generate virtual file list: one entry per frame with target size
                # Format: "video://<video_path>::<frame_index>::<width>x<height>"
                files = [
                    f"video://{video_path}::{i}::{target_width}x{target_height}" 
                    for i in range(num_frames)
                ]
                return files
            except Exception as e:
                print(f"Error reading video {video_path}: {e}. Falling back to images.")
                # Fall through to image loading
        
        # Fallback to image files
        files = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png'))
        
        def get_sort_key(f):
            name = os.path.splitext(os.path.basename(f))[0]
            if 'droid' in path.lower():
                # droid format: extract last number after underscore
                return int(name.split('_')[-1])
            elif 'taco_play' in path.lower():
                # taco_play format: 0057_gripper.jpg or 0057.jpg -> extract first part
                return int(name.split('_')[0])
            else:
                # Default: pure number filename
                return int(name)
        
        files.sort(key=get_sort_key)
        return files

    def _sample_steps(self, seq_len, num_steps):
        """Sample frames based on strategy."""
        stride = CONFIG.DATA.STRIDE
        sampling_strategy = CONFIG.DATA.SAMPLING_STRATEGY
        
        if sampling_strategy == 'stride':
            max_offset = max(1, seq_len - stride * num_steps)
            offset = random.randint(0, max_offset - 1)
            steps = np.arange(offset, offset + num_steps * stride + 1, stride)
            steps = steps[:num_steps]
            steps = np.minimum(seq_len - 1, steps)
            
        elif sampling_strategy == 'offset_uniform':
            random_offset = int(CONFIG.DATA.RANDOM_OFFSET)
            if seq_len < random_offset:
                # Fallback if video is too short
                steps = np.arange(0, min(seq_len, num_steps))
                # Pad if needed
                if len(steps) < num_steps:
                    steps = np.pad(steps, (0, num_steps - len(steps)), 'edge')
            else:
                if num_steps <= seq_len - random_offset:
                    # Sample random offset
                    offset = random_offset
                    # Sample random frames from [offset, seq_len)
                    available_indices = np.arange(offset, seq_len)
                    np.random.shuffle(available_indices)
                    steps = available_indices[:num_steps]
                    # Sort them to keep temporal order
                    steps = np.sort(steps)
                else:
                    # Fallback: sample all available
                    steps = np.arange(0, min(seq_len, num_steps))
                    if len(steps) < num_steps:
                        steps = np.pad(steps, (0, num_steps - len(steps)), 'edge')
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
            
        # TCN Positive Window Sampling
        if 'tcn' in CONFIG.TRAINING_ALGO:
            pos_window = CONFIG.TCN.POSITIVE_WINDOW
            pos_steps = []
            for step in steps:
                min_val = max(0, step - pos_window)
                max_val = max(min_val + 1, step) # Ensure range is valid
                pos_step = random.randint(min_val, max_val - 1) if max_val > min_val else step
                pos_steps.append(pos_step)
            
            # Interleave pos_steps and steps: [p0, s0, p1, s1, ...]
            combined_steps = np.empty(len(steps) * 2, dtype=steps.dtype)
            combined_steps[0::2] = pos_steps
            combined_steps[1::2] = steps
            steps = combined_steps

        return steps

    def _get_stride(self, video_path):
        dataset_name = os.path.basename(os.path.dirname(video_path))
        
        for ds in datasets_with_lower_frequency:
            if ds in dataset_name:
                return random.randint(3, 5)
        
        for ds in datasets_with_higher_frequency:
            if ds in dataset_name:
                return random.randint(15, 20)
                
        return CONFIG.DATA.FRAME_STRIDE

    def _get_context_steps(self, steps, seq_len, video_path, reverse=False):
        """Get multiple context steps for each chosen step."""
        num_context = CONFIG.DATA.NUM_STEPS
        stride = self._get_stride(video_path)
        
        # steps shape: (N,)
        # output shape: (N * num_context,)
        
        context_steps = []
        for step in steps:
            if step == -1:
                # Dustbin Chunk: return -1s
                indices = np.full(num_context, -1, dtype=int)
                context_steps.append(indices)
                continue

            # Range: [step - (num-1)*stride, ..., step + stride] with stride
            if not reverse:
                start = step - (num_context - 1) * stride
                end = step + stride
                # Generate indices
                indices = np.arange(start, end, stride)
            else:
                # [step + (num-1)*stride, ..., step]
                start = step + (num_context - 1) * stride
                end = step - stride
                # Generate indices in descending order
                indices = np.arange(start, end, -stride)
            
            # Clip to valid range [0, seq_len-1]
            indices = np.clip(indices, 0, seq_len - 1)
            context_steps.append(indices)
            
        return np.concatenate(context_steps)

    def _load_frames(self, files, steps):
        """
        Load frames from either file paths, numpy arrays, or torch tensors.
        
        Args:
            files: List of file paths (strings), numpy array (N, H, W, 3), or torch.Tensor
            steps: Array of frame indices to load
        
        Returns:
            Tensor of shape (T, C, H, W)
        """
        frames = []
        is_numpy_array = isinstance(files, np.ndarray)
        is_torch_tensor = torch.is_tensor(files)
        
        for step in steps:
            if step == -1:
                # Dustbin: Black Image
                img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
            else:
                if is_torch_tensor:
                    # Load from torch tensor (rollout data)
                    img_array = files[step]  # (H, W, 3)
                    if img_array.device != torch.device('cpu'):
                        img_array = img_array.cpu()
                    img = Image.fromarray(img_array.numpy().astype(np.uint8)).convert('RGB')
                elif is_numpy_array:
                    # Load from numpy array (pickle data)
                    img_array = files[step]  # (H, W, 3)
                    img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
                else:
                    # Load from file path
                    path = files[step]
                    with Image.open(path) as img:
                        img = img.convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            frames.append(img)
                
        return torch.stack(frames) # (T, C, H, W)

    def _get_image_dirs(self, path):
        """Find subdirectories to use as views. Uses DATASET_VIEW_CONFIG when path matches a dataset; else keyword filter."""
        path_lower = path.lower() if isinstance(path, str) else ""
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        # Resolve dataset name from path (first matching key in DATASET_VIEW_CONFIG)
        custom_views = None
        for ds_name, allowed in DATASET_VIEW_CONFIG.items():
            if ds_name in path_lower:
                custom_views = set(allowed)
                break

        if custom_views is not None:
            # Restrict to allowed views only (intersection with existing subdirs)
            image_dirs = [d for d in subdirs if d in custom_views]
            return image_dirs

        # Default: keyword-based filter for datasets not in DATASET_VIEW_CONFIG
        keywords = ['image', 'gripper', 'rgb']
        image_dirs = [d for d in subdirs if any(k in d.lower() for k in keywords)]
        return image_dirs

    def _load_video_data_from_pickle(self, video_idx):
        """
        Stage 1: Data Loading from Pickle.
        Returns a dict with main_files, ref_files, align_files, instructions, etc.
        
        Supports two pickle formats:
        1. Rollout mode: {rollout_batch: {observation/full_image_list, task_descriptions, ...}}
        2. Standard mode: {task_name: {view: numpy_array}}
        """
        dataset_name = self.video_dataset_names[video_idx] if video_idx < len(self.video_dataset_names) else "pickle_dataset"
        view = 'images'
        
        if self.is_rollout_mode:
            # Rollout mode: video_paths[video_idx] is an integer index
            video_idx_int = self.video_paths[video_idx]
            task_name = self.task_descriptions[video_idx_int]
            
            # Main files: rollout images for this video index
            main_files = self.rollout_images[video_idx_int]  # (Total_frames, H, W, 3)
            
            # Reference files: expert data for this task
            if self.expert_data is not None and task_name in self.expert_data:
                if view in self.expert_data[task_name]:
                    ref_files = self.expert_data[task_name][view]
                else:
                    ref_files = main_files
            else:
                ref_files = main_files
            
            # Use task name as video_path for logical paths
            video_path = task_name
            ref_video_path = task_name
            align_video_path = task_name
            instruction = task_name
            ref_instruction = task_name
        else:
            # Standard expert pickle mode: {task_name: {view: array}}
            video_path = self.video_paths[video_idx]
            
            try:
                task_data = self.pickle_data[video_path]
                if isinstance(task_data, dict) and view in task_data:
                    main_files = task_data[view]
                else:
                    return None
            except (KeyError, TypeError) as e:
                print(f"Error accessing pickle_data[{video_path}]: {e}")
                return None
            
            # Reference from expert_data
            if self.expert_data is not None and video_path in self.expert_data:
                expert_task_data = self.expert_data[video_path]
                if isinstance(expert_task_data, dict) and view in expert_task_data:
                    ref_files = expert_task_data[view]
                else:
                    ref_files = main_files
            else:
                ref_files = main_files
            
            ref_video_path = video_path
            align_video_path = video_path
            instruction = video_path
            ref_instruction = video_path
        
        # Align files: use reference files
        align_files = ref_files
        
        # Initial frames: store as logical path with pickle file path for traceability
        initial_frame_path = f"{self.pickle_path}:{video_path}/{view}/frame_0"
        ref_source = self.expert_path if self.expert_path else self.pickle_path
        ref_initial_frame_path = f"{ref_source}:{ref_video_path}/{view}/frame_0"
        
        return {
            'video_path': video_path,
            'dataset_name': dataset_name,
            'view': view,
            'main_files': main_files,
            'ref_video_path': video_path,  # In pickle mode, ref is from expert of same task
            'ref_files': ref_files,
            'align_video_path': video_path,
            'align_files': align_files,
            'instruction': instruction,
            'ref_instruction': ref_instruction,
            'initial_frame_path': initial_frame_path,
            'ref_initial_frame_path': ref_initial_frame_path
        }

    def _load_video_data_from_json(self, video_idx):
        """
        Stage 1: Data Loading from JSON/filesystem.
        Returns a dict with main_files, ref_files, align_files, instructions, etc.
        """
        video_path = self.video_paths[video_idx]
        dataset_name = self.video_dataset_names[video_idx] if video_idx < len(self.video_dataset_names) else "unknown"
        
        # Select view for main video
        image_dirs = self._get_image_dirs(video_path)
        if not image_dirs:
            if os.path.exists(os.path.join(video_path, 'images')):
                image_dirs = ['images']
            elif os.path.exists(os.path.join(video_path, 'gripper_images')):
                image_dirs = ['gripper_images']
            else:
                image_dirs = [d for d in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, d))]
        
        if not image_dirs:
            return None  # Signal failure
        
        # Select view: fixed 'images' in eval mode, random in train mode
        if self.mode == 'eval':
            view = 'images' if 'images' in image_dirs else image_dirs[0]
        else:
            view = random.choice(image_dirs)
        
        # Get Task Paths (Reference Videos)
        task_paths_file = os.path.join(video_path, 'task_paths.json')
        if video_path not in self.task_paths_cache:
            if os.path.exists(task_paths_file):
                with open(task_paths_file, 'r') as f:
                    self.task_paths_cache[video_path] = json.load(f)
            else:
                self.task_paths_cache[video_path] = {}
        
        task_paths = self.task_paths_cache[video_path]
        same_pool_keys = ["same", "100-95"]
        candidate_pool = []
        for key in same_pool_keys:
            if key in task_paths and task_paths[key]:
                candidate_pool.extend(task_paths[key])
        
        ref_video_path = random.choice(candidate_pool) if candidate_pool else video_path
        align_video_path = ref_video_path

        # Get available views for reference video
        ref_image_dirs = self._get_image_dirs(ref_video_path)
        if not ref_image_dirs:
            ref_video_path = video_path
            ref_image_dirs = image_dirs
        
        if view in ref_image_dirs:
            ref_view = view
        else:
            ref_view = 'images' if (self.mode == 'eval' and 'images' in ref_image_dirs) else random.choice(ref_image_dirs)

        # Get available views for align video
        align_image_dirs = self._get_image_dirs(align_video_path)
        if not align_image_dirs:
            align_video_path = video_path
            align_image_dirs = image_dirs
        
        if view in align_image_dirs:
            align_view = view
        else:
            align_view = 'images' if (self.mode == 'eval' and 'images' in align_image_dirs) else random.choice(align_image_dirs)

        # Get files
        main_view_path = os.path.join(video_path, view)
        ref_view_path = os.path.join(ref_video_path, ref_view)
        align_view_path = os.path.join(align_video_path, align_view)
        
        main_files = self._get_files(main_view_path)
        ref_files = self._get_files(ref_view_path)
        align_files = self._get_files(align_view_path)
        
        if not main_files:
            return None  # Signal failure
        if not ref_files:
            ref_files = main_files
            ref_video_path = video_path
        if not align_files:
            align_files = main_files
            align_video_path = video_path

        # Read instructions
        instruction_path = os.path.join(video_path, 'instruction.txt')
        instruction = "Perform the task."
        if os.path.exists(instruction_path):
            with open(instruction_path, 'r') as f:
                instruction = f.read().strip()
        
        ref_instruction_path = os.path.join(ref_video_path, 'instruction.txt')
        ref_instruction = "Perform the task."
        if os.path.exists(ref_instruction_path):
            with open(ref_instruction_path, 'r') as f:
                ref_instruction = f.read().strip()
        
        return {
            'video_path': video_path,
            'dataset_name': dataset_name,
            'view': view,
            'main_files': main_files,
            'ref_video_path': ref_video_path,
            'ref_files': ref_files,
            'align_video_path': align_video_path,
            'align_files': align_files,
            'instruction': instruction,
            'ref_instruction': ref_instruction,
            'initial_frame_path': main_files[0],
            'ref_initial_frame_path': ref_files[0]
        }

    def _get_item_impl(self, index):
        # Map index to video
        video_idx = index // len(self.views)
        
        # ==================== STAGE 1: Data Loading ====================
        if self.pickle_data is not None:
            # Load from pickle
            loaded_data = self._load_video_data_from_pickle(video_idx)
        else:
            # Load from JSON/filesystem
            loaded_data = self._load_video_data_from_json(video_idx)
        
        if loaded_data is None:
            # Retry with random index
            return self._get_item_impl(random.randint(0, len(self) - 1))
        
        # Extract loaded data
        video_path = loaded_data['video_path']
        dataset_name = loaded_data['dataset_name']
        main_files = loaded_data['main_files']
        ref_video_path = loaded_data['ref_video_path']
        ref_files = loaded_data['ref_files']
        align_video_path = loaded_data['align_video_path']
        align_files = loaded_data['align_files']
        instruction = loaded_data['instruction']
        ref_instruction = loaded_data['ref_instruction']
        initial_frame_path = loaded_data['initial_frame_path']
        ref_initial_frame_path = loaded_data['ref_initial_frame_path']
        
        # Runtime filtering: check if video has enough frames
        min_frames = CONFIG.TRAIN.NUM_FRAMES
        if len(main_files) < min_frames:
            raise ValueError(f"Main video too short: {len(main_files)} frames, expected at least {min_frames}. Path: {video_path}")
        
        if len(ref_files) < min_frames:
            raise ValueError(f"Reference video too short: {len(ref_files)} frames, expected at least {min_frames}. Path: {ref_video_path}")

        # ==================== STAGE 2: Unified Processing ====================
        # Sampling: num_steps tiers are 4×/2×/1× NUM_FRAMES (aligned with CHUNK_PROBS M multipliers)
        nf = min_frames
        steps_4x, steps_2x, steps_1x = 4 * nf, 2 * nf, nf
        max_allowed = getattr(CONFIG.TRAIN, 'MAX_BATCH_FRAMES', steps_4x)
        seq_len = min(len(main_files), len(ref_files))

        if seq_len >= steps_4x and max_allowed >= steps_4x:
            num_steps = steps_4x
        elif seq_len >= steps_2x and max_allowed >= steps_2x:
            num_steps = steps_2x
        else:
            num_steps = steps_1x
        
        main_steps = self._sample_steps(len(main_files), num_steps)
        
        # Dustbin Logic: Sample N-1 frames, prepend -1 (Dustbin)
        # if self.mode == 'train':
        #     ref_steps_real = self._sample_steps(len(ref_files), num_steps - 1)
        #     ref_steps = np.concatenate([[-1], ref_steps_real])
        # else:
        #     # Eval mode might not use dustbin or handle it differently?
        #     # Assuming typically TCC eval doesn't use dustbin or uses standard flow. 
        #     # If we want consistent eval, we should probably follow suit or keep as is.
        #     # Keeping as is for eval to avoid breaking metrics unless specified.
        #     ref_steps = self._sample_steps(len(ref_files), num_steps)
        ref_steps_real = self._sample_steps(len(ref_files), num_steps - 1)
        ref_steps = np.concatenate([[-1], ref_steps_real])
        
        # User requested modification: 
        # align_frame_paths (abstraction of Ref) and main_align_frame_paths (abstraction of Main)
        # should be fixed to CONFIG.TRAIN.NUM_ALIGN_FRAMES, downsampled from the long sequence.
        
        num_align = getattr(CONFIG.TRAIN, 'NUM_ALIGN_FRAMES', 24)
        
        def downsample_indices(steps, target_num):
            if len(steps) == 0:
                return steps
            # Always sample exactly target_num frames (upsample if short / downsample if long)
            indices = np.linspace(0, len(steps) - 1, target_num, dtype=int)
            return steps[indices]
            
        # Fixed Grounding: Always use Ref Video
        # Full Video Return for Cut Logic in Collator
        align_frame_paths = ref_files
        main_align_frame_paths = ref_files
        
        # Legacy / Tensor loading needs downsampled version
        all_ref_indices = np.arange(len(ref_files))
        align_indices_legacy = downsample_indices(all_ref_indices, num_align)
        
        align_files_to_load = ref_files
        align_steps_to_load = align_indices_legacy

        do_reverse_main = (self.mode == 'train' and random.random() < getattr(CONFIG.AUGMENTATION, 'REVERSE_PROB', 0.5))
        do_reverse_ref = (self.mode == 'train' and random.random() < getattr(CONFIG.AUGMENTATION, 'REVERSE_PROB', 0.5))
        
        if do_reverse_main:
            main_steps = main_steps[::-1].copy()
        if do_reverse_ref:
            # Only reverse the REAL frames part
            if self.mode == 'train':
                # ref_steps[0] is dustbin, reverse [1:]
                ref_steps[1:] = ref_steps[1:][::-1].copy()
            else:
                ref_steps = ref_steps[::-1].copy()

        # Context Expansion (Sampling sparse chunks for the groups)
        main_context_steps = self._get_context_steps(main_steps, len(main_files), video_path, reverse=do_reverse_main)
        ref_context_steps = self._get_context_steps(ref_steps, len(ref_files), ref_video_path, reverse=do_reverse_ref)
        
        # Construct frame_paths
        # For Collator: pass actual data (numpy/tensor) or file paths
        # For logging: store logical paths separately
        is_pickle_mode = isinstance(main_files, (np.ndarray, torch.Tensor))
        view = loaded_data['view']
        
        if is_pickle_mode:
            # Pickle mode: pass numpy arrays directly to Collator
            main_frame_paths = []
            main_frame_paths_logical = []  # For JSON logging
            for i in main_context_steps:
                if i == -1:
                    main_frame_paths.append("DUSTBIN")
                    main_frame_paths_logical.append("DUSTBIN")
                else:
                    # Pass actual numpy array for Collator
                    main_frame_paths.append(main_files[i])
                    # Store logical path for JSON logging
                    main_frame_paths_logical.append(f"{self.pickle_path}:{video_path}/{view}/frame_{i}")
            
            # Ref uses expert_path if available
            ref_source_path = self.expert_path if self.expert_path else self.pickle_path
            
            final_ref_paths = []
            final_ref_paths_logical = []
            for i in ref_context_steps:
                if i == -1:
                    final_ref_paths.append("DUSTBIN")
                    final_ref_paths_logical.append("DUSTBIN")
                else:
                    final_ref_paths.append(ref_files[i])
                    final_ref_paths_logical.append(f"{ref_source_path}:{ref_video_path}/{view}/frame_{i}")
            
            # Align paths
            num_align_frames = len(align_files)
            align_frame_paths_list = [align_files[i] for i in range(num_align_frames)]
            align_frame_paths_logical = [f"{ref_source_path}:{align_video_path}/{view}/frame_{i}" for i in range(num_align_frames)]
            main_align_frame_paths_list = align_frame_paths_list
            main_align_frame_paths_logical = align_frame_paths_logical
        else:
            # JSON mode: file paths serve both purposes
            main_frame_paths = [main_files[i] for i in main_context_steps]
            main_frame_paths_logical = main_frame_paths
            
            final_ref_paths = []
            for i in ref_context_steps:
                if i == -1:
                    final_ref_paths.append("DUSTBIN")
                else:
                    final_ref_paths.append(ref_files[i])
            final_ref_paths_logical = final_ref_paths
            
            align_frame_paths_list = align_frame_paths
            align_frame_paths_logical = align_frame_paths
            main_align_frame_paths_list = main_align_frame_paths
            main_align_frame_paths_logical = main_align_frame_paths
        
        # Construct Output
        data = {
            'chosen_steps': torch.from_numpy(main_steps),
            'ref_chosen_steps': torch.from_numpy(ref_steps),
            'seq_lens': torch.tensor(len(main_files)),
            'ref_seq_lens': torch.tensor(len(ref_files)),
            'seq_labels': torch.tensor(0), # Dummy
            'name': video_path,
            'ref_name': ref_video_path,
            'frame_paths': main_frame_paths,  # For Collator (actual data)
            'ref_frame_paths': final_ref_paths,
            'align_frame_paths': align_frame_paths_list, # FULL LIST
            'main_align_frame_paths': main_align_frame_paths_list, # FULL LIST
            'instruction': instruction,
            'initial_frame_path': initial_frame_path,
            'ref_instruction': ref_instruction,
            'dataset_name': dataset_name,
            'ref_initial_frame_path': ref_initial_frame_path,
            # Add logical paths for JSON logging (only in pickle mode)
            'frame_paths_str': main_frame_paths_logical if is_pickle_mode else None,
            'ref_frame_paths_str': final_ref_paths_logical if is_pickle_mode else None,
            'align_frame_paths_str': align_frame_paths_logical if is_pickle_mode else None,
            'main_align_frame_paths_str': main_align_frame_paths_logical if is_pickle_mode else None,
        }
        
        return data

    def __getitem__(self, index):
        while True:
            try:
                return self._get_item_impl(index)
            except Exception as e:
                # Get video_idx and path information for better error logging
                video_idx = index // len(self.views)
                video_path = self.video_paths[video_idx] if video_idx < len(self.video_paths) else "Unknown"
                
                # Add task description for rollout mode
                if self.is_rollout_mode and hasattr(self, 'task_descriptions'):
                    task_name = self.task_descriptions[video_path] if video_path < len(self.task_descriptions) else "Unknown"
                    print(f"Error loading index {index} (video_idx={video_idx}, task='{task_name}'): {e}. Retrying with random index...")
                else:
                    print(f"Error loading index {index} (video_idx={video_idx}, path='{video_path}'): {e}. Retrying with random index...")
                
                index = random.randint(0, len(self) - 1)

def get_transforms(mode='train'):
    transforms_list = []
    
    # Resize
    transforms_list.append(transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)))
    
    if mode == 'train':
        # Augmentation
        if CONFIG.AUGMENTATION.RANDOM_FLIP:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
        if CONFIG.AUGMENTATION.RANDOM_CROP:
            # TF used RandomCrop with min_scale 0.8. 
            # RandomResizedCrop is similar but also resizes back.
            # Or RandomCrop then Resize?
            # Let's use RandomResizedCrop for simplicity and robustness
            transforms_list.append(transforms.RandomResizedCrop(CONFIG.IMAGE_SIZE, scale=(0.8, 1.0)))
        
        # Color Jitter
        brightness = CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA if CONFIG.AUGMENTATION.BRIGHTNESS else 0
        contrast = 0.5 if CONFIG.AUGMENTATION.CONTRAST else 0 # TF used lower=0.5, upper=1.5 -> factor ~0.5
        hue = CONFIG.AUGMENTATION.HUE_MAX_DELTA if CONFIG.AUGMENTATION.HUE else 0
        saturation = 0.5 if CONFIG.AUGMENTATION.SATURATION else 0
        
        if brightness > 0 or contrast > 0 or hue > 0 or saturation > 0:
            transforms_list.append(transforms.ColorJitter(
                brightness=brightness, 
                contrast=contrast, 
                saturation=saturation, 
                hue=hue
            ))
            
    # ToTensor (Converts to [0, 1])
    transforms_list.append(transforms.ToTensor())
    
    # Normalize (Mean 0.5, Std 0.5 -> Map [0, 1] to [-1, 1])
    # TF code: (x - 0) / (255 - 0) * (1 - 0) + 0 -> [0, 1]
    # Then preprocess_sequence.NORMALIZE_MEAN_STDDEV: mean=0.5, std=0.5
    transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    return transforms.Compose(transforms_list)


class InfiniteDataLoader:
    def __init__(self, dataloader, sampler=None):
        self.dataloader = dataloader
        self.sampler = sampler
        self.epoch = 0
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self.epoch)
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
                self.epoch += 1
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

def worker_init_fn(worker_id):
    """Worker initialization function to ensure different seeds for each worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # Initialize decord bridge for video reading
    try:
        decord.bridge.set_bridge('native')  # Use numpy arrays
    except Exception as e:
        print(f"Warning: Failed to set decord bridge in worker {worker_id}: {e}")

def create_dataset(split, mode, batch_size=None, return_iterator=True, distributed=False, video_paths_json=None, processor=None):
    """Creates dataset iterator."""
    if not batch_size:
        batch_size = CONFIG.TRAIN.BATCH_SIZE if mode == 'train' else CONFIG.EVAL.BATCH_SIZE
        
    transform = get_transforms(mode)
    
    dataset = LiberoDataset(mode=mode, transform=transform, video_paths_json=video_paths_json, processor=processor)
    
    sampler = None
    use_weighted_sampler = (hasattr(dataset, 'weights') and len(dataset.weights) > 0 and mode == 'train')
    
    if distributed:
        if use_weighted_sampler and mode == 'train':
             # Use WeightedRandomSampler on each rank (only for training)
             weights = torch.DoubleTensor(dataset.weights)
             # Fix: Use a generator seeded by rank to ensure different data on each rank
             rank = torch.distributed.get_rank()
             g = torch.Generator()
             g.manual_seed(42 + rank)
             sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
             shuffle = False
        else:
             # Use DistributedSampler for both train and eval modes
             sampler = torch.utils.data.distributed.DistributedSampler(
                 dataset, 
                 shuffle=(mode == 'train')  # Shuffle for train, sequential for eval
             )
             shuffle = False
    else:
        if use_weighted_sampler:
             weights = torch.DoubleTensor(dataset.weights)
             sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
             shuffle = False
        else:
             sampler = None 
             shuffle = (mode == 'train')
    
    collator = TCCCollator(processor=processor, mode=mode)
    
    # Use fewer workers for eval to avoid initialization issues
    num_workers = 12 if mode == 'train' else 4

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=(mode == 'train'),  # Only drop last batch in training mode
        collate_fn=collator,
        worker_init_fn=worker_init_fn
    )
    
    if return_iterator and mode == 'train':
        return InfiniteDataLoader(dataloader, sampler=sampler)
    return dataloader

class LiberoVideoDataset(LiberoDataset):
    def __init__(self, mode='eval', transform=None, video_paths_json=None, processor=None, chunk_size=None):
        self.pickle_data = None
        self.expert_data = None
        self.is_rollout_mode = False
        
        is_pickle = isinstance(video_paths_json, str) and video_paths_json.endswith('.pkl')
        
        if is_pickle:
            # Initialize without paths first to avoid super().__init__ trying to load pkl as json
            super().__init__(mode, transform, None, processor)
            print(f"Loading video paths from pickle: {video_paths_json}")
            
            if 'rollout' in os.path.basename(video_paths_json):
                self.is_rollout_mode = True
                with open(video_paths_json, 'rb') as f:
                    self.pickle_data = pickle.load(f)
                
                # Load corresponding expert data
                expert_path = os.path.join(os.path.dirname(video_paths_json), 'libero_10-expert-video.pkl')
                if os.path.exists(expert_path):
                    print(f"Loading corresponding expert data from: {expert_path}")
                    with open(expert_path, 'rb') as f:
                        self.expert_data = pickle.load(f)
                
                rb = self.pickle_data['rollout_batch']
                self.task_descriptions = rb['task_descriptions']
                num_videos = len(self.task_descriptions)
                
                # Reorganize images for easier access: (N_chunks, B, Chunk_size, H, W, 3) -> (B, Total_frames, H, W, 3)
                # observation/full_image_list: [48, 16, 10, 256, 256, 3]
                full_imgs = rb['observation/full_image_list']
                n_chunks, b_size, c_size = full_imgs.shape[:3]
                # Permute to (batch, chunk, chunk_frames, H, W, 3) then reshape
                self.rollout_images = full_imgs.permute(1, 0, 2, 3, 4, 5).reshape(b_size, n_chunks * c_size, *full_imgs.shape[3:])
                
                # Assume wrist images are also there if available, else fallback
                if 'observation/wrist_image' in rb:
                    # observation/wrist_image: [48, 16, 256, 256, 3]
                    wrist_imgs = rb['observation/wrist_image']
                    # Repeat or pad to match full_image_list length if necessary
                    # For now, let's just use what we have, but it might be shorter (1/10th frequency)
                    self.rollout_wrist_images = wrist_imgs.permute(1, 0, 2, 3, 4)
                else:
                    self.rollout_wrist_images = None

                self.video_paths = [f"rollout_{i}" for i in range(num_videos)]
                self.video_weights = [1.0] * num_videos
                self.video_dataset_names = ["rollout_dataset"] * num_videos
            else:
                with open(video_paths_json, 'rb') as f:
                    self.pickle_data = pickle.load(f)
                self.video_paths = sorted(list(self.pickle_data.keys()))
                self.video_weights = [1.0] * len(self.video_paths)
                self.video_dataset_names = ["pickle_dataset"] * len(self.video_paths)
            
            # Re-initialize weights for sampler if needed (though usually for training)
            self.weights = []
            for w in self.video_weights:
                self.weights.append(w)
            
            self.views = ['images']  # Force single view for pickle/rollout mode
        else:
            super().__init__(mode, transform, video_paths_json, processor)
            self.views = ['images']  # Also force single view here for evaluation consistency
            
        self.chunk_size = chunk_size
        self.all_chunks = []
        
        # Stride for frames to sample from video
        stride = CONFIG.DATA.SAMPLE_ALL_STRIDE if hasattr(CONFIG.DATA, 'SAMPLE_ALL_STRIDE') else 1
        
        for video_idx, video_path in enumerate(self.video_paths):
            for view in self.views:
                if self.is_rollout_mode:
                    # In rollout mode, we use the pre-extracted tensors/arrays
                    if view == 'images':
                        files = self.rollout_images[video_idx]
                    else:
                        files = self.rollout_wrist_images[video_idx] if self.rollout_wrist_images is not None else self.rollout_images[video_idx]
                elif self.pickle_data is not None:
                    # In standard expert pickle mode, files is a numpy array (N, H, W, 3)
                    files = self.pickle_data[video_path].get(view, [])
                else:
                    view_path = os.path.join(video_path, view)
                    files = self._get_files(view_path)
                
                if files is None or len(files) == 0:
                    continue
                
                # Full list of sampled indices for this video
                indices = np.arange(0, len(files), stride)
                
                # We need to know candidate files to know how many candidate chunks there are
                # But to keep it simple and consistent with query chunks:
                # We just assume we extract candidate chunks paired with query chunks
                
                if self.chunk_size and self.chunk_size > 0:
                    for i in range(0, len(indices), self.chunk_size):
                        chunk_indices = indices[i : i + self.chunk_size]
                        self.all_chunks.append({
                            'video_idx': video_idx,
                            'view': view,
                            'indices': chunk_indices,
                            'chunk_idx': i // self.chunk_size,
                            'frame_offset': i,
                            'global_indices': chunk_indices 
                        })
                else:
                    self.all_chunks.append({
                        'video_idx': video_idx,
                        'view': view,
                        'indices': indices,
                        'chunk_idx': 0,
                        'frame_offset': 0,
                        'global_indices': indices
                    })

    def __len__(self):
        return len(self.all_chunks)

    def _get_item_impl(self, index):
        chunk_info = self.all_chunks[index]
        video_idx = chunk_info['video_idx']
        view = chunk_info['view']
        indices = chunk_info['indices']
        global_indices = chunk_info['global_indices']
        
        video_path = self.video_paths[video_idx]
        task_name = None
        
        if self.is_rollout_mode:
            task_name = self.task_descriptions[video_idx]
            if view == 'images':
                files = self.rollout_images[video_idx]
            else: # gripper_images
                # observation/wrist_image is (48, 256, 256, 3)
                # rollout_images is (480, 256, 256, 3)
                # We need to map 480 to 48
                wrist_imgs = self.rollout_wrist_images[video_idx] if self.rollout_wrist_images is not None else self.rollout_images[video_idx]
                
                # We will handle the indexing in the loop by creating a custom 'files' object or repeating
                # For simplicity, if frequencies don't match, we map current idx to wrist idx
                files = wrist_imgs # (48, 256, 256, 3)
        elif self.pickle_data is not None:
            files = self.pickle_data[video_path][view]
        else:
            view_path = os.path.join(video_path, view)
            files = self._get_files(view_path)
        
        # --- Query Frames (Dense Extraction) ---
        # Stride
        stride = CONFIG.DATA.SAMPLE_ALL_STRIDE if hasattr(CONFIG.DATA, 'SAMPLE_ALL_STRIDE') else 1
        
        # We need to handle context if NUM_STEPS > 1
        num_context = CONFIG.DATA.NUM_STEPS
        context_stride = self._get_stride(video_path)
        
        # indices is provided by chunk_info in __getitem__
        
        # Pad indices to chunk_size if necessary
        if self.chunk_size and self.chunk_size > 0:
            if len(indices) < self.chunk_size:
                pad_len = self.chunk_size - len(indices)
                indices = np.concatenate([indices, [indices[-1]] * pad_len])
                global_indices = np.concatenate([global_indices, [global_indices[-1]] * pad_len])
        
        frames = []
        for idx in indices:
            if idx == -1:
                # Dustbin Context: NUM_STEPS black frames
                frames.append(torch.zeros(num_context, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE))
                continue

            # Context window logic matching _get_context_steps
            # Range: [step - (num-1)*stride, ..., step + stride] with stride
            start = idx - (num_context - 1) * context_stride
            end = idx + context_stride
            
            ctx_indices = np.arange(start, end, context_stride)
            ctx_indices = np.clip(ctx_indices, 0, len(files) - 1)
            
            # Load context frames
            ctx_frames = []
            for ctx_idx in ctx_indices:
                if self.is_rollout_mode:
                    if view == 'images':
                        img_data = files[ctx_idx]
                    else:
                        # Map 480 frames to 48 (10x difference)
                        img_data = files[ctx_idx // 10]
                    
                    # Convert torch tensor to PIL
                    if torch.is_tensor(img_data):
                        img = Image.fromarray(img_data.cpu().numpy().astype(np.uint8)).convert('RGB')
                    else:
                        img = Image.fromarray(img_data.astype(np.uint8)).convert('RGB')
                        
                    if self.transform:
                        img = self.transform(img)
                    ctx_frames.append(img)
                elif self.pickle_data is not None:
                    # frames in pickle are numpy arrays
                    img = Image.fromarray(files[ctx_idx]).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    ctx_frames.append(img)
                else:
                    path = files[ctx_idx]
                    try:
                        with Image.open(path) as img:
                            img = img.convert('RGB')
                            if self.transform:
                                img = self.transform(img)
                            ctx_frames.append(img)
                    except Exception as e:
                        print(f"Error loading image {path}: {e}")
                        ctx_frames.append(torch.zeros(3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE))
            
            # Stack context frames: (NUM_STEPS, C, H, W)
            frames.append(torch.stack(ctx_frames))
        
        if not frames:
             frames = torch.zeros(1, num_context, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
        else:
            # Stack all steps: (T, NUM_STEPS, C, H, W)
            frames = torch.stack(frames)
            
        # --- Candidate Frames (Sampled like Training) ---
        if self.is_rollout_mode:
            candidate_video_path = task_name
            # Use expert data as candidate
            if self.expert_data is not None and task_name in self.expert_data:
                candidate_files = self.expert_data[task_name][view]
            else:
                candidate_files = files # Fallback to query
            candidate_pool = [task_name]
        elif self.pickle_data is not None:
            candidate_video_path = video_path
            candidate_files = files
            candidate_pool = [video_path]
        else:
            # Get Task Paths (Reference Videos)
            task_paths_file = os.path.join(video_path, 'task_paths.json')
            candidate_video_path = None
            
            if video_path not in self.task_paths_cache:
                if os.path.exists(task_paths_file):
                    try:
                        with open(task_paths_file, 'r') as f:
                            self.task_paths_cache[video_path] = json.load(f)
                    except:
                        self.task_paths_cache[video_path] = {}
                else:
                    self.task_paths_cache[video_path] = {}
                    
            task_paths = self.task_paths_cache[video_path]
            
            same_pool_keys = ["same", "100-95"]
            candidate_pool = []
            for key in same_pool_keys:
                if key in task_paths and task_paths[key]:
                    candidate_pool.extend(task_paths[key])
            
            if self.mode == 'train':
                if candidate_pool:
                    candidate_video_path = random.choice(candidate_pool)
                else:
                    # Fallback: use same video as candidate if no others available
                    candidate_video_path = video_path
            else:
                # Deterministic for eval
                if candidate_pool:
                    candidate_video_path = sorted(candidate_pool)[0]
                else:
                    candidate_video_path = video_path
                
            candidate_view_path = os.path.join(candidate_video_path, view)
            candidate_files = self._get_files(candidate_view_path)
            
            if not candidate_files:
                candidate_files = files # Fallback to query files
                candidate_video_path = video_path
            
        # Sampling
        # Use dense sampling for candidate frames as well (same as query frames)
        candidate_indices = np.arange(0, len(candidate_files), stride)
        # Prepend Dustbin to Candidate base list as well
        candidate_indices = np.concatenate([[-1], candidate_indices])
        
        # Chunk candidate indices if needed
        if self.chunk_size and self.chunk_size > 0:
            start_off = chunk_info['frame_offset']
            candidate_indices = candidate_indices[start_off : start_off + self.chunk_size]
            
            # Pad candidate_indices to chunk_size
            if len(candidate_indices) < self.chunk_size:
                pad_len = self.chunk_size - len(candidate_indices)
                if len(candidate_indices) > 0:
                    candidate_indices = np.concatenate([candidate_indices, [candidate_indices[-1]] * pad_len])
                else:
                    # Fallback for empty candidate
                    candidate_indices = np.zeros(self.chunk_size, dtype=int)
            
        candidate_frames_list = []
        for idx in candidate_indices:
            if idx == -1:
                candidate_frames_list.append(torch.zeros(num_context, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE))
                continue

            start = idx - (num_context - 1) * context_stride
            end = idx + context_stride
            
            ctx_indices = np.arange(start, end, context_stride)
            ctx_indices = np.clip(ctx_indices, 0, len(candidate_files) - 1)
            
            ctx_frames = []
            for ctx_idx in ctx_indices:
                if self.is_rollout_mode or self.pickle_data is not None:
                    # In pickle/rollout mode, candidate_files is a numpy array (expert data)
                    img = Image.fromarray(candidate_files[ctx_idx]).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    ctx_frames.append(img)
                else:
                    path = candidate_files[ctx_idx]
                    try:
                        with Image.open(path) as img:
                            img = img.convert('RGB')
                            if self.transform:
                                img = self.transform(img)
                            ctx_frames.append(img)
                    except Exception as e:
                        print(f"Error loading image {path}: {e}")
                        ctx_frames.append(torch.zeros(3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE))
            
            candidate_frames_list.append(torch.stack(ctx_frames))
            
        if not candidate_frames_list:
             candidate_frames = torch.zeros(1, num_context, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
        else:
            candidate_frames = torch.stack(candidate_frames_list)
        
        # Collect paths for Qwen
        # Flatten frame paths: [Step0_Ctx0, Step0_Ctx1, ..., Step1_Ctx0, ...]
        frame_paths = []
        for idx in indices:
            if idx == -1:
                for _ in range(num_context):
                    frame_paths.append("DUSTBIN")
                continue

            start = idx - (num_context - 1) * context_stride
            end = idx + context_stride
            ctx_indices = np.arange(start, end, context_stride)
            ctx_indices = np.clip(ctx_indices, 0, len(files) - 1)
            for ctx_idx in ctx_indices:
                if self.pickle_data is not None:
                    # Store numpy array for TCCCollator.load_imgs
                    frame_paths.append(files[ctx_idx])
                else:
                    frame_paths.append(files[ctx_idx])
                
        candidate_frame_paths = []
        for idx in candidate_indices:
            if idx == -1:
                for _ in range(num_context):
                    candidate_frame_paths.append("DUSTBIN")
                continue

            start = idx - (num_context - 1) * context_stride
            end = idx + context_stride
            ctx_indices = np.arange(start, end, context_stride)
            ctx_indices = np.clip(ctx_indices, 0, len(candidate_files) - 1)
            for ctx_idx in ctx_indices:
                if self.pickle_data is not None:
                    candidate_frame_paths.append(candidate_files[ctx_idx])
                else:
                    candidate_frame_paths.append(candidate_files[ctx_idx])

        # Instruction
        if self.is_rollout_mode:
            instruction = task_name
            candidate_instruction = task_name
        elif self.pickle_data is not None:
            instruction = video_path # Task name
            candidate_instruction = video_path
        else:
            instruction_path = os.path.join(video_path, 'instruction.txt')
            instruction = "Perform the task."
            if os.path.exists(instruction_path):
                try:
                    with open(instruction_path, 'r') as f:
                        instruction = f.read().strip()
                except:
                    pass
            
            # Candidate Instruction
            candidate_instruction_path = os.path.join(candidate_video_path, view) # Use view path for instruction if exists
            candidate_instruction_path = os.path.join(candidate_video_path, 'instruction.txt')
            candidate_instruction = "Perform the task."
            if os.path.exists(candidate_instruction_path):
                try:
                    with open(candidate_instruction_path, 'r') as f:
                        candidate_instruction = f.read().strip()
                except:
                    pass

        # --- Align Video (New for Alignment) ---
        # Use the same Reference/Candidate video as the source for the task abstraction prefix
        align_video_path = candidate_video_path
            
        if self.is_rollout_mode:
            align_files = self.expert_data[align_video_path][view] if self.expert_data and align_video_path in self.expert_data else files
        elif self.pickle_data is not None:
            align_files = self.pickle_data[align_video_path][view]
        else:
            align_view_path = os.path.join(align_video_path, view)
            align_files = self._get_files(align_view_path)
            if not align_files:
                align_files = files
            
        num_align_frames = getattr(CONFIG.TRAIN, 'NUM_ALIGN_FRAMES', 24)
        align_steps = np.linspace(0, len(align_files)-1, num_align_frames, dtype=int)
        
        if self.pickle_data is not None:
             align_frame_paths = [align_files[i] for i in align_steps]
        else:
             align_frame_paths = [align_files[i] for i in align_steps]

        return {
            'frames': frames,
            'candidate_frames': candidate_frames,
            'name': f"{video_path}/{view}",
            'candidate_name': f"{candidate_video_path}/{view}",
            'seq_lens': torch.tensor(len(frames)),
            'candidate_seq_lens': torch.tensor(len(candidate_frames)),
            'label': torch.tensor(0), # Dummy
            # For Reconstruction
            'video_name': f"{video_path}/{view}",
            'chunk_idx': chunk_info['chunk_idx'],
            'global_indices': torch.tensor(global_indices),
            'candidate_global_indices': torch.tensor(candidate_indices),
            'ref_chosen_steps': torch.tensor(candidate_indices), # Expose for TCCCollator dustbin detection
            # For Qwen
            'frame_paths': frame_paths,
            'ref_frame_paths': candidate_frame_paths, # Use ref_frame_paths key for collator compatibility
            'align_frame_paths': align_frame_paths,
            'instruction': instruction,
            'initial_frame_path': files[0],
            'ref_instruction': candidate_instruction,
            'ref_initial_frame_path': candidate_files[0]
        }

def create_one_epoch_dataset(split, mode, batch_size=1, return_iterator=True, video_paths_json=None, processor=None, chunk_size=None):
    transform = get_transforms(mode)
    dataset = LiberoVideoDataset(mode=mode, transform=transform, video_paths_json=video_paths_json, processor=processor, chunk_size=chunk_size)
    
    collator = None
    if processor is not None:
        collator = TCCCollator(processor=processor, mode=mode)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collator)
    
    if return_iterator:
        return iter(dataloader)
    return dataloader
    
    return dataloader
