# coding=utf-8
"""Datasets."""

import logging
import os
import json
import glob
import random
import zlib
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from config import CONFIG

from data_utils import get_image_info, get_video_info

from torch.nn.utils.rnn import pad_sequence

# Decord for efficient video reading
import decord

# Qwen VL utils for smart resize
from qwen_vl_utils.vision_process import smart_resize


def _discretize_joints(joint_vec, num_bins=192):
    """Discretize a normalized [-1, 1] joint vector into <|joint_N|> token string."""
    bins = np.linspace(-1, 1, num_bins + 1)[:-1]
    discretized = np.clip(np.digitize(joint_vec, bins=bins) - 1, 0, num_bins - 1).astype(int)
    return "".join(f"<|joint_{b}|>" for b in discretized)


def get_stratified_idxs(total_len, target_num, mode='train'):
    if total_len <= target_num or mode == 'eval':
        return np.linspace(0, total_len - 1, target_num, dtype=int)
    
    # Fixed endpoints; middle indices sampled randomly from the remaining range, then
    # sorted to preserve temporal order (random.sample does not return sorted output).
    idxs = [0, total_len - 1]
    middle_idxs = random.sample(range(1, total_len - 1), target_num - 2)
    idxs.extend(middle_idxs)
    idxs.sort()
    return np.array(idxs, dtype=int)

class IgnoreSample(Exception):
    """Skip this index without logging (e.g. missing cam_mapping entry, empty cam_list). Caught in __getitem__."""


# From 3Hz to 5Hz control frequency
# 低频控制数据集 (通常 <= 5Hz)
datasets_with_lower_frequency = [
    'rt1', 
    'berkeley_autolab_ur5', 
    'bridgev2',  # Bridge V2 通常是 5Hz
    'SSv2'
]

# From 15Hz to 30 Hz control frequency
# 高频控制数据集 (通常 >= 15Hz)
datasets_with_higher_frequency = [
    '10042',
    
]



class AlignmentCollator:
    def __init__(self, processor=None, mode='train'):
        self.processor = processor
        self.mode = mode
        if self.processor and hasattr(self.processor, 'tokenizer'):
            self.pad_token_id = self.processor.tokenizer.pad_token_id
        else:
            self.pad_token_id = 0
            
        print('self.pad_token_id:', self.pad_token_id)

        self.do_color_jitter = CONFIG.AUGMENTATION.get('BRIGHTNESS', True) or \
                               CONFIG.AUGMENTATION.get('CONTRAST', True)

        self.color_jitter = transforms.ColorJitter(
            brightness=CONFIG.AUGMENTATION.get('BRIGHTNESS_MAX_DELTA', 32.0 / 255) if CONFIG.AUGMENTATION.get('BRIGHTNESS', True) else 0,
            contrast=(CONFIG.AUGMENTATION.get('CONTRAST_LOWER', 0.5), CONFIG.AUGMENTATION.get('CONTRAST_UPPER', 1.5)) if CONFIG.AUGMENTATION.get('CONTRAST', True) else 0,
        )
        self.do_random_flip = CONFIG.AUGMENTATION.RANDOM_FLIP

        if self.mode != 'train':
            self.do_color_jitter = False
            self.do_random_flip = False

        # Shared index for cache-aware image loading (injected by evaluate_v2.py)
        self._ref_cache_index = None

    def augment_sequence(self, images, jitter_transform, random_flip):
        """
        Applies consistent geometric and independent photometric augmentation to a sequence of images.
        """
        augmented = []
        do_flip = False
        if random_flip and random.random() < 0.5:
            do_flip = True

        jitter_params_list = []

        for img in images:
            # Geometric: Consistent
            if do_flip:
                img = TF.hflip(img)

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

                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = TF.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = TF.adjust_contrast(img, contrast_factor)

                img_jitter_params = {
                    'fn_idx': fn_idx.tolist() if hasattr(fn_idx, 'tolist') else fn_idx,
                    'b': float(brightness_factor) if brightness_factor is not None else None,
                    'c': float(contrast_factor) if contrast_factor is not None else None,
                }

            jitter_params_list.append(img_jitter_params)
            augmented.append(img)

        params = {
            'flip': do_flip,
            'jitter': jitter_params_list
        }

        return augmented, params

    def load_imgs(self, paths):
        """
        Load images from various sources (file paths, arrays, video frames).
        Uses batch reading for video frames to maximize performance with decord.
        """
        # Phase 1: Scan and group video frames by video file
        video_frame_groups = {}  # {video_path: [(idx_in_paths, frame_idx, target_w, target_h), ...]}

        for idx, p in enumerate(paths):
            if isinstance(p, str) and p.startswith("video://"):
                # Parse video protocol: "video://<video_path>::<frame_idx>::<width>x<height>"
                try:
                    path_and_info = p[8:]
                    parts = path_and_info.rsplit('::', 2)

                    if len(parts) == 3:
                        # New protocol with size
                        video_path, frame_idx_str, size_str = parts
                        frame_idx = int(frame_idx_str)
                        width  = int(size_str.split('x')[0])
                        height = int(size_str.split('x')[1])
                    else:
                        # Fallback for old protocol without size
                        video_path, frame_idx_str = path_and_info.rsplit('::', 1)
                        frame_idx = int(frame_idx_str)
                        width, height = None, None  # Will use original size
                    
                    if video_path not in video_frame_groups:
                        video_frame_groups[video_path] = []
                    video_frame_groups[video_path].append((idx, frame_idx, width, height))
                except Exception as e:
                    print(f"[AlignmentCollator load_imgs] Error parsing video path {p}: {e}")
        
        # Phase 2: Batch read video frames
        video_frames_map = {}  # {(video_path, frame_idx): PIL.Image}
        
        for video_path, frame_list in video_frame_groups.items():
            try:
                vr = decord.VideoReader(video_path, num_threads=1)
                indices = [frame_idx for _, frame_idx, _, _ in frame_list]
                frames_np = vr.get_batch(indices).asnumpy()

                for i, (path_idx, frame_idx, target_w, target_h) in enumerate(frame_list):
                    frame = frames_np[i]  # (H, W, C)
                    img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')

                    if target_w is not None and target_h is not None:
                        img = img.resize((target_w, target_h), Image.BILINEAR)

                    video_frames_map[(video_path, frame_idx)] = img

                del vr
                    
            except Exception as e:
                print(f"[AlignmentCollator load_imgs] Error batch reading video {video_path}: {e}")
                # Fallback: create black images for failed frames
                for path_idx, frame_idx, _, _ in frame_list:
                    print(f"[AlignmentCollator load_imgs] Fallback to black image for {video_path} frame {frame_idx} (Phase 2)")
                    video_frames_map[(video_path, frame_idx)] = Image.new(
                        'RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0)
                    )
        
        # Phase 3: Assemble results in original order.
        imgs = []
        for idx, p in enumerate(paths):
            img = None
            if isinstance(p, str) and p.startswith("video://"):
                try:
                    path_and_info = p[8:]
                    parts = path_and_info.rsplit('::', 2)

                    if len(parts) == 3:
                        # size is only needed at load time (Phase 2), not for the map lookup key
                        video_path, frame_idx_str, _ = parts
                        frame_idx = int(frame_idx_str)
                    else:
                        video_path, frame_idx_str = path_and_info.rsplit('::', 1)
                        frame_idx = int(frame_idx_str)

                    img = video_frames_map.get((video_path, frame_idx))

                    if img is None:
                        print(f"[AlignmentCollator load_imgs] Fallback to black image for {video_path} frame {frame_idx} (Phase 3: img is None)")
                        img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
                except Exception as e:
                    print(f"[AlignmentCollator load_imgs] Error retrieving video frame {p}: {e}")
                    img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))
            
            elif isinstance(p, Image.Image):
                img = p.convert('RGB')
            
            elif isinstance(p, np.ndarray):
                img = Image.fromarray(p).convert('RGB')
            
            elif torch.is_tensor(p):
                # Accepts (C, H, W) or (H, W, C)
                if p.dim() == 3:
                    if p.shape[0] == 3:  # C, H, W
                        p = p.permute(1, 2, 0)
                    img = Image.fromarray(p.cpu().numpy().astype(np.uint8)).convert('RGB')
                else:
                    img = Image.new('RGB', (CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), (0, 0, 0))

            else:
                img = Image.open(p).convert('RGB')
            
            imgs.append(img)

        return imgs

    # -----------------------------------------------------------------------
    # Eval helpers
    # -----------------------------------------------------------------------
    def _resolve_eval_M(self, batch, target_M, c_size, chunk_len):
        """Validate target_M against the first batch sample's ref length (eval uses ref, not main).

        Returns the largest feasible M <= target_M. Logs a warning if M is
        downgraded. Falls back to M=1 if the video is too short for any M.
        """
        multipliers = [1, 2, 4]
        num_frames = len(batch[0]['ref_frame_paths']) // c_size
        for m in reversed(multipliers):
            if m > target_M:
                continue
            if num_frames >= m * chunk_len:
                if m != target_M:
                    print(f"[AlignmentCollator Eval] Warning: video too short for target M={target_M} "
                          f"(frames={num_frames}, need {target_M * chunk_len}). Falling back to M={m}.")
                return m
        print(f"[AlignmentCollator Eval] Warning: video too short even for M=1 "
              f"(frames={num_frames}, need {chunk_len}). Forcing M=1.")
        return 1

    # -----------------------------------------------------------------------
    # Phase 1: M-chunking
    # -----------------------------------------------------------------------
    def _apply_m_chunking(self, batch):
        """Select M, chunk_indices, cut_params."""
        batch_size = len(batch)
        c_size = CONFIG.DATA.NUM_STEPS
        chunk_len = getattr(CONFIG.TRAIN, 'NUM_FRAMES', 24)

        if self.mode == 'train':
            multiplier_probs = getattr(CONFIG.TRAIN, 'CHUNK_PROBS', [0.5, 0.25, 0.25])
            multipliers = [1, 2, 4]
            actually_valid_mults = []
            for m in multipliers:
                if batch_size % m != 0:
                    continue
                num_u = batch_size // m
                can_do_m = True
                for i in range(num_u):
                    if len(batch[i]['frame_paths']) // c_size < m * chunk_len:
                        can_do_m = False
                        break
                if can_do_m:
                    actually_valid_mults.append(m)
            if not actually_valid_mults:
                actually_valid_mults = [1]
            valid_probs = [multiplier_probs[multipliers.index(m)] for m in actually_valid_mults]
            p_sum = sum(valid_probs)
            valid_probs = [p / p_sum for p in valid_probs] if p_sum > 0 else [1.0 / len(actually_valid_mults)] * len(actually_valid_mults)
            M = random.choices(actually_valid_mults, weights=valid_probs)[0]
        else:
            multiplier_probs = getattr(CONFIG.EVAL, 'CHUNK_PROBS', [0.0, 0.0, 1.0])
            multipliers = [1, 2, 4]
            target_M, max_prob = 1, 0
            for i, prob in enumerate(multiplier_probs):
                if prob > max_prob:
                    max_prob = prob
                    target_M = multipliers[i]
            M = self._resolve_eval_M(batch, target_M, c_size, chunk_len)

        num_unique = 1 if self.mode == 'eval' else batch_size // M
        new_batch = []

        for i in range(num_unique):
            orig_data = batch[i]
            main_paths_all = orig_data['frame_paths']
            # Train: chunk budget from main. Eval: from ref (paired with main chunk indices).
            if self.mode == 'eval':
                S = len(orig_data['ref_frame_paths']) // c_size
            else:
                S = len(main_paths_all) // c_size
            indices = np.linspace(0, S - 1, M * chunk_len, dtype=int)

            for m in range(M):
                chunk_data = orig_data.copy()

                chunk_indices = indices[m * chunk_len : (m + 1) * chunk_len]

                # Student frame paths
                chunk_data['frame_paths'] = [
                    f for idx in chunk_indices
                    for f in main_paths_all[idx * c_size : (idx + 1) * c_size]]
                chunk_data['ref_frame_paths'] = [
                    f for idx in chunk_indices
                    for f in orig_data['ref_frame_paths'][idx * c_size : (idx + 1) * c_size]]

                # Logical paths (retained for downstream compatibility; unused, always None)
                if orig_data.get('frame_paths_str') is not None:
                    chunk_data['frame_paths_str'] = [
                        f for idx in chunk_indices
                        for f in orig_data['frame_paths_str'][idx * c_size : (idx + 1) * c_size]]
                if orig_data.get('ref_frame_paths_str') is not None:
                    chunk_data['ref_frame_paths_str'] = [
                        f for idx in chunk_indices
                        for f in orig_data['ref_frame_paths_str'][idx * c_size : (idx + 1) * c_size]]

                if 'chosen_steps' in orig_data:
                    chunk_data['chosen_steps'] = orig_data['chosen_steps'][chunk_indices]
                if 'ref_chosen_steps' in orig_data:
                    chunk_data['ref_chosen_steps'] = orig_data['ref_chosen_steps'][chunk_indices]

                # Joint per-step lists (one entry per anchor step)
                if orig_data.get('main_joint_per_step') is not None:
                    chunk_data['main_joint_per_step'] = [
                        orig_data['main_joint_per_step'][idx] for idx in chunk_indices
                    ]
                if orig_data.get('ref_joint_per_step') is not None:
                    chunk_data['ref_joint_per_step'] = [
                        orig_data['ref_joint_per_step'][idx] for idx in chunk_indices
                    ]

                chunk_data['group_id'] = i
                chunk_data['chunk_id'] = m
                chunk_data['multiplier'] = M
                new_batch.append(chunk_data)

        return new_batch

    # -----------------------------------------------------------------------
    # Phase 2: Per-sample preparation (align downsample)
    # -----------------------------------------------------------------------
    def _prepare_per_sample(self, batch):
        """Compute align_paths after downsample; cache in batch[i]."""
        for i in range(len(batch)):
            # Per-sample target: dataset pre-builds combined align (48 frames) and stores the
            # expected length as 'num_align_frames'; fall back to config for legacy samples.
            target_num = batch[i].get('num_align_frames',
                                      getattr(CONFIG.TRAIN, 'NUM_ALIGN_FRAMES', 24))
            align_paths = list(batch[i].get('align_frame_paths', []))

            if len(align_paths) > target_num:
                idxs = np.linspace(0, len(align_paths) - 1, target_num, dtype=int)
                align_paths = [align_paths[k] for k in idxs]

            batch[i]['_cached_align_paths'] = align_paths

        return batch

    # -----------------------------------------------------------------------
    # Phase 3 / 4 helpers: _pack_sequence and _build_qwen_input
    # -----------------------------------------------------------------------
    def _pack_sequence(self, sink, video_imgs, chunk_imgs, is_ref_row, seq_len_val,
                       group_id, chunk_id, frame_paths, ref_frame_paths, name, ref_name,
                       dataset_name, aug_params, ref_aug_params,
                       update_metadata=True, joint_per_chunk=None):
        """Run Qwen processor for one row and write results to sink.

        update_metadata=False skips populating num_mains/refs, seq_lens, masks, etc.
        joint_per_chunk: list of np.ndarray [D_joint] or None, one per chunk (anchor step).
            When provided, joint tokens are injected BEFORE each chunk video.
        """
        # Split align video into main-half and ref-half so each is presented as a separate
        # Qwen video entry.  Each half comes from one source video and is internally
        # size-consistent; they are processed independently by get_video_info.
        _nh = len(video_imgs) // 2
        align_half_main = video_imgs[:_nh]
        align_half_ref  = video_imgs[_nh:]

        content = [
            {"type": "video", "video": align_half_main},
            {"type": "video", "video": align_half_ref},
            {"type": "text", "text": "<|fim_pad|>"},
        ]
        for ci, chunk in enumerate(chunk_imgs):
            # Joint tokens go BEFORE the chunk video (state → observation conditioning order)
            if (joint_per_chunk is not None
                    and ci < len(joint_per_chunk)
                    and joint_per_chunk[ci] is not None):
                joint_str = _discretize_joints(joint_per_chunk[ci], CONFIG.JOINTS.NUM_BINS)
                content.append({"type": "text", "text": joint_str})
            content.append({"type": "video", "video": chunk})
            content.append({"type": "text", "text": "<|file_sep|>"})

        num_c = len(chunk_imgs)

        if update_metadata:
            if is_ref_row:
                sink['num_mains'].append(0)
                sink['num_refs'].append(num_c)
                sink['ref_seq_lens'].extend([seq_len_val] * num_c)
            else:
                sink['num_mains'].append(num_c)
                sink['num_refs'].append(0)
                sink['main_seq_lens'].extend([seq_len_val] * num_c)

            sink['group_ids'].append(group_id)
            sink['chunk_ids'].append(chunk_id)
            sink['frame_paths'].append(frame_paths)
            sink['ref_frame_paths'].append(ref_frame_paths)
            sink['names'].append(name)
            sink['ref_names'].append(ref_name)
            sink['dataset_names'].append(dataset_name)
            sink['aug_params'].append(aug_params)
            sink['ref_aug_params'].append(ref_aug_params)

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process each align half independently — each is internally size-consistent.
        _pair1, video_kwargs = get_video_info(
            align_half_main, min_pixels=224 * 224, max_pixels=256 * 256,
            width=None, height=None, fps=2.0, image_patch_size=16, return_video_metadata=True)
        _pair2, _ = get_video_info(
            align_half_ref, min_pixels=224 * 224, max_pixels=256 * 256,
            width=None, height=None, fps=2.0, image_patch_size=16, return_video_metadata=True)
        align_tensors = [_pair1[0], _pair2[0]]
        align_metas   = [_pair1[1], _pair2[1]]

        group_video_tensors, group_video_metadatas = [], []
        for chunk in chunk_imgs:
            gv_pair, _ = get_video_info(
                chunk, min_pixels=224 * 224, max_pixels=256 * 256,
                width=None, height=None, fps=2.0, image_patch_size=16, return_video_metadata=True)
            group_video_tensors.append(gv_pair[0])
            group_video_metadatas.append(gv_pair[1])

        inputs = self.processor(
            text=[text], images=None,
            videos=align_tensors + group_video_tensors,
            padding=False, return_tensors="pt",
            video_metadata=align_metas + group_video_metadatas,
            **video_kwargs)

        sink['input_ids'].append(inputs['input_ids'][0])
        if 'pixel_values'        in inputs: sink['pixel_values'].append(inputs['pixel_values'])
        if 'image_grid_thw'      in inputs: sink['image_grid_thw'].append(inputs['image_grid_thw'])
        if 'pixel_values_videos' in inputs: sink['pixel_values_videos'].append(inputs['pixel_values_videos'])
        if 'video_grid_thw'      in inputs: sink['video_grid_thw'].append(inputs['video_grid_thw'])

    def _prepare_images(self, batch, context_size,
                        frame_paths_key, ref_frame_paths_key,
                        do_augment, align_paths_key='_cached_align_paths'):
        """Load images, augment, split into align/chunks. Returns `prepared` list.

        ref_frame_paths_key=None skips loading ref images entirely (cache-hit path).
        """
        prepared = []
        for i in range(len(batch)):
            main_paths = batch[i].get(frame_paths_key, [])
            ref_paths  = batch[i].get(ref_frame_paths_key, []) if ref_frame_paths_key else []
            align      = batch[i].get(align_paths_key, batch[i]['_cached_align_paths'])

            if not main_paths:
                prepared.append(None)
                continue

            main_all = self.load_imgs(align + main_paths)
            ref_all  = self.load_imgs(align + ref_paths) if ref_paths else []

            if do_augment:
                jitter = self.color_jitter if self.do_color_jitter else None
                main_all, params_main = self.augment_sequence(main_all, jitter, self.do_random_flip)
                if ref_all:
                    ref_all, params_ref = self.augment_sequence(ref_all, jitter, self.do_random_flip)
                else:
                    params_ref = {'flip': False, 'jitter': []}
            else:
                params_main = {'flip': False, 'jitter': [None] * len(main_all)}
                params_ref  = {'flip': False, 'jitter': [None] * len(ref_all)}

            a_len = len(align)
            main_align = main_all[:a_len];  main_flat = main_all[a_len:]
            ref_align  = ref_all[:a_len]  if ref_all else main_align   # fallback to main_align when ref not loaded
            ref_flat   = ref_all[a_len:]  if ref_all else []

            main_chunks = [main_flat[j:j+context_size] for j in range(0, len(main_flat), context_size)]
            ref_chunks  = [ref_flat[j:j+context_size]  for j in range(0, len(ref_flat),  context_size)]

            prepared.append({
                'main_align':   main_align,   'main_chunks': main_chunks,
                'ref_align':    ref_align,    'ref_chunks':  ref_chunks,
                'params_main':  params_main,  'params_ref':  params_ref,
                'sl':  batch[i].get('seq_lens', torch.tensor(0)),
                'rsl': batch[i].get('ref_seq_lens', batch[i].get('candidate_seq_lens', torch.tensor(0))),
                'has_ref': bool(ref_chunks),
                'main_joints': batch[i].get('main_joint_per_step'),
                'ref_joints':  batch[i].get('ref_joint_per_step'),
            })

        return prepared

    def _build_qwen_input(self, batch, context_size,
                          frame_paths_key, ref_frame_paths_key,
                          do_augment, update_metadata,
                          align_paths_key='_cached_align_paths',
                          prepared=None):
        """Load images (unless `prepared` is given), optionally augment, and run Pass 1 + Pass 2.

        Rows are emitted in [all_mains, all_refs] order — required by utils.py's
        half-split assumption (group_ids[:half_len] searches only the main half).

        frame_paths_key / ref_frame_paths_key select which frame paths to use.
        align_paths_key selects which cached align paths to use.
        do_augment=False produces clean images.
        update_metadata=False skips metadata accumulation.
        prepared: if provided, skip the image-loading preparation pass (reuse loaded images).
        """
        sink = {
            'input_ids': [], 'pixel_values': [], 'image_grid_thw': [],
            'pixel_values_videos': [], 'video_grid_thw': [],
            'num_mains': [], 'num_refs': [],
            'group_ids': [], 'chunk_ids': [],
            'main_seq_lens': [], 'ref_seq_lens': [],
            'frame_paths': [], 'ref_frame_paths': [],
            'names': [], 'ref_names': [], 'dataset_names': [],
            'aug_params': [], 'ref_aug_params': [],
        }

        if prepared is None:
            prepared = self._prepare_images(
                batch, context_size,
                frame_paths_key=frame_paths_key,
                ref_frame_paths_key=ref_frame_paths_key,
                do_augment=do_augment,
                align_paths_key=align_paths_key)

        # ---- Pass 1: all Main rows → first half of sink ----
        for i, p in enumerate(prepared):
            if p is None:
                continue
            self._pack_sequence(
                sink, p['ref_align'], p['main_chunks'],
                is_ref_row=False, seq_len_val=p['sl'],
                group_id=batch[i].get('group_id', i),
                chunk_id=batch[i].get('chunk_id', 0),
                frame_paths=batch[i].get('frame_paths', []),
                ref_frame_paths=batch[i].get('ref_frame_paths', []),
                name=batch[i].get('name', 'unknown'),
                ref_name=batch[i].get('ref_name', 'unknown'),
                dataset_name=batch[i].get('dataset_name', 'unknown'),
                aug_params=p['params_main'], ref_aug_params=p['params_ref'],
                update_metadata=update_metadata,
                joint_per_chunk=p['main_joints'])

        # ---- Pass 2: all Ref rows → second half of sink ----
        for i, p in enumerate(prepared):
            if p is None or not p['has_ref'] or ref_frame_paths_key is None:
                continue
            self._pack_sequence(
                sink, p['main_align'], p['ref_chunks'],
                is_ref_row=True, seq_len_val=p['rsl'],
                group_id=batch[i].get('group_id', i),
                chunk_id=batch[i].get('chunk_id', 0),
                frame_paths=batch[i].get('frame_paths', []),
                ref_frame_paths=batch[i].get('ref_frame_paths', []),
                name=batch[i].get('name', 'unknown'),
                ref_name=batch[i].get('ref_name', 'unknown'),
                dataset_name=batch[i].get('dataset_name', 'unknown'),
                aug_params=p['params_main'], ref_aug_params=p['params_ref'],
                update_metadata=update_metadata,
                joint_per_chunk=p['ref_joints'])

        return sink

    # -----------------------------------------------------------------------
    # Phase 5: Assemble qwen_input dict from sink
    # -----------------------------------------------------------------------
    def _assemble_qwen_dict(self, sink, shared_metadata=None):
        """Pad/cat pixel tensors and build the final qwen_input dict.

        shared_metadata: when provided, metadata tensors are copied from the
        provided qwen_input instead of being built from sink.
        """
        if not sink['input_ids']:
            return None

        input_ids = pad_sequence(sink['input_ids'], batch_first=True, padding_value=self.pad_token_id)
        d = {
            'input_ids':           input_ids,
            'attention_mask':      (input_ids != self.pad_token_id),
            'pixel_values':        torch.cat(sink['pixel_values'],        dim=0) if sink['pixel_values']        else None,
            'image_grid_thw':      torch.cat(sink['image_grid_thw'],      dim=0) if sink['image_grid_thw']      else None,
            'pixel_values_videos': torch.cat(sink['pixel_values_videos'], dim=0) if sink['pixel_values_videos'] else None,
            'video_grid_thw':      torch.cat(sink['video_grid_thw'],      dim=0) if sink['video_grid_thw']      else None,
        }

        if shared_metadata is not None:
            # Teacher: reuse all metadata tensors from student
            for k in ('seq_lens', 'num_mains', 'num_refs', 'group_ids', 'chunk_ids',
                      'cls_token_id', 'align_end_token_id'):
                d[k] = shared_metadata[k]
        else:
            # Student: build metadata from sink arrays
            all_seq_lens = sink['main_seq_lens'] + sink['ref_seq_lens']
            d['seq_lens']   = torch.stack(all_seq_lens) if all_seq_lens else None
            d['num_mains']  = torch.tensor(sink['num_mains'],  dtype=torch.long)
            d['num_refs']   = torch.tensor(sink['num_refs'],   dtype=torch.long)
            d['group_ids']  = torch.tensor(sink['group_ids'],  dtype=torch.long)
            d['chunk_ids']  = torch.tensor(sink['chunk_ids'],  dtype=torch.long)
            d['cls_token_id']        = torch.tensor(CONFIG.SPECIAL_TOKENS.CLS_TOKEN_ID,        dtype=torch.long)
            d['align_end_token_id']  = torch.tensor(CONFIG.SPECIAL_TOKENS.ALIGN_END_TOKEN_ID,  dtype=torch.long)

        return d

    # -----------------------------------------------------------------------
    # Simple collate (no nested function in __call__)
    # -----------------------------------------------------------------------
    def _simple_collate(self, data_list):
        """Stack/list-collate a list of sample dicts. Used after raw batch and after M-chunking."""
        if not data_list:
            return {}
        elem = data_list[0]
        res = {}
        list_keys = [
            'frame_paths', 'ref_frame_paths',
            'name', 'ref_name', 'align_frame_paths', 'main_align_frame_paths',
            'frame_paths_str', 'ref_frame_paths_str',
            'align_frame_paths_str', 'main_align_frame_paths_str',
        ]
        for key in elem:
            if key.startswith('_'):
                continue
            if key in list_keys:
                res[key] = [d[key] for d in data_list if key in d]
            elif isinstance(elem[key], torch.Tensor):
                try:
                    res[key] = torch.stack([d[key] for d in data_list])
                except RuntimeError:
                    res[key] = [d[key] for d in data_list]
            else:
                res[key] = [d[key] for d in data_list]
        return res

    # -----------------------------------------------------------------------
    # __call__: orchestrator
    # -----------------------------------------------------------------------
    def __call__(self, batch):
        if self.mode == 'train':
            batch = sorted(batch, key=lambda d: len(d.get('frame_paths', [])), reverse=True)

        collated_batch = self._simple_collate(batch)

        if self.processor is None:
            return collated_batch

        # --- Phase 1: M-chunking ---
        batch = self._apply_m_chunking(batch)
        collated_batch = self._simple_collate(batch)

        # --- Phase 2: Per-sample preparation (cut, align downsample, mask) ---
        batch = self._prepare_per_sample(batch)

        context_size = CONFIG.DATA.NUM_STEPS

        # --- Check whether all refs are cached (via shared index, no CUDA access) ---
        # _ref_cache_index is a Manager().dict() injected by evaluate_v2.py; keys are
        # int hashes (zlib.crc32) of ref_frame_paths — identical to RefEmbeddingCache keys.
        skip_ref = False
        if self._ref_cache_index is not None and self.mode == 'eval':
            ref_paths_list = [b['ref_frame_paths'] for b in batch]
            skip_ref = all(
                zlib.crc32('\0'.join(rp).encode()) in self._ref_cache_index
                for rp in ref_paths_list
            )

        if skip_ref:
            # --- Cache hit: only load main, skip ref image loading entirely ---
            prepared = self._prepare_images(
                batch, context_size,
                frame_paths_key='frame_paths',
                ref_frame_paths_key=None,       # skip ref load
                do_augment=True,
                align_paths_key='_cached_align_paths')
            sink_main = self._build_qwen_input(
                batch, context_size,
                frame_paths_key='frame_paths',
                ref_frame_paths_key=None,
                do_augment=True,
                update_metadata=True,
                align_paths_key='_cached_align_paths',
                prepared=prepared)
            # Assemble main_only BEFORE augmenting sink with ref metadata
            # (Qwen uses num_mains/num_refs to count CLS tokens; main_only has B rows only)
            qwen_main_only = self._assemble_qwen_dict(sink_main)

            # Symmetrically fill ref metadata from main (structure is identical in eval)
            # Conditional fields (Pass 2 / ref row)
            sink_main['num_refs']       = sink_main['num_mains'][:]
            sink_main['ref_seq_lens']   = sink_main['main_seq_lens'][:]
            # Unconditional fields (_pack_sequence appends these for every row)
            sink_main['group_ids']      = sink_main['group_ids']      + sink_main['group_ids'][:]
            sink_main['chunk_ids']      = sink_main['chunk_ids']      + sink_main['chunk_ids'][:]
            sink_main['frame_paths']    = sink_main['frame_paths']    + sink_main['frame_paths'][:]
            sink_main['ref_frame_paths']= sink_main['ref_frame_paths']+ sink_main['ref_frame_paths'][:]
            sink_main['names']          = sink_main['names']          + sink_main['names'][:]
            sink_main['ref_names']      = sink_main['ref_names']      + sink_main['ref_names'][:]
            sink_main['dataset_names']  = sink_main['dataset_names']  + sink_main['dataset_names'][:]
            sink_main['aug_params']     = sink_main['aug_params']     + sink_main['aug_params'][:]
            sink_main['ref_aug_params'] = sink_main['ref_aug_params'] + sink_main['ref_aug_params'][:]
            qwen_paired = self._assemble_qwen_dict(sink_main)

            collated_batch['qwen_input']           = qwen_paired  # Qwen forward (main only)
            collated_batch['qwen_input_paired']    = qwen_paired      # merge logic (full 2B metadata)
            collated_batch['qwen_input_main_only'] = qwen_main_only
            student_sink = sink_main   # for metadata below (ref_frame_paths already correct: Pass 1 only)
        else:
            # --- Cache miss (or cache not active): load main + ref, reuse prepared ---
            prepared = self._prepare_images(
                batch, context_size,
                frame_paths_key='frame_paths',
                ref_frame_paths_key='ref_frame_paths',
                do_augment=True,
                align_paths_key='_cached_align_paths')

            # Paired: Pass 1 (main) + Pass 2 (ref)
            student_sink = self._build_qwen_input(
                batch, context_size,
                frame_paths_key='frame_paths',
                ref_frame_paths_key='ref_frame_paths',
                do_augment=True,
                update_metadata=True,
                align_paths_key='_cached_align_paths',
                prepared=prepared)
            qwen_paired = self._assemble_qwen_dict(student_sink)

            # Main-only: reuse prepared (no extra image load)
            sink_main_only = self._build_qwen_input(
                batch, context_size,
                frame_paths_key='frame_paths',
                ref_frame_paths_key=None,
                do_augment=True,
                update_metadata=True,
                align_paths_key='_cached_align_paths',
                prepared=prepared)
            collated_batch['qwen_input']           = qwen_paired
            collated_batch['qwen_input_paired']    = qwen_paired
            collated_batch['qwen_input_main_only'] = self._assemble_qwen_dict(sink_main_only)

        # Row-level logging metadata (aligned with student Qwen rows)
        collated_batch['aug_params']      = student_sink['aug_params']
        collated_batch['ref_aug_params']  = student_sink['ref_aug_params']
        collated_batch['frame_paths']     = student_sink['frame_paths']
        collated_batch['ref_frame_paths'] = student_sink['ref_frame_paths']
        collated_batch['name']            = student_sink['names']
        collated_batch['ref_name']        = student_sink['ref_names']
        collated_batch['dataset_name']    = student_sink['dataset_names']

        return collated_batch

class AlignmentDataset(Dataset):
    def __init__(self, mode='train', transform=None, video_paths_json=None, processor=None):
        self.mode = mode
        self.transform = transform
        self.processor = processor
        self.debug_step = 0

        self.video_paths = []
        self.video_dataset_names = []
        self.video_weights = []
        self.video_dataset_types = []  # Store dataset type for each video (for path transforms)

        # Supports list or comma-separated string
        if video_paths_json:
            if isinstance(video_paths_json, str):
                paths = [p.strip() for p in video_paths_json.split(',')]
            elif isinstance(video_paths_json, list):
                paths = video_paths_json
            else:
                paths = []
            
            def get_dataset_name(json_path):
                # e.g. /path/to/berkeley_autolab_ur5_video_paths.json -> berkeley_autolab_ur5
                base = os.path.basename(json_path)
                name = base.replace('_video_paths.json', '').replace('.json', '')
                return name

            # First pass: Count totals to calculate probabilities
            dataset_counts = {}
            dataset_weights = {} 
            dataset_types = {}  # name -> type (直接使用 name 作为 type)
            temp_paths = {} # name -> list of paths

            for path in paths:
                if os.path.exists(path):
                    print(f"Loading video paths from: {path}")
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            ds_name = get_dataset_name(path)
                            # Parse video paths when dataset uses segmented format (config: SEGMENTED_PATH_DATASETS)
                            parsed_paths = [self._parse_video_path(p, ds_name) for p in data]
                            temp_paths[ds_name] = parsed_paths
                            dataset_counts[ds_name] = len(data)
                            dataset_types[ds_name] = ds_name  # dataset_name 即为 dataset_type
                            ds_weight = CONFIG.DATA.DATASET_WEIGHTS.get(ds_name, 1.0)
                            dataset_weights[ds_name] = ds_weight
                        else:
                            print(f"Warning: {path} content is not a list.")
                else:
                    print(f"Warning: Provided video_paths file not found: {path}")
            
            total_weighted_count = sum(c * w for k, c in dataset_counts.items() for w in [dataset_weights[k]])
            print("\n--- Dataset Sampling Probabilities ---")
            for name, count in dataset_counts.items():
                w = dataset_weights[name]
                prob = (count * w) / total_weighted_count if total_weighted_count > 0 else 0
                print(f"Dataset: {name}, Count: {count}, Weight: {w}, Sampling Prob: {prob:.4f}")
            print("--------------------------------------\n")

            for name, paths_list in temp_paths.items():
                ds_type = dataset_types[name]
                self.video_paths.extend(paths_list)
                self.video_dataset_names.extend([name] * len(paths_list))
                self.video_dataset_types.extend([ds_type] * len(paths_list))  # 存储类型
                self.video_weights.extend([dataset_weights[name]] * len(paths_list))

        
        if not self.video_paths:
            print(f"WARNING: No video paths loaded.")
        
        # No path validation is performed in any mode
        print(f"Total video paths loaded: {len(self.video_paths)}")

        self.views = ['images']

        # Prepare weights for WeightedRandomSampler (per item in __len__)
        # Each video has len(self.views) items.
        # weight list must match __len__ size which is len(videos) * len(views)
        self.weights = []
        for w in self.video_weights:
            self.weights.extend([w] * len(self.views))

        # Per-dataset index for negative instruction candidate sampling: {dataset_name: [video_idx, ...]}
        self._dataset_video_indices: dict = {}
        for idx, dname in enumerate(self.video_dataset_names):
            self._dataset_video_indices.setdefault(dname, []).append(idx)

        # task_path -> [camera names]; loaded from {CAM_MAPPING_DIR}/{dataset_name}_cam_mapping.json
        self._cam_mapping_by_dataset: dict = {}
        _cam_dir = getattr(CONFIG.DATA, 'CAM_MAPPING_DIR', '') or ''
        if _cam_dir and getattr(CONFIG.DATA, 'USE_CAM_MAPPING', True):
            for dname in set(self.video_dataset_names):
                _mp = os.path.join(_cam_dir, f'{dname}_cam_mapping.json')
                if os.path.isfile(_mp):
                    try:
                        with open(_mp, 'r', encoding='utf-8') as _f:
                            _raw = json.load(_f)
                        # Pre-build normpath index for O(1) task-key lookup
                        self._cam_mapping_by_dataset[dname] = {
                            'raw': _raw,
                            'index': {os.path.normpath(k): k for k in _raw},
                        }
                        print(f"  Loaded cam mapping {_mp} ({len(_raw)} task keys)")
                    except Exception as _e:
                        logging.warning(f"Failed to load cam mapping {_mp}: {_e}")
                else:
                    logging.warning(
                        f"CAM_MAPPING_DIR is set but file missing for dataset {dname!r}: {_mp} "
                        f"(_load_video_data_from_json will return None for this dataset)")

        self._joint_action_mapping_cache = {}
        if getattr(CONFIG.JOINTS, 'USE_JOINTS', False):
            _jam_dir = CONFIG.JOINTS.JOINT_ACTION_MAPPING_DIR
            if _jam_dir and os.path.isdir(_jam_dir):
                for fname in os.listdir(_jam_dir):
                    if fname.endswith('_joint_action_mapping.json'):
                        ds = fname.replace('_joint_action_mapping.json', '')
                        with open(os.path.join(_jam_dir, fname), 'r') as f:
                            self._joint_action_mapping_cache[ds] = json.load(f)
                logging.info(f"[joints] Loaded mappings for {len(self._joint_action_mapping_cache)} datasets")
            else:
                logging.warning(
                    f"[joints] USE_JOINTS=True but JOINT_ACTION_MAPPING_DIR={_jam_dir!r} "
                    f"is missing or not a directory"
                )

        print(f"AlignmentDataset initialized with {len(self.video_paths)} videos and views: {self.views}")
        
    def __len__(self):
        return len(self.video_paths) * len(self.views)
    
    def _parse_video_path(self, path_str, dataset_name=None):
        """
        解析视频路径。是否按分段格式解析由 config 中的数据集名称决定，而非路径中是否含 ':'。
        分段格式 (仅当 dataset_name in CONFIG.DATA.SEGMENTED_PATH_DATASETS 时): "/path/to/video:segment_id:start-end"
        否则一律视为旧格式: "/path/to/video"

        Args:
            path_str: 路径字符串
            dataset_name: 数据集名称（来自 JSON 文件名等），用于判断是否解析分段格式

        Returns: dict with keys:
            - video_dir: 视频目录路径
            - segment_id: 分段ID (None for old format)
            - frame_start: 起始帧 (None for old format)
            - frame_end: 结束帧 (None for old format)
            - original_path: 原始路径字符串
        """
        segmented = (
            dataset_name is not None
            and hasattr(CONFIG.DATA, 'SEGMENTED_PATH_DATASETS')
            and dataset_name in CONFIG.DATA.SEGMENTED_PATH_DATASETS
        )
        if not segmented:
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }

        # 分段格式：解析 "path:segment_id:start-end"
        if ':' not in path_str:
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }
        parts = path_str.rsplit(':', 2)
        if len(parts) != 3:
            print(f"Warning: Path format incorrect '{path_str}'. Treating as old format.")
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }
        video_dir, segment_id_str, frame_range_str = parts
        try:
            segment_id = int(segment_id_str)
            if '-' in frame_range_str:
                frame_start_str, frame_end_str = frame_range_str.split('-')
                frame_start = int(frame_start_str)
                frame_end = int(frame_end_str)
            else:
                print(f"Warning: Failed to parse frame range '{frame_range_str}'. Treating as old format.")
                frame_start = None
                frame_end = None
            return {
                'video_dir': video_dir,
                'segment_id': segment_id,
                'frame_start': frame_start,
                'frame_end': frame_end,
                'original_path': path_str
            }
        except (ValueError, AttributeError) as e:
            print(f"Warning: Failed to parse video path '{path_str}': {e}. Treating as old format.")
            return {
                'video_dir': path_str,
                'segment_id': None,
                'frame_start': None,
                'frame_end': None,
                'original_path': path_str
            }
    
    def _apply_task_paths_transform(self, task_paths_file, dataset_type):
        """
        根据数据集类型对 task_paths_file 路径进行转换
        
        Args:
            task_paths_file: str, task_paths.json 的原始路径
            dataset_type: str, 数据集类型（从 video_dataset_types 获取）
        
        Returns:
            str: 转换后的路径
        """
        # 检查是否有该数据集的转换规则
        if dataset_type not in CONFIG.DATA.TASK_PATHS_TRANSFORMS:
            return task_paths_file
        
        transform_rules = CONFIG.DATA.TASK_PATHS_TRANSFORMS[dataset_type]
        transformed_path = task_paths_file
        
        # 应用所有转换规则（按顺序）
        for old_prefix, new_prefix in transform_rules.items():
            if transformed_path.startswith(old_prefix):
                transformed_path = transformed_path.replace(old_prefix, new_prefix, 1)
                break  # 只应用第一个匹配的规则
        
        return transformed_path
    
    def _get_files(self, video_info, view=None):
        """
        Get list of frame paths from a directory or video file.
        Supports both image files (jpg/png) and video files (mp4).
        For videos, returns a list of virtual paths in the format:
        "video://<video_path>::<frame_index>::<width>x<height>"
        
        Args:
            video_info: dict或str
                - 如果是dict: 包含video_dir, segment_id, frame_start等
                - 如果是str: 兼容旧格式，直接作为路径
            view: str, 指定的视角名称（mp4文件名或子目录名）
        
        Returns:
            list: 文件路径或帧路径列表
        """
        if isinstance(video_info, str):
            # 旧格式兼容：直接作为路径
            video_dir = video_info
            frame_start = None
            frame_end = None
        else:
            # 新格式：从dict中提取信息
            video_dir = video_info.get('video_dir', video_info)
            frame_start = video_info.get('frame_start')
            frame_end = video_info.get('frame_end')
        
        mp4_files = glob.glob(os.path.join(video_dir, '*.mp4'))
        
        if mp4_files:
            # 有mp4: 根据view参数选择对应的mp4文件
            target_mp4 = None
            if view:
                for mp4 in mp4_files:
                    mp4_name = os.path.splitext(os.path.basename(mp4))[0]
                    if mp4_name == view:
                        target_mp4 = mp4
                        break
            
            if not target_mp4:
                print(f"Warning: No mp4 file found for view: {view}")
                target_mp4 = mp4_files[0]

            try:
                vr = decord.VideoReader(target_mp4, num_threads=1)
                num_frames = len(vr)

                sample_frame = vr[0].asnumpy()
                orig_height, orig_width = sample_frame.shape[:2]

                del vr  # Immediately release resources after getting dimensions

                try:
                    target_height, target_width = smart_resize(
                        height=orig_height,
                        width=orig_width,
                        factor=32,  # image_patch_size (from Qwen VL config)
                        min_pixels=224 * 224,
                        max_pixels=256 * 256
                    )
                except ValueError as e:
                    print(f"Warning: smart_resize failed for {target_mp4}: {e}")
                    target_width, target_height = 224, 224

                if frame_start is not None and frame_end is not None:
                    # 新格式: 只返回segment范围内的帧
                    files = [
                        f"video://{target_mp4}::{i}::{target_width}x{target_height}"
                        for i in range(frame_start, frame_end + 1)
                        if i < num_frames
                    ]
                else:
                    # 旧格式: 返回所有帧
                    files = [
                        f"video://{target_mp4}::{i}::{target_width}x{target_height}"
                        for i in range(num_frames)
                    ]
                return files
            except Exception as e:
                print(f"Error reading video {target_mp4}: {e}. Falling back to images.")
                # Fall through to image loading
        
        # 无mp4: 旧格式，从video_dir/view/目录加载图片
        if view:
            path = os.path.join(video_dir, view)
        else:
            path = video_dir
        
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

    @staticmethod
    def _get_view_frame_count(video_info, view):
        """Return frame count for one view without full _get_files overhead.

        mp4 mode : len(decord.VideoReader), respects frame_start/frame_end.
        image dir: glob count of jpg/png in video_dir/view/.
        Returns None if the view cannot be found or read.
        """
        if isinstance(video_info, str):
            video_dir, frame_start, frame_end = video_info, None, None
        else:
            video_dir   = video_info.get('video_dir', video_info)
            frame_start = video_info.get('frame_start')
            frame_end   = video_info.get('frame_end')

        mp4 = os.path.join(video_dir, f'{view}.mp4')
        if os.path.exists(mp4):
            try:
                vr = decord.VideoReader(mp4, num_threads=1)
                n = len(vr)
                del vr
                if frame_start is not None and frame_end is not None:
                    n = len([i for i in range(frame_start, frame_end + 1) if i < n])
                return n
            except Exception as e:
                # Raise error if file exists but is unreadable (e.g. moov atom missing)
                raise ValueError(f"Failed to read video file {mp4} in _get_view_frame_count: {e}")

        img_dir = os.path.join(video_dir, view)
        if os.path.isdir(img_dir):
            return len(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')))

        return None

    def _sample_steps(self, seq_len, num_steps):
        """Sample frames based on strategy.

        In eval mode sampling is always deterministic ('uniform', evenly spaced);
        in train mode it uses a random offset window ('offset_uniform').
        """
        if self.mode == 'eval':
            if seq_len <= num_steps:
                steps = np.arange(0, seq_len)
                if len(steps) < num_steps:
                    steps = np.pad(steps, (0, num_steps - len(steps)), 'edge')
            else:
                steps = np.linspace(0, seq_len - 1, num_steps, dtype=int)

        else:
            random_offset = int(CONFIG.DATA.RANDOM_OFFSET)
            if seq_len < random_offset:
                # Fallback if video is too short
                steps = np.arange(0, min(seq_len, num_steps))
                if len(steps) < num_steps:
                    steps = np.pad(steps, (0, num_steps - len(steps)), 'edge')
            else:
                if num_steps <= seq_len - random_offset:
                    offset = random_offset
                    available_indices = np.arange(offset, seq_len)
                    np.random.shuffle(available_indices)
                    steps = available_indices[:num_steps]
                    # Sort to keep temporal order
                    steps = np.sort(steps)
                else:
                    # Fallback: sample all available
                    steps = np.arange(0, min(seq_len, num_steps))
                    if len(steps) < num_steps:
                        steps = np.pad(steps, (0, num_steps - len(steps)), 'edge')

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

    def _get_context_steps(self, steps, seq_len, video_path, reverse=False, num_context=None):
        """Get multiple context steps for each chosen step.

        Args:
            num_context: frames per anchor; defaults to CONFIG.DATA.NUM_STEPS.
                         Pass a smaller value (e.g. NUM_STEPS // NUM_VIEWS) for
                         multi-view interleaving so each view contributes a half-chunk.
        """
        if num_context is None:
            num_context = CONFIG.DATA.NUM_STEPS
        stride = self._get_stride(video_path)
        
        # steps shape: (N,)
        # output shape: (N * num_context,)
        
        context_steps = []
        for step in steps:
            # Range: [step - (num-1)*stride, ..., step + stride] with stride
            if not reverse:
                start = step - (num_context - 1) * stride
                end = step + stride
                indices = np.arange(start, end, stride)
            else:
                # [step + (num-1)*stride, ..., step]
                start = step + (num_context - 1) * stride
                end = step - stride
                indices = np.arange(start, end, -stride)

            indices = np.clip(indices, 0, seq_len - 1)
            context_steps.append(indices)
            
        return np.concatenate(context_steps)

    def _context_steps_to_paths(self, context_steps, files):
        """Convert flat context_steps array to a list of file paths."""
        return [files[i] for i in context_steps]

    def _resolve_joint_mapping(self, dataset_name, video_dir):
        """Look up joint mapping entry for this dataset + video_dir.

        Returns (joint_keys, nmd) or ([], None).
        Single-key mapping → return the unique entry directly.
        Multi-key mapping → longest prefix match on dirname(dirname(video_dir))
        (avoids e.g. .../Franka matching .../Franka-sim).
        """
        if dataset_name not in self._joint_action_mapping_cache:
            logging.warning(
                f"[joints] No mapping for dataset {dataset_name!r}; "
                f"joint tokens disabled for this sample."
            )
            self._joint_action_mapping_cache[dataset_name] = None
            return [], None
        mapping = self._joint_action_mapping_cache[dataset_name]
        if mapping is None:
            return [], None
        if len(mapping) == 1:
            entry = next(iter(mapping.values()))
        else:
            sub_data_path = os.path.dirname(os.path.dirname(video_dir.rstrip('/')))
            best_key, entry = None, None
            for key, value in mapping.items():
                if sub_data_path.startswith(key):
                    if best_key is None or len(key) > len(best_key):
                        best_key, entry = key, value
            if entry is None:
                logging.warning(
                    f"[joints] No matching key for sub_data_path={sub_data_path!r} "
                    f"in dataset {dataset_name!r}"
                )
                return [], None
        joint_keys = entry.get('joint_keys', [])
        nmd = entry.get('norm_min_delta')
        return joint_keys, nmd

    def _load_joint_data(self, video_info, dataset_name):
        """Load and normalize per-frame joint states from episode JSON.

        Returns np.float32 [T, D_joint] or None on failure.
        T = clip length (frame_start..frame_end if set, else full episode).
        D_joint = total active joint dims.
        """
        if isinstance(video_info, dict):
            video_dir = video_info.get('video_dir')
            frame_start = video_info.get('frame_start')
            frame_end   = video_info.get('frame_end')
        else:
            video_dir = video_info
            frame_start = None
            frame_end   = None
        joint_keys, nmd = self._resolve_joint_mapping(dataset_name, video_dir)
        if not joint_keys or nmd is None:
            return None

        video_dir = video_dir.rstrip('/')
        ep_name = os.path.basename(video_dir)
        json_path = os.path.join(video_dir, f"{ep_name}.json")
        if not os.path.exists(json_path):
            logging.warning(f"[joints] Episode JSON not found: {json_path}")
            return None

        try:
            with open(json_path, 'r') as f:
                raw = json.load(f)
            entries = raw['data']

            # Slice to clip range if provided (video_info carries frame_start/frame_end
            # for sub-episode clips; without slicing, joint T = full episode != clip len)
            if frame_start is not None and frame_end is not None:
                entries = entries[frame_start : frame_end + 1]
                if not entries:
                    logging.warning(
                        f"[joints] frame range [{frame_start}, {frame_end}] out of bounds "
                        f"for {json_path} (total={len(raw['data'])}); skipping."
                    )
                    return None

            first_entry = entries[0]

            # Filter to keys present in data
            active_keys = [k for k in joint_keys if k in first_entry]
            if not active_keys:
                logging.warning(
                    f"[joints] None of joint_keys {joint_keys} present in {json_path}"
                )
                return None

            # Build norm vectors from nmd
            nm_list, nd_list = [], []
            for k in active_keys:
                field = nmd.get(k)
                if field is None:
                    logging.warning(
                        f"[joints] norm_min_delta missing key {k!r} in {json_path}"
                    )
                    return None
                nm_list.append(np.atleast_1d(np.array(field['min'], dtype=np.float32)))
                nd_list.append(np.atleast_1d(np.array(field['delta'], dtype=np.float32)))
            nm = np.concatenate(nm_list)   # [D_joint]
            nd = np.concatenate(nd_list)   # [D_joint]

            # Assemble [T, D_joint] raw matrix
            rows = []
            for e in entries:
                # JSON may store 1-DOF fields as scalars (e.g. gripper); mapping uses length-1 vectors.
                row = np.concatenate(
                    [np.atleast_1d(np.array(e[k], dtype=np.float32)) for k in active_keys]
                )
                rows.append(row)
            raw_joints = np.stack(rows, axis=0)  # [T, D_joint]

            if raw_joints.shape[1] != nm.shape[0]:
                logging.warning(
                    f"[joints] joint dim mismatch: D_data={raw_joints.shape[1]} != "
                    f"D_norm={nm.shape[0]} (check joint mapping / episode JSON). json={json_path}"
                )
                return None

            # Normalize to [-1, 1]
            safe_nd = np.where(nd == 0, 1.0, nd)
            joints_norm = np.clip(2 * (raw_joints - nm) / safe_nd - 1, -1.0, 1.0)
            return joints_norm.astype(np.float32)

        except Exception as e:
            logging.warning(f"[joints] Failed to load {json_path}: {e}")
            return None

    @staticmethod
    def _replace_view_in_path(path, old_view, new_view):
        """Derive the path for new_view by string-substituting old_view.

        Supports two path styles:
          - MP4 virtual: "video:///ep/cam_front.mp4::42::224x224"
            → replaces "{old_view}.mp4" with "{new_view}.mp4"
          - Image directory: "/ep/images/frame_042.jpg"
            → replaces "/{old_view}/" with "/{new_view}/"
        """
        if f'{old_view}.mp4' in path:
            return path.replace(f'{old_view}.mp4', f'{new_view}.mp4')
        return path.replace(f'/{old_view}/', f'/{new_view}/')

    # -----------------------------------------------------------------------
    # cam_mapping helpers (ported from datasets_3_views.py)
    # -----------------------------------------------------------------------

    @staticmethod
    def _norm_path(p: str) -> str:
        return os.path.normpath(p)

    def _video_dir_from_info(self, video_info):
        if isinstance(video_info, str):
            return video_info
        return video_info.get('video_dir', video_info)

    def _task_path_from_episode(self, video_info) -> str:
        """Episode directory is one level below task; task path = dirname(episode_dir)."""
        ep = self._norm_path(self._video_dir_from_info(video_info))
        return self._norm_path(os.path.dirname(ep))

    def _match_cam_mapping_task_key(self, mapping: dict, task_path: str):
        """Map normalized task path → original JSON task key via precomputed index (O(1))."""
        return mapping['index'].get(self._norm_path(task_path))

    @staticmethod
    def _cam_mapping_task_dict(mapping: dict) -> dict:
        """Task-key -> camera list from loaded wrapper mapping['raw']."""
        return mapping['raw']

    def _view_exists_on_disk(self, video_info, view: str) -> bool:
        vd = self._video_dir_from_info(video_info)
        if os.path.isfile(os.path.join(vd, f'{view}.mp4')):
            return True
        if os.path.isdir(os.path.join(vd, view)):
            return True
        for mp4 in glob.glob(os.path.join(vd, '*.mp4')):
            if os.path.splitext(os.path.basename(mp4))[0] == view:
                return True
        return False

    def _resolve_cam_list_main(self, video_info, dataset_name: str, mapping: dict) -> list:
        """Resolve ordered camera list from cam_mapping; every listed view must exist on main disk."""
        task_path = self._task_path_from_episode(video_info)
        key = self._match_cam_mapping_task_key(mapping, task_path)
        if key is None:
            logging.warning(
                f"[cam_mapping] No task key matches dirname(episode_dir)={task_path!r} for dataset {dataset_name!r} "
                f"(episode_dir={self._video_dir_from_info(video_info)!r})."
            )
            raise IgnoreSample()
        tasks = self._cam_mapping_task_dict(mapping)
        raw = tasks[key]
        if not isinstance(raw, list) or not raw:
            logging.warning(f"[cam_mapping] Empty or invalid camera list for task key {key!r} ({dataset_name})")
            raise IgnoreSample()
        ep = self._video_dir_from_info(video_info)
        missing = [v for v in raw if not self._view_exists_on_disk(video_info, v)]
        if missing:
            logging.warning(
                f"[cam_mapping] Incomplete cameras on main disk for dataset {dataset_name!r} episode {ep!r}: "
                f"missing views {missing} (mapping requires all of {raw!r})"
            )
            raise IgnoreSample()
        # Validate readability and cross-view frame count consistency
        frame_counts = {}
        for v in raw:
            try:
                cnt = AlignmentDataset._get_view_frame_count(video_info, v)
            except ValueError as exc:
                logging.warning(
                    f"[cam_mapping] Unreadable video for view {v!r} in episode {ep!r}: {exc}; skipping sample."
                )
                raise IgnoreSample()
            if cnt is None:
                logging.warning(
                    f"[cam_mapping] _get_view_frame_count returned None for view {v!r} in episode {ep!r}; skipping sample."
                )
                raise IgnoreSample()
            frame_counts[v] = cnt
        if len(set(frame_counts.values())) > 1:
            logging.warning(
                f"[cam_mapping] Frame count mismatch across views for episode {ep!r}: {frame_counts}; skipping sample."
            )
            raise IgnoreSample()
        return list(raw)

    def _cam_list_for_main_ref_align(self, cam_list_main: list, video_info, ref_video_info, align_video_info) -> list:
        """Require every view in cam_list_main to exist on ref and align disks (main already validated)."""
        ref_ep = self._video_dir_from_info(ref_video_info)
        al_ep = self._video_dir_from_info(align_video_info)
        missing_ref = [v for v in cam_list_main if not self._view_exists_on_disk(ref_video_info, v)]
        if missing_ref:
            logging.warning(
                f"[cam_mapping] Incomplete cameras on ref disk {ref_ep!r}: missing views {missing_ref} "
                f"(required all of {cam_list_main!r}); resampling sample."
            )
            raise IgnoreSample()
        missing_align = [v for v in cam_list_main if not self._view_exists_on_disk(align_video_info, v)]
        if missing_align:
            logging.warning(
                f"[cam_mapping] Incomplete cameras on align disk {al_ep!r}: missing views {missing_align} "
                f"(required all of {cam_list_main!r}); resampling sample."
            )
            raise IgnoreSample()
        # Validate readability and cross-view frame count consistency for ref and align episodes
        for ep_label, ep_info, ep_dir in [
            ('ref', ref_video_info, ref_ep),
            ('align', align_video_info, al_ep),
        ]:
            frame_counts = {}
            for v in cam_list_main:
                try:
                    cnt = AlignmentDataset._get_view_frame_count(ep_info, v)
                except ValueError as exc:
                    logging.warning(
                        f"[cam_mapping] Unreadable video for view {v!r} in {ep_label} episode {ep_dir!r}: {exc}; skipping sample."
                    )
                    raise IgnoreSample()
                if cnt is None:
                    logging.warning(
                        f"[cam_mapping] _get_view_frame_count returned None for view {v!r} in {ep_label} episode {ep_dir!r}; skipping sample."
                    )
                    raise IgnoreSample()
                frame_counts[v] = cnt
            if len(set(frame_counts.values())) > 1:
                logging.warning(
                    f"[cam_mapping] Frame count mismatch across views for {ep_label} episode {ep_dir!r}: {frame_counts}; skipping sample."
                )
                raise IgnoreSample()
        return list(cam_list_main)

    @staticmethod
    def _interleave_n_views(path_lists: list, n: int):
        """Interleave k view streams temporally (time-first within each chunk).

        Result ordering (2 views, n=3, 2 chunks):
          [v0_t0, v1_t0, v0_t1, v1_t1, v0_t2, v1_t2,  <- chunk 0
           v0_t3, v1_t3, v0_t4, v1_t4, v0_t5, v1_t5]  <- chunk 1

        Implemented via numpy reshape — no Python loops over individual frames.
        Shape walk: (k, L) → (k, num_chunks, n) → transpose → (num_chunks, n, k) → flatten
        """
        k = len(path_lists)
        if k == 0:
            return []
        if k == 1:
            return path_lists[0][:]
        import numpy as np
        L = len(path_lists[0])
        num_chunks = L // n
        arr = np.empty((k, L), dtype=object)
        for i, pl in enumerate(path_lists):
            arr[i] = pl
        # (k, num_chunks, n) → (num_chunks, n, k) → (num_chunks*n*k,)
        arr = arr.reshape(k, num_chunks, n).transpose(1, 2, 0).reshape(-1)
        return arr.tolist()

    def _apply_multiview_n_views(self, paths_primary, primary_view: str, ordered_views: list, num_ctx: int):
        """ordered_views: views in order (first is primary, matches paths_primary). Supports N views."""
        if len(ordered_views) <= 1:
            return paths_primary, ordered_views
        v0 = primary_view
        if ordered_views[0] != v0:
            logging.warning(f"_apply_multiview_n_views: ordered_views[0]={ordered_views[0]!r} != primary {v0!r}")
        path_lists = [paths_primary]
        for v in ordered_views[1:]:
            path_lists.append([self._replace_view_in_path(p, v0, v) for p in paths_primary])
        return self._interleave_n_views(path_lists, num_ctx), ordered_views

    def _load_video_data_from_json(self, video_idx):
        """
        Stage 1: Data Loading from JSON/filesystem.
        Returns a dict with main_files, ref_files, align_files, instructions, etc.
        View selection is driven by {CAM_MAPPING_DIR}/{dataset_name}_cam_mapping.json.
        If the mapping is missing or the task path is not in the JSON, returns None (skip sample).
        """
        video_info = self.video_paths[video_idx]
        dataset_name = self.video_dataset_names[video_idx] if video_idx < len(self.video_dataset_names) else "unknown"
        dataset_type = self.video_dataset_types[video_idx] if video_idx < len(self.video_dataset_types) else "unknown"

        # ---- cam_mapping check: no fallback to random view ----
        mapping = self._cam_mapping_by_dataset.get(dataset_name)
        use_cam = (
            getattr(CONFIG.DATA, 'USE_CAM_MAPPING', True)
            and (getattr(CONFIG.DATA, 'CAM_MAPPING_DIR', '') or '')
            and mapping is not None
        )
        if not use_cam:
            logging.warning(
                f"[cam_mapping] Missing mapping or CAM_MAPPING_DIR for dataset {dataset_name!r}; "
                f"_load_video_data_from_json requires cam_mapping (no legacy view sampling)."
            )
            return None

        video_dir = video_info.get('video_dir') if isinstance(video_info, dict) else video_info
        filename = 'task_paths_eval.json' if self.mode == 'eval' else 'task_paths.json'

        if isinstance(video_info, dict) and video_info.get('segment_id') is not None:
            # New format: load from segment subfolder
            segment_id = video_info['segment_id']
            task_paths_file = os.path.join(video_dir, str(segment_id), filename)
        else:
            # Old format: load from video root directory
            task_paths_file = os.path.join(video_dir, filename)

        task_paths_file = self._apply_task_paths_transform(task_paths_file, dataset_type)

        # Load task_paths directly (no cache)
        task_paths = {}
        if os.path.exists(task_paths_file):
            try:
                with open(task_paths_file, 'r') as f:
                    task_paths = json.load(f)
            except:
                task_paths = {}
        else:
            if self.mode == 'eval':
                print(f"WARNING: task_paths_eval.json not found at {task_paths_file}. Falling back to self-alignment.")
            else:
                print(f"WARNING: task_paths.json not found at {task_paths_file}. Falling back to self-alignment.")

        if self.mode == 'eval':
            if "same" in task_paths and task_paths["same"]:
                ref_video_info = self._parse_video_path(task_paths["same"][0], dataset_name)
            else:
                ref_video_info = video_info
        else:
            same_pool_keys = ["same", "100-95"]
            candidate_pool = []
            for key in same_pool_keys:
                if key in task_paths and task_paths[key]:
                    candidate_pool.extend(task_paths[key])

            if candidate_pool:
                ref_video_info = self._parse_video_path(random.choice(candidate_pool), dataset_name)
            else:
                ref_video_info = video_info
        align_video_info = ref_video_info

        # ---- Resolve camera list from cam_mapping JSON (no random view selection) ----
        # _resolve_cam_list_main raises IgnoreSample if task key missing or any view absent on disk.
        # _cam_list_for_main_ref_align raises IgnoreSample if any view absent on ref/align disk.
        cam_list_main = self._resolve_cam_list_main(video_info, dataset_name, mapping)
        cam_list = self._cam_list_for_main_ref_align(
            cam_list_main, video_info, ref_video_info, align_video_info)
        view = cam_list[0]
        ref_view = view
        align_view = view
        image_dirs = cam_list
        ref_image_dirs = cam_list
        align_image_dirs = cam_list

        main_files = self._get_files(video_info, view=view)
        ref_files = self._get_files(ref_video_info, view=ref_view)
        align_files = self._get_files(align_video_info, view=align_view)

        if not main_files:
            return None  # Signal failure
        if not ref_files:
            ref_files = main_files
            ref_video_info = video_info
        if not align_files:
            align_files = main_files
            align_video_info = video_info

        main_video_dir = video_info.get('video_dir') if isinstance(video_info, dict) else video_info
        instruction_path = os.path.join(main_video_dir, 'instruction.txt')
        instruction = "Perform the task."
        if os.path.exists(instruction_path):
            with open(instruction_path, 'r') as f:
                instruction = f.read().strip()

        ref_video_dir = ref_video_info.get('video_dir') if isinstance(ref_video_info, dict) else ref_video_info
        ref_instruction_path = os.path.join(ref_video_dir, 'instruction.txt')
        ref_instruction = "Perform the task."
        if os.path.exists(ref_instruction_path):
            with open(ref_instruction_path, 'r') as f:
                ref_instruction = f.read().strip()

        # For compatibility, store original_path as video_path
        video_path_str = video_info.get('original_path') if isinstance(video_info, dict) else video_info
        ref_video_path_str = ref_video_info.get('original_path') if isinstance(ref_video_info, dict) else ref_video_info
        align_video_path_str = align_video_info.get('original_path') if isinstance(align_video_info, dict) else align_video_info

        return {
            'video_path': video_path_str,
            'dataset_name': dataset_name,
            'view': view,
            'available_views':     image_dirs,
            'ref_view':            ref_view,
            'ref_available_views': ref_image_dirs,
            'cam_list':            cam_list,
            'main_files': main_files,
            'ref_video_path': ref_video_path_str,
            'ref_files': ref_files,
            'align_video_path': align_video_path_str,
            'align_files': align_files,
            'instruction': instruction,
            'ref_instruction': ref_instruction,
            'initial_frame_path': main_files[0],
            'ref_initial_frame_path': ref_files[0],
            'video_info': video_info,
            'ref_video_info': ref_video_info,
        }

    def _get_item_impl(self, index):
        video_idx = index // len(self.views)

        # ==================== STAGE 1: Data Loading ====================
        loaded_data = self._load_video_data_from_json(video_idx)

        if loaded_data is None:
            return self._get_item_impl(random.randint(0, len(self) - 1))

        video_path = loaded_data['video_path']
        dataset_name = loaded_data['dataset_name']
        main_files = loaded_data['main_files']
        ref_video_path = loaded_data['ref_video_path']
        ref_files = loaded_data['ref_files']
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
        # Train: bottleneck of both streams. Eval: tier from ref length only (Collator eval matches).
        if self.mode == 'eval':
            seq_len = len(ref_files)
        else:
            seq_len = min(len(main_files), len(ref_files))

        if seq_len >= steps_4x and max_allowed >= steps_4x:
            num_steps = steps_4x
        elif seq_len >= steps_2x and max_allowed >= steps_2x:
            num_steps = steps_2x
        else:
            num_steps = steps_1x
        
        main_steps = self._sample_steps(len(main_files), num_steps)
        ref_steps = self._sample_steps(len(ref_files), num_steps)

        # Joint state loading (no-op when USE_JOINTS=False)
        main_joint_per_step = None
        ref_joint_per_step = None
        if getattr(CONFIG.JOINTS, 'USE_JOINTS', False):
            _vi = loaded_data.get('video_info')
            _ref_vi = loaded_data.get('ref_video_info')
            if _vi is not None:
                main_joint_all = self._load_joint_data(_vi, dataset_name)
                if main_joint_all is not None:
                    if len(main_joint_all) != len(main_files):
                        logging.warning(
                            f"[joints] main joint length mismatch: "
                            f"joint T={len(main_joint_all)} != video frames={len(main_files)}; "
                            f"skipping joints for this sample. path={video_path}"
                        )
                    else:
                        main_joint_per_step = [
                            main_joint_all[s] if s >= 0 else None for s in main_steps
                        ]
            if _ref_vi is not None:
                ref_joint_all = self._load_joint_data(_ref_vi, dataset_name)
                if ref_joint_all is not None:
                    if len(ref_joint_all) != len(ref_files):
                        logging.warning(
                            f"[joints] ref joint length mismatch: "
                            f"joint T={len(ref_joint_all)} != video frames={len(ref_files)}; "
                            f"skipping joints for this sample. path={ref_video_path}"
                        )
                    else:
                        ref_joint_per_step = [
                            ref_joint_all[s] if s >= 0 else None for s in ref_steps
                        ]

        num_align = getattr(CONFIG.TRAIN, 'NUM_ALIGN_FRAMES', 24)

        # Multi-view: JSON cam_list is normalized to exactly 3 views (cycle-pad if <3, first-3 truncate if >3).
        # num_ctx = NUM_STEPS // num_views must be integral (set NUM_STEPS to a multiple of 3).
        cam_list_ld = loaded_data.get('cam_list') if loaded_data is not None else None
        assert cam_list_ld is not None
        cam_list_ld = list(cam_list_ld)
        n_cam = len(cam_list_ld)
        if n_cam == 0:
            raise IgnoreSample()
        if n_cam < 3:
            base = cam_list_ld
            cam_list_ld = [base[i % len(base)] for i in range(3)]
        elif n_cam > 3:
            base = cam_list_ld
            cam_list_ld = base[:3]
        assert len(cam_list_ld) == 3
        num_views = 3
        num_ctx = CONFIG.DATA.NUM_STEPS // num_views   # frames per view per anchor

        main_context_steps = self._get_context_steps(main_steps, len(main_files), video_path, num_context=num_ctx)
        ref_context_steps = self._get_context_steps(ref_steps, len(ref_files), ref_video_path, num_context=num_ctx)

        view = loaded_data['view']

        # num_views and cam_list_ld already set above from loaded_data['cam_list'].
        ref_view = loaded_data.get('ref_view', view)

        main_frame_paths = self._context_steps_to_paths(main_context_steps, main_files)
        if num_views > 1 and cam_list_ld:
            main_frame_paths, _ = self._apply_multiview_n_views(
                main_frame_paths, view, cam_list_ld, num_ctx)

        final_ref_paths = self._context_steps_to_paths(ref_context_steps, ref_files)
        if num_views > 1 and cam_list_ld:
            final_ref_paths, _ = self._apply_multiview_n_views(
                final_ref_paths, ref_view, cam_list_ld, num_ctx)

        # Build combined align: main (first half) + ref (second half).
        # Single-view: align_half = num_align per segment → 2*num_align total.
        # N-view:      align_half = num_align // num_views temporal positions per segment,
        #              then ×num_views views → num_align slots per segment → 2*num_align total.
        align_half = num_align // num_views

        main_align_idxs = get_stratified_idxs(len(main_files), align_half, mode=self.mode)
        ref_align_idxs  = get_stratified_idxs(len(ref_files), align_half, mode=self.mode)
        main_align_paths = [main_files[k] for k in main_align_idxs]
        ref_align_paths  = [ref_files[k]  for k in ref_align_idxs]
        if num_views > 1 and cam_list_ld:
            # n=1: frame-level interleave [v1_t0, v2_t0, ..., vN_t0, v1_t1, ...]
            main_align_paths, _ = self._apply_multiview_n_views(
                main_align_paths, view, cam_list_ld, num_ctx=1)
            ref_align_paths, _  = self._apply_multiview_n_views(
                ref_align_paths, ref_view, cam_list_ld, num_ctx=1)
        align_frame_paths_list     = main_align_paths + ref_align_paths
        main_align_frame_paths_list    = align_frame_paths_list
        num_align_frames_total = len(align_frame_paths_list)

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
            'align_frame_paths': align_frame_paths_list,           # combined main+ref
            'main_align_frame_paths': main_align_frame_paths_list, # same combined list
            'num_align_frames': num_align_frames_total,            # hint for _prepare_per_sample
            'instruction': instruction,
            'initial_frame_path': initial_frame_path,
            'ref_instruction': ref_instruction,
            'dataset_name': dataset_name,
            'ref_initial_frame_path': ref_initial_frame_path,
            'frame_paths_str': None,
            'ref_frame_paths_str': None,
            'align_frame_paths_str': None,
            'main_align_frame_paths_str': None,
            'main_joint_per_step': main_joint_per_step,
            'ref_joint_per_step':  ref_joint_per_step,
        }

        return data

    def __getitem__(self, index):
        while True:
            try:
                return self._get_item_impl(index)
            except Exception as e:
                video_idx = index // len(self.views)
                video_path = self.video_paths[video_idx] if video_idx < len(self.video_paths) else "Unknown"
                
                # Suppress "too short" errors and IgnoreSample (cam_mapping skip) as they are expected
                is_too_short_error = isinstance(e, ValueError) and "too short" in str(e)
                is_ignore_sample = isinstance(e, IgnoreSample)

                if not is_too_short_error and not is_ignore_sample:
                    print(f"Error loading index {index} (video_idx={video_idx}, path='{video_path}'): {e}. Retrying with random index...")
                
                index = random.randint(0, len(self) - 1)

def get_transforms(mode='train'):
    transforms_list = []

    transforms_list.append(transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)))

    if mode == 'train':
        if CONFIG.AUGMENTATION.RANDOM_FLIP:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

        brightness = CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA if CONFIG.AUGMENTATION.BRIGHTNESS else 0
        contrast = 0.5 if CONFIG.AUGMENTATION.CONTRAST else 0 # TF used lower=0.5, upper=1.5 -> factor ~0.5

        if brightness > 0 or contrast > 0:
            transforms_list.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
            ))
            
    # ToTensor (Converts to [0, 1])
    transforms_list.append(transforms.ToTensor())
    
    # Normalize (Mean 0.5, Std 0.5 -> Map [0, 1] to [-1, 1])
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
    
    # Set decord to use native bridge for numpy arrays
    decord.bridge.set_bridge('native')


class ContiguousDistributedSampler(Sampler):
    """Assigns each rank a contiguous index range instead of strided indices.

    PyTorch ``DistributedSampler`` (eval, shuffle=False) yields indices
    ``rank, rank+world_size, rank+2*world_size, ...``. This sampler yields
    ``start, start+1, ..., end-1`` for rank ``rank`` so each GPU processes a
    contiguous block of the dataset ordering (e.g. JSON line order).
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        n = len(dataset)
        r = self.num_replicas
        if r <= 0:
            raise ValueError("num_replicas must be positive")
        # Even split: first (n % r) ranks get one extra sample when n not divisible by r.
        remainder = n % r
        per_base = n // r
        if self.rank < remainder:
            self._start = self.rank * (per_base + 1)
            self.num_samples = per_base + 1
        else:
            self._start = remainder * (per_base + 1) + (self.rank - remainder) * per_base
            self.num_samples = per_base
        self._end = self._start + self.num_samples

    def __iter__(self):
        return iter(range(self._start, self._end))

    def __len__(self):
        return self.num_samples


def create_dataset(split, mode, batch_size=None, return_iterator=True, distributed=False, video_paths_json=None, processor=None, contiguous_distributed_eval=False):
    """Creates dataset iterator."""
    if not batch_size:
        batch_size = CONFIG.TRAIN.BATCH_SIZE if mode == 'train' else CONFIG.EVAL.BATCH_SIZE
        
    transform = get_transforms(mode)
    
    dataset = AlignmentDataset(mode=mode, transform=transform, video_paths_json=video_paths_json, processor=processor)
    
    sampler = None
    use_weighted_sampler = (hasattr(dataset, 'weights') and len(dataset.weights) > 0 and mode == 'train')
    
    if distributed:
        if use_weighted_sampler and mode == 'train':
             weights = torch.DoubleTensor(dataset.weights)
             # Fix: Use a generator seeded by rank to ensure different data on each rank
             rank = torch.distributed.get_rank()
             g = torch.Generator()
             g.manual_seed(42 + rank)
             sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
             shuffle = False
        else:
             # Train / eval: default DistributedSampler is *strided* per rank (0, W, 2W, ...).
             # Optional contiguous shard for eval only (see ContiguousDistributedSampler).
             if mode == 'eval' and contiguous_distributed_eval:
                 sampler = ContiguousDistributedSampler(dataset)
             else:
                 sampler = torch.utils.data.distributed.DistributedSampler(
                     dataset,
                     shuffle=(mode == 'train'),  # Shuffle for train, sequential for eval
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
    
    collator = AlignmentCollator(processor=processor, mode=mode)

    num_workers = 12 if mode == 'train' else 12

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
        collate_fn=collator,
        worker_init_fn=worker_init_fn
    )
    
    if return_iterator and mode == 'train':
        return InfiniteDataLoader(dataloader, sampler=sampler)
    return dataloader
