# coding=utf-8
"""Visualize high loss cases from JSONL logs."""

import argparse
import json
import os
import pickle
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Decord for efficient video reading
from decord import VideoReader, cpu

# Global cache for pickle data to avoid repeated loading
_pickle_cache = {}

# Global cache for video frames to avoid repeated loading
_video_cache = {}

def parse_pickle_path(path):
    """
    Parse pickle-format path: /path/to/file.pkl:task_name/view/frame_idx
    
    Returns:
        tuple: (path_type, path_info)
            - If pickle: ('pickle', (pkl_path, task_name, view, frame_idx))
            - If regular: ('file', path)
    """
    if ':' in path and '.pkl' in path:
        try:
            # Split at first colon
            pkl_path, rest = path.split(':', 1)
            
            # Parse rest: task_name/view/frame_idx
            parts = rest.split('/')
            if len(parts) >= 3:
                task_name = parts[0]
                view = parts[1]
                frame_str = parts[2]
                
                # Extract frame index from 'frame_123' format
                frame_idx = int(frame_str.replace('frame_', ''))
                
                return 'pickle', (pkl_path, task_name, view, frame_idx)
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse pickle path '{path}': {e}")
    
    return 'file', path


def parse_video_path(path):
    """
    Parse video-format path: video://<video_path>::<frame_idx>::<width>x<height>
    
    Returns:
        tuple: (path_type, path_info)
            - If video: ('video', (video_path, frame_idx, width, height))
            - If regular: ('file', path)
    """
    if isinstance(path, str) and path.startswith("video://"):
        try:
            # Remove "video://" prefix
            path_and_info = path[8:]
            parts = path_and_info.rsplit('::', 2)  # Split by last 2 "::"
            
            if len(parts) == 3:
                # New protocol with size
                video_path, frame_idx_str, size_str = parts
                frame_idx = int(frame_idx_str)
                width, height = map(int, size_str.split('x'))
                return 'video', (video_path, frame_idx, width, height)
            else:
                # Fallback for old protocol without size
                video_path, frame_idx_str = path_and_info.rsplit('::', 1)
                frame_idx = int(frame_idx_str)
                return 'video', (video_path, frame_idx, None, None)
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse video path '{path}': {e}")
    
    return 'file', path


def load_image_from_pickle(pkl_path, task_name, view, frame_idx):
    """
    Load image from pickle file.
    
    Args:
        pkl_path: Path to .pkl file
        task_name: Task name key
        view: View name (e.g., 'images', 'wrist_images')
        frame_idx: Frame index
        
    Returns:
        PIL.Image: Loaded image
    """
    global _pickle_cache
    
    # Load pickle data (with caching)
    # Use a tuple key to cache both raw data and reorganized data
    cache_key = pkl_path
    reorganized_cache_key = f"{pkl_path}_reorganized"
    
    if cache_key not in _pickle_cache:
        with open(pkl_path, 'rb') as f:
            _pickle_cache[cache_key] = pickle.load(f)
    
    data = _pickle_cache[cache_key]
    
    # Handle rollout_batch structure
    if 'rollout_batch' in data:
        rollout_batch = data['rollout_batch']
        
        # Map view name to observation key
        if view == 'images':
            obs_key = 'observation/full_image_list'
        elif view == 'wrist_images':
            obs_key = 'observation/wrist_image_list'
        else:
            obs_key = f'observation/{view}_list'
        
        # Get task descriptions to find task index
        task_descriptions = rollout_batch.get('task_descriptions', [])
        
        # Find matching task index
        task_idx = None
        for i, desc in enumerate(task_descriptions):
            if desc == task_name:
                task_idx = i
                break
        
        if task_idx is None:
            raise ValueError(f"Task '{task_name}' not found in pickle file")
        
        # Reorganize frames if not already cached
        # Original shape: (n_chunks, batch_size, chunk_size, H, W, C)
        # Target shape: (batch_size, total_frames, H, W, C)
        reorganized_key = f"{reorganized_cache_key}_{obs_key}"
        if reorganized_key not in _pickle_cache:
            frames_raw = rollout_batch[obs_key]
            if torch.is_tensor(frames_raw):
                # (n_chunks, batch_size, chunk_size, ...) -> (batch_size, n_chunks, chunk_size, ...)
                n_chunks, batch_size, chunk_size = frames_raw.shape[:3]
                frames_reorganized = frames_raw.permute(1, 0, 2, 3, 4, 5).reshape(
                    batch_size, n_chunks * chunk_size, *frames_raw.shape[3:]
                )
                _pickle_cache[reorganized_key] = frames_reorganized
            elif isinstance(frames_raw, np.ndarray):
                # NumPy version
                n_chunks, batch_size, chunk_size = frames_raw.shape[:3]
                frames_reorganized = frames_raw.transpose(1, 0, 2, 3, 4, 5).reshape(
                    batch_size, n_chunks * chunk_size, *frames_raw.shape[3:]
                )
                _pickle_cache[reorganized_key] = frames_reorganized
            else:
                raise TypeError(f"Unexpected type for frames: {type(frames_raw)}")
        
        frames = _pickle_cache[reorganized_key]
        frame = frames[task_idx, frame_idx]
    else:
        # Standard structure: {task_name: {view: array}}
        frames = data[task_name][view]
        frame = frames[frame_idx]
    
    # Convert to PIL Image
    if torch.is_tensor(frame):
        frame = frame.cpu().numpy()
    
    # Handle different array shapes
    if frame.ndim == 3:
        # (H, W, C) or (C, H, W)
        if frame.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            frame = np.transpose(frame, (1, 2, 0))
    
    # Ensure uint8
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    
    return Image.fromarray(frame)


def load_image_from_video(video_path, frame_idx, target_width=None, target_height=None):
    """
    Load image from video file using decord.
    
    Args:
        video_path: Path to .mp4 file
        frame_idx: Frame index to extract
        target_width: Optional target width for resize during decoding
        target_height: Optional target height for resize during decoding
        
    Returns:
        PIL.Image: Loaded image
    """
    global _video_cache
    
    # Create cache key
    cache_key = (video_path, frame_idx, target_width, target_height)
    
    # Check cache first
    if cache_key in _video_cache:
        return _video_cache[cache_key]
    
    try:
        # Create VideoReader with target dimensions if provided
        if target_width is not None and target_height is not None:
            try:
                vr = VideoReader(
                    video_path,
                    ctx=cpu(0),
                    width=target_width,
                    height=target_height
                )
            except TypeError:
                # Fallback if VideoReader doesn't support width/height
                print(f"Warning: VideoReader doesn't support resize, using original size")
                vr = VideoReader(video_path, ctx=cpu(0))
        else:
            # Use original size
            vr = VideoReader(video_path, ctx=cpu(0))
        
        # Read single frame
        frame = vr[frame_idx]  # Returns NDArray (H, W, C)
        
        # Convert to numpy then PIL Image
        frame_np = frame.asnumpy()
        img = Image.fromarray(frame_np.astype(np.uint8)).convert('RGB')
        
        # Release VideoReader
        del vr
        
        # Cache the result
        _video_cache[cache_key] = img
        
        return img
        
    except Exception as e:
        print(f"Error loading video frame from {video_path} at index {frame_idx}: {e}")
        # Return black image on error
        size = (target_width, target_height) if (target_width and target_height) else (224, 224)
        return Image.new('RGB', size, (0, 0, 0))


def apply_single_augmentation(img, params, step_idx):
    if params is None:
        return img
    
    # 1. Flip
    if params.get('flip', False):
        img = TF.hflip(img)
        
    # 2. Crop
    crop_params = params.get('crop')
    if crop_params is not None:
        i, j, h, w = crop_params
        # size=img.size preserves the original size after cropping a region
        img = TF.resized_crop(img, i, j, h, w, size=img.size[::-1]) # PIL size is (W, H), TF.resized_crop expects (H, W)
        
    # 3. Jitter
    jitter_list = params.get('jitter', [])
    if step_idx < len(jitter_list):
        j_params = jitter_list[step_idx]
        if j_params is not None:
            fn_idx = j_params.get('fn_idx', [])
            b = j_params.get('b')
            c = j_params.get('c')
            s = j_params.get('s')
            h_val = j_params.get('h')
            
            for f_id in fn_idx:
                if f_id == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif f_id == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif f_id == 2 and s is not None:
                    img = TF.adjust_saturation(img, s)
                elif f_id == 3 and h_val is not None:
                    img = TF.adjust_hue(img, h_val)
    return img

def load_image(path, params=None, step_idx=0):
    """
    Load image from file path, pickle path, or video path.
    
    Supports:
        - Regular file paths: /path/to/image.jpg
        - Pickle paths: /path/to/file.pkl:task_name/view/frame_idx
        - Video paths: video://<video_path>::<frame_idx>::<width>x<height>
        - DUSTBIN marker: "DUSTBIN"
    """
    try:
        if path == "DUSTBIN":
            # Create black image for dustbin
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            # Check for video protocol first
            if isinstance(path, str) and path.startswith("video://"):
                path_type, path_info = parse_video_path(path)
                if path_type == 'video':
                    # Load from video
                    video_path, frame_idx, width, height = path_info
                    img = load_image_from_video(video_path, frame_idx, width, height)
                else:
                    # Fallback to regular file
                    img = Image.open(path).convert('RGB')
            else:
                # Parse path to determine type (pickle or regular file)
                path_type, path_info = parse_pickle_path(path)
                
                if path_type == 'pickle':
                    # Load from pickle
                    pkl_path, task_name, view, frame_idx = path_info
                    img = load_image_from_pickle(pkl_path, task_name, view, frame_idx)
                else:
                    # Load from regular file
                    img = Image.open(path).convert('RGB')
            
        if params is not None:
            img = apply_single_augmentation(img, params, step_idx)
        
        img = img.resize((224, 224))  # Standardize size for visualization
        return np.array(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((224, 224, 3), dtype=np.uint8)

def _fps_interval(num_frames, duration, fps=None):
    """Return (fps, interval_ms). If fps is set, use it; else spread num_frames across duration."""
    if fps is not None:
        f = max(float(fps), 1e-6)
        return f, max(1, int(round(1000.0 / f)))
    interval = max(1, round(duration * 1000 / max(num_frames, 1)))
    return round(1000 / interval), interval

def create_alignment_video_with_topn(main_paths, ref_paths, alignment_indices, top5_indices, top5_probs, output_path, loss_val, duration=12.0, fps=None, main_aug=None, ref_aug=None, multiplier=1, chunk_id=0, is_cut_mask=None, topn=5, dtw_loss_val=None, is_masked=None):
    """
    Creates a comparison video with top-N alignment candidates and their probabilities.
    Layout:
    - Row 1: Main Frame | Top-1 Aligned Ref | Normal Ref Playback
    - Row 2: Top-N candidates (small thumbnails with probabilities)
    """
    
    # Store original indices
    main_original_indices = list(range(len(main_paths)))
    ref_original_indices = list(range(len(ref_paths)))
    
    main_len = len(main_paths)
    ref_len = len(ref_paths)
    align_len = len(alignment_indices)
    
    # context_size = Total_Recorded_Frames / Aligned_Sequence_Length
    if align_len > 0:
        c_size_main = main_len // align_len
        c_size_ref = ref_len // align_len
    else:
        c_size_main = c_size_ref = 1
    
    print(f"DEBUG: Shapes - main_paths: {main_len}, ref_paths: {ref_len}, alignment_indices: {align_len}, top5: {len(top5_indices) if top5_indices else 0}")
    
    # Subsample paths to match alignment_indices
    if c_size_main > 1:
        processed_main_indices = []
        sampled_main_paths = []
        for i in range(align_len):
            idx = (i * c_size_main) + (c_size_main - 1)
            sampled_main_paths.append(main_paths[idx])
            processed_main_indices.append(idx)
        main_paths = sampled_main_paths
        main_original_indices = processed_main_indices
    else:
        main_original_indices = list(range(main_len))
        
    if c_size_ref > 1:
        processed_ref_indices = []
        sampled_ref_paths = []
        for i in range(align_len):
            idx = (i * c_size_ref) + (c_size_ref - 1)
            sampled_ref_paths.append(ref_paths[idx])
            processed_ref_indices.append(idx)
        ref_paths = sampled_ref_paths
        ref_original_indices = processed_ref_indices
    else:
        ref_original_indices = list(range(ref_len))
    
    num_frames = min(len(main_paths), len(alignment_indices))
    
    # Create figure with 2 rows: top row for main images, bottom row for top-N candidates
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3 + topn, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Top row: Main, Aligned Ref, Normal Ref
    ax_main = fig.add_subplot(gs[0, 0])
    ax_aligned = fig.add_subplot(gs[0, 1])
    ax_normal = fig.add_subplot(gs[0, 2])
    
    # Bottom row: Top-N candidates
    ax_topn = [fig.add_subplot(gs[1, i]) for i in range(topn)]
    
    _title_topn = f"Align Loss: {loss_val:.4f}"
    if dtw_loss_val is not None:
        _title_topn += f" | DTW Loss: {dtw_loss_val:.4f}"
    if is_masked is not None:
        _title_topn += f" | Masked: {is_masked}"
    _title_topn += f" | M={multiplier} C={chunk_id}"
    fig.suptitle(_title_topn, fontsize=16)
    
    def update(i):
        # Main Frame (Query)
        ax_main.cla()
        main_img = load_image(main_paths[i], params=main_aug, step_idx=main_original_indices[i])
        ax_main.imshow(main_img)
        ax_main.set_title(f'Main Frame {i}', fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Top-1 Aligned Ref Frame
        ax_aligned.cla()
        ref_idx = alignment_indices[i]
        if ref_idx < len(ref_paths):
            ref_path = ref_paths[ref_idx]
            ref_img = load_image(ref_path, params=ref_aug, step_idx=ref_original_indices[ref_idx])
            
            # Draw red border if CUT
            if is_cut_mask is not None and ref_idx < len(is_cut_mask) and is_cut_mask[ref_idx] > 0.5:
                border = 8
                if ref_img.ndim == 3 and ref_img.shape[2] == 3:
                    ref_img[:border, :] = [255, 0, 0]
                    ref_img[-border:, :] = [255, 0, 0]
                    ref_img[:, :border] = [255, 0, 0]
                    ref_img[:, -border:] = [255, 0, 0]
            
            ax_aligned.imshow(ref_img)
            title_text = f'Top-1 Aligned: Ref {ref_idx}'
            if ref_path == "DUSTBIN":
                title_text += " (Dustbin)"
            if is_cut_mask is not None and ref_idx < len(is_cut_mask) and is_cut_mask[ref_idx] > 0.5:
                title_text += " [CUT]"
            ax_aligned.set_title(title_text, fontsize=14, fontweight='bold')
        else:
            ax_aligned.text(0.5, 0.5, "Index Out of Bounds", ha='center')
        ax_aligned.axis('off')
        
        # Normal Ref Playback
        ax_normal.cla()
        if i < len(ref_paths):
            ref_path_norm = ref_paths[i]
            norm_ref_img = load_image(ref_path_norm, params=ref_aug, step_idx=ref_original_indices[i])
            
            # Draw red border if CUT
            if is_cut_mask is not None and i < len(is_cut_mask) and is_cut_mask[i] > 0.5:
                border = 8
                if norm_ref_img.ndim == 3 and norm_ref_img.shape[2] == 3:
                    norm_ref_img[:border, :] = [255, 0, 0]
                    norm_ref_img[-border:, :] = [255, 0, 0]
                    norm_ref_img[:, :border] = [255, 0, 0]
                    norm_ref_img[:, -border:] = [255, 0, 0]
            
            ax_normal.imshow(norm_ref_img)
            t_text = f'Ref Frame {i}'
            if ref_path_norm == "DUSTBIN":
                t_text += " (Dustbin)"
            if is_cut_mask is not None and i < len(is_cut_mask) and is_cut_mask[i] > 0.5:
                t_text += " [CUT]"
            ax_normal.set_title(t_text, fontsize=14)
        else:
            ax_normal.text(0.5, 0.5, "End of Ref Video", ha='center')
        ax_normal.axis('off')
        
        # Top-N candidates
        if top5_indices and i < len(top5_indices):
            candidates = top5_indices[i]
            probs = top5_probs[i] if top5_probs else [1.0] * len(candidates)
            
            for rank, (cand_idx, prob) in enumerate(zip(candidates[:topn], probs[:topn])):
                ax = ax_topn[rank]
                ax.cla()
                
                if cand_idx < len(ref_paths):
                    cand_path = ref_paths[cand_idx]
                    cand_img = load_image(cand_path, params=ref_aug, step_idx=ref_original_indices[cand_idx])
                    ax.imshow(cand_img)
                    
                    # Highlight top-1 with green border
                    if rank == 0:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('green')
                            spine.set_linewidth(3)
                    
                    title = f'Top-{rank+1}: Ref {cand_idx}\nProb: {prob:.3f}'
                    ax.set_title(title, fontsize=10)
                else:
                    ax.text(0.5, 0.5, "OOB", ha='center', va='center')
                    ax.set_title(f'Top-{rank+1}: Invalid', fontsize=10)
                
                ax.axis('off')
        else:
            # No top-N data available
            for rank in range(topn):
                ax_topn[rank].cla()
                ax_topn[rank].text(0.5, 0.5, "No Data", ha='center', va='center')
                ax_topn[rank].set_title(f'Top-{rank+1}', fontsize=10)
                ax_topn[rank].axis('off')
    
    out_fps, interval = _fps_interval(num_frames, duration, fps)
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    try:
        # Use FFmpeg writer for MP4 format with fast encoding and small file size
        anim.save(output_path, writer='ffmpeg', fps=out_fps, 
                  codec='libx264', bitrate=500, 
                  extra_args=['-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-crf', '28'])
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
    plt.close(fig)


def create_alignment_video(main_paths, ref_paths, alignment_indices, output_path, loss_val, duration=12.0, fps=None, main_aug=None, ref_aug=None, multiplier=1, chunk_id=0, is_cut_mask=None, alignment_indices_real=None, alignment_indices_dtw=None, dtw_loss_val=None, is_masked=None):
    """
    Creates a comparison video of main video aligned to ref video.
    Left: Main Video (Source)
    Center-Left: Ref Video (Target) - Aligned Frame (full softmax with dustbin)
    Center: Ref Video (Target) - Aligned Frame (real softmax, dustbin excluded) [if available]
    Center-Right: Ref Video (Target) - DTW Aligned Frame [if available]
    Right: Ref Video (Target) - Normal Playback
    """
    
    # Load Main Frames (these are the query frames)
    # alignment_indices is a list of length T (main sequence length)
    # Each value alignment_indices[i] points to the index in ref frames.
    
    # Store original indices to map back to augmentation params
    main_original_indices = list(range(len(main_paths)))
    ref_original_indices = list(range(len(ref_paths)))

    # Handle Subsampling (NUM_STEPS > 1)
    # The record contains the aggregated frames for the main part only.
    # alignment_indices matches the Main sequence groups.
    
    main_len = len(main_paths)
    ref_len = len(ref_paths)
    align_len = len(alignment_indices)
    
    # context_size = Total_Recorded_Frames / Aligned_Sequence_Length
    if align_len > 0:
        c_size_main = main_len // align_len
        c_size_ref = ref_len // align_len
    else:
        c_size_main = c_size_ref = 1

    print(f"DEBUG: Shapes - main_paths: {main_len}, ref_paths: {ref_len}, alignment_indices: {align_len}")
    print(f"DEBUG: Inferred context_size - Main: {c_size_main}, Ref: {c_size_ref}")

    # Subsample paths to match alignment_indices
    # We take the LAST frame of each context window (matching TCC projection)
    if c_size_main > 1:
        processed_main_indices = []
        sampled_main_paths = []
        for i in range(align_len):
            idx = (i * c_size_main) + (c_size_main - 1)
            sampled_main_paths.append(main_paths[idx])
            processed_main_indices.append(idx)
        main_paths = sampled_main_paths
        main_original_indices = processed_main_indices
    else:
        main_original_indices = list(range(main_len))
        
    if c_size_ref > 1:
        processed_ref_indices = []
        sampled_ref_paths = []
        for i in range(align_len):
            idx = (i * c_size_ref) + (c_size_ref - 1)
            sampled_ref_paths.append(ref_paths[idx])
            processed_ref_indices.append(idx)
        ref_paths = sampled_ref_paths
        ref_original_indices = processed_ref_indices
    else:
        ref_original_indices = list(range(ref_len))

    # Ensure final count matches
    num_frames = min(len(main_paths), len(alignment_indices))
    
    # Determine layout based on whether real alignment and DTW alignment are available
    has_real_alignment = alignment_indices_real is not None
    has_dtw_alignment = alignment_indices_dtw is not None
    
    # Calculate number of columns: Main + Argmax + Real? + DTW? + Normal
    num_cols = 3  # Main + Argmax + Normal (baseline)
    if has_real_alignment:
        num_cols += 1
    if has_dtw_alignment:
        num_cols += 1
    
    fig_width = 5 * num_cols
    
    fig, ax = plt.subplots(ncols=num_cols, figsize=(fig_width, 5), tight_layout=True)
    _title = f"Align Loss: {loss_val:.4f}"
    if dtw_loss_val is not None:
        _title += f" | DTW Loss: {dtw_loss_val:.4f}"
    if is_masked is not None:
        _title += f" | Masked: {is_masked}"
    _title += f" | M={multiplier} C={chunk_id}"
    fig.suptitle(_title, fontsize=16)

    def update(i):
        # Main Frame (Query)
        ax[0].cla()
        main_img = load_image(main_paths[i], params=main_aug, step_idx=main_original_indices[i])
        ax[0].imshow(main_img)
        ax[0].set_title(f'Main Frame {i}')
        ax[0].axis('off')
        
        # Ref Frame (Aligned Target)
        ax[1].cla()
        ref_idx_in_alignment = alignment_indices[i]
        
        # We need to map ref_idx_in_alignment back to ref_paths and ref_aug
        # Since ref_paths was already subsampled if needed, ref_idx_in_alignment should be 
        # relative to the ORIGINAL ref_paths length if alignment_indices were computed on original?
        # NO, in training, alignment labels are index within the sequence passed to loss.
        # If NUM_STEPS > 1, the sequence passed to loss is sequence of groups.
        # So ref_idx_in_alignment is index of the GROUP.
        
        if ref_idx_in_alignment < len(ref_paths):
            ref_path = ref_paths[ref_idx_in_alignment]
            ref_img = load_image(ref_path, params=ref_aug, step_idx=ref_original_indices[ref_idx_in_alignment])
            
            # Draw Red Border if CUT
            if is_cut_mask is not None and ref_idx_in_alignment < len(is_cut_mask) and is_cut_mask[ref_idx_in_alignment] > 0.5:
                 border = 8
                 # Ensure RGB
                 if ref_img.ndim == 3 and ref_img.shape[2] == 3:
                     ref_img[:border, :] = [255, 0, 0]
                     ref_img[-border:, :] = [255, 0, 0]
                     ref_img[:, :border] = [255, 0, 0]
                     ref_img[:, -border:] = [255, 0, 0]

            ax[1].imshow(ref_img)
            
            title_text = f'Aligned {ref_idx_in_alignment}'
            if ref_path == "DUSTBIN":
                title_text += " (Dustbin)"
            elif ref_idx_in_alignment == 0:
                # Just in case other logic puts dustbin at 0 even if path not explicit
                title_text += " (Possible Dustbin)"
            
            if is_cut_mask is not None and ref_idx_in_alignment < len(is_cut_mask) and is_cut_mask[ref_idx_in_alignment] > 0.5:
                title_text += " [CUT]"

            ax[1].set_title(title_text)
        else:
            ax[1].text(0.5, 0.5, "Index Out of Bounds", ha='center')
        
        ax[1].axis('off')

        # Track current column index
        col_idx = 2
        
        # Ref Frame (Aligned Real - if available)
        if has_real_alignment:
            ax[col_idx].cla()
            ref_idx_real = alignment_indices_real[i]
            
            if ref_idx_real < len(ref_paths):
                ref_path_real = ref_paths[ref_idx_real]
                ref_img_real = load_image(ref_path_real, params=ref_aug, step_idx=ref_original_indices[ref_idx_real])
                
                # Draw red border if CUT
                if is_cut_mask is not None and ref_idx_real < len(is_cut_mask) and is_cut_mask[ref_idx_real] > 0.5:
                    border = 8
                    if ref_img_real.ndim == 3 and ref_img_real.shape[2] == 3:
                        ref_img_real[:border, :] = [255, 0, 0]
                        ref_img_real[-border:, :] = [255, 0, 0]
                        ref_img_real[:, :border] = [255, 0, 0]
                        ref_img_real[:, -border:] = [255, 0, 0]

                ax[col_idx].imshow(ref_img_real)
                title_text_real = f'Aligned (Real) {ref_idx_real}'
                if ref_path_real == "DUSTBIN":
                    title_text_real += " (Dustbin)"
                if is_cut_mask is not None and ref_idx_real < len(is_cut_mask) and is_cut_mask[ref_idx_real] > 0.5:
                    title_text_real += " [CUT]"
                ax[col_idx].set_title(title_text_real)
            else:
                ax[col_idx].text(0.5, 0.5, "Index Out of Bounds", ha='center')
            ax[col_idx].axis('off')
            col_idx += 1
        
        # Ref Frame (DTW Aligned - if available)
        if has_dtw_alignment:
            ax[col_idx].cla()
            ref_idx_dtw = alignment_indices_dtw[i]
            
            if ref_idx_dtw < len(ref_paths):
                ref_path_dtw = ref_paths[ref_idx_dtw]
                ref_img_dtw = load_image(ref_path_dtw, params=ref_aug, step_idx=ref_original_indices[ref_idx_dtw])
                
                # Draw blue border to distinguish from others
                border = 8
                if ref_img_dtw.ndim == 3 and ref_img_dtw.shape[2] == 3:
                    # Use blue border for DTW
                    ref_img_dtw[:border, :] = [0, 0, 255]
                    ref_img_dtw[-border:, :] = [0, 0, 255]
                    ref_img_dtw[:, :border] = [0, 0, 255]
                    ref_img_dtw[:, -border:] = [0, 0, 255]
                    
                    # Add red border if CUT
                    if is_cut_mask is not None and ref_idx_dtw < len(is_cut_mask) and is_cut_mask[ref_idx_dtw] > 0.5:
                        # Override with red for CUT
                        ref_img_dtw[:border, :] = [255, 0, 0]
                        ref_img_dtw[-border:, :] = [255, 0, 0]
                        ref_img_dtw[:, :border] = [255, 0, 0]
                        ref_img_dtw[:, -border:] = [255, 0, 0]

                ax[col_idx].imshow(ref_img_dtw)
                title_text_dtw = f'Aligned (DTW) {ref_idx_dtw}'
                if ref_path_dtw == "DUSTBIN":
                    title_text_dtw += " (Dustbin)"
                if is_cut_mask is not None and ref_idx_dtw < len(is_cut_mask) and is_cut_mask[ref_idx_dtw] > 0.5:
                    title_text_dtw += " [CUT]"
                ax[col_idx].set_title(title_text_dtw)
            else:
                ax[col_idx].text(0.5, 0.5, "Index Out of Bounds", ha='center')
            ax[col_idx].axis('off')
            col_idx += 1

        # Ref Frame (Normal Playback)
        normal_ax_idx = col_idx
        ax[normal_ax_idx].cla()
        if i < len(ref_paths):
            ref_path_norm = ref_paths[i]
            norm_ref_img = load_image(ref_path_norm, params=ref_aug, step_idx=ref_original_indices[i])
            
            # Draw Red Border if CUT
            if is_cut_mask is not None and i < len(is_cut_mask) and is_cut_mask[i] > 0.5:
                 border = 8
                 if norm_ref_img.ndim == 3 and norm_ref_img.shape[2] == 3:
                     norm_ref_img[:border, :] = [255, 0, 0]
                     norm_ref_img[-border:, :] = [255, 0, 0]
                     norm_ref_img[:, :border] = [255, 0, 0]
                     norm_ref_img[:, -border:] = [255, 0, 0]
            
            ax[normal_ax_idx].imshow(norm_ref_img)
            
            t_text = f'Ref Frame {i}'
            if ref_path_norm == "DUSTBIN":
                t_text += " (Dustbin)"
            
            if is_cut_mask is not None and i < len(is_cut_mask) and is_cut_mask[i] > 0.5:
                t_text += " [CUT]"

            ax[normal_ax_idx].set_title(t_text)
        else:
            ax[normal_ax_idx].text(0.5, 0.5, "End of Ref Video", ha='center')
        
        ax[normal_ax_idx].axis('off')

    out_fps, interval = _fps_interval(num_frames, duration, fps)
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    try:
        # Use FFmpeg writer for MP4 format with fast encoding and small file size
        anim.save(output_path, writer='ffmpeg', fps=out_fps, 
                  codec='libx264', bitrate=500, 
                  extra_args=['-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-crf', '28'])
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
    plt.close(fig)

def main(args):
    log_file = args.log_file
    output_dir = args.output_dir
    max_samples = args.max_samples
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Reading logs from {log_file}...")
    print(f"Filtering for type: {args.viz_type}")
    
    # Set random seed
    random.seed(args.seed)
    
    # Read all lines and filter them
    all_records = []
    with open(log_file, 'r') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record['_line_no'] = line_no
                
                # Filter by type
                sample_type = record.get('type', 'high_loss')
                if args.viz_type != 'all' and sample_type != args.viz_type:
                    continue
                
                # Filter by step
                step = record.get('step', 'unknown')
                if isinstance(step, int) and step < args.min_step:
                    continue
                
                all_records.append(record)
            except json.JSONDecodeError:
                continue
    
    print(f"Found {len(all_records)} matching records. Randomly sampling up to {max_samples}...")
    
    # Randomly shuffle and sample
    random.shuffle(all_records)
    sampled_records = all_records[:max_samples] if max_samples is not None else all_records
    
    count = 0
    for record in sampled_records:
        line_no = record['_line_no']
        # Extract info
        sample_type = record.get('type', 'high_loss')
        step = record.get('step', 'unknown')

        dataset = record.get('dataset', 'unknown')
        loss = record.get('loss', 0.0)
        dtw_loss = record.get('dtw_loss', None)
        main_paths = record.get('frame_paths', [])
        ref_paths = record.get('ref_frame_paths', [])
        main_aug = record.get('aug_params')
        ref_aug = record.get('ref_aug_params')
        chunk_id = record.get('chunk_id', 0)
        multiplier = record.get('multiplier', 1)
        is_cut_mask = record.get('is_cut_mask', None)
        is_masked = record.get('is_masked')
        
        # Use argmax indices for visualization (not DTW indices)
        # forward_argmax_indices: always available, represents greedy alignment
        alignment = record.get('forward_argmax_indices', record.get('alignment_indices', []))
        
        # Extract real softmax alignment (dustbin excluded) if available
        alignment_real = None
        if 'forward_alignment_indices_real' in record:
            # Real alignment: indices in [0, T_ref-2] correspond to frames [1, T_ref-1]
            # Add 1 to map back to original ref frame indices (accounting for dustbin at 0)
            alignment_real = [idx + 1 for idx in record['forward_alignment_indices_real']]
        
        # Extract DTW alignment if available
        alignment_dtw = record.get('forward_dtw_indices', None)
        
        if not main_paths or not ref_paths or not alignment:
            print(f"Skipping record at step {step}: missing paths or alignment info.")
            continue
        
        # Extract top-N data if available
        top5_indices = record.get('forward_top5_indices', None)
        top5_probs = record.get('forward_top5_probs', None)
        
        # Generate output filename
        # format: viz_{type}_line{line_no}_step_{step}_m{multiplier}_c{chunk_id}_{dataset}_loss_{loss:.2f}[_dtw_{dtw:.2f}].mp4
        loss_tag = f"loss_{loss:.2f}"
        if dtw_loss is not None:
            loss_tag += f"_dtw_{dtw_loss:.2f}"
        if is_masked:
            loss_tag += "_masked"
        
        if args.show_topn and top5_indices:
            filename = f"viz_topn_{sample_type}_line{line_no}_step_{step}_m{multiplier}_c{chunk_id}_{dataset}_{loss_tag}.mp4"
        else:
            filename = f"viz_{sample_type}_line{line_no}_step_{step}_m{multiplier}_c{chunk_id}_{dataset}_{loss_tag}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        dtw_str = f", DTW Loss {dtw_loss:.4f}" if dtw_loss is not None else ""
        print(f"Processing sample {count+1} (line {line_no}): Type {sample_type}, Step {step}, Dataset {dataset}, Loss {loss:.4f}{dtw_str} (M={multiplier}, C={chunk_id})")
        
        # Choose visualization function based on args.show_topn
        if args.show_topn and top5_indices:
            create_alignment_video_with_topn(
                main_paths, ref_paths, alignment, top5_indices, top5_probs,
                output_path, loss, duration=args.duration, fps=args.fps,
                main_aug=main_aug, ref_aug=ref_aug, 
                multiplier=multiplier, chunk_id=chunk_id, 
                is_cut_mask=is_cut_mask, topn=args.topn,
                dtw_loss_val=dtw_loss,
                is_masked=is_masked
            )
        else:
            create_alignment_video(
                main_paths, ref_paths, alignment, 
                output_path, loss, duration=args.duration, fps=args.fps,
                main_aug=main_aug, ref_aug=ref_aug, 
                multiplier=multiplier, chunk_id=chunk_id, 
                is_cut_mask=is_cut_mask,
                alignment_indices_real=alignment_real,
                alignment_indices_dtw=alignment_dtw,
                dtw_loss_val=dtw_loss,
                is_masked=is_masked
            )
        
        count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Path to log directory containing high_loss_samples.jsonl')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save GIFs. Defaults to logdir/high_loss_viz')
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples to visualize')
    parser.add_argument('--min_step', type=int, default=0, help='Minimum step to visualize')
    parser.add_argument('--duration', type=float, default=12.0, help='Target length in seconds when --fps is omitted (default: 12)')
    parser.add_argument('--fps', type=float, default=None, help='Output FPS; if set, overrides duration-based pacing')
    parser.add_argument('--viz_type', type=str, default='all', choices=['all', 'high_loss', 'low_loss', 'periodic_batch'], help='Type of samples to visualize')
    parser.add_argument('--show_topn', action='store_true', help='Visualize top-N alignment candidates with probabilities')
    parser.add_argument('--topn', type=int, default=5, help='Number of top candidates to show (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Smart file detection: support multiple naming conventions
    # Priority: high_loss_samples_all.jsonl > high_loss_samples.jsonl > high_loss_samples_*.jsonl
    possible_files = [
        'high_loss_samples_all.jsonl',  # Multi-GPU merged file
        'high_loss_samples.jsonl',      # Single-GPU or legacy file
    ]
    
    log_file = None
    for filename in possible_files:
        candidate = os.path.join(args.logdir, filename)
        if os.path.exists(candidate):
            log_file = candidate
            print(f"Using log file: {filename}")
            break
    
    if log_file is None:
        # Check for rank-specific files
        import glob
        rank_files = glob.glob(os.path.join(args.logdir, 'high_loss_samples_*.jsonl'))
        if rank_files:
            log_file = sorted(rank_files)[0]
            print(f"Found rank-specific files: {[os.path.basename(f) for f in rank_files]}")
            print(f"Using: {os.path.basename(log_file)}")

    if log_file is None:
        # Last-resort fallback to a known good file
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _fallback = os.path.join(
            _script_dir,
            'logs', 'bridge_data_v2_img1_v2_20260316_161829',
            'high_loss_samples_0.jsonl',
        )
        if os.path.exists(_fallback):
            log_file = _fallback
            print(f"Warning: no log file found in {args.logdir}; falling back to {_fallback}")
            args.logdir = os.path.dirname(_fallback)
        else:
            raise FileNotFoundError(
                f"Could not find high_loss_samples file in {args.logdir}\n"
                f"Looked for: {', '.join(possible_files)}\n"
                f"Fallback also missing: {_fallback}"
            )
    
    # Update args for main consumption
    args.log_file = log_file 
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.logdir, 'high_loss_viz')
        
    main(args)
