# coding=utf-8
"""Visualize high loss cases from JSONL logs."""

import argparse
import json
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Global cache for pickle data to avoid repeated loading
_pickle_cache = {}

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
    Load image from file path or pickle path.
    
    Supports:
        - Regular file paths: /path/to/image.jpg
        - Pickle paths: /path/to/file.pkl:task_name/view/frame_idx
        - DUSTBIN marker: "DUSTBIN"
    """
    try:
        if path == "DUSTBIN":
            # Create black image for dustbin
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            # Parse path to determine type
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

def create_alignment_video_with_topn(main_paths, ref_paths, alignment_indices, top5_indices, top5_probs, output_path, loss_val, interval=1000, main_aug=None, ref_aug=None, multiplier=1, chunk_id=0, is_cut_mask=None, topn=5):
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
    
    fig.suptitle(f"Loss: {loss_val:.4f} | M={multiplier} C={chunk_id}", fontsize=16)
    
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
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    try:
        anim.save(output_path, writer='pillow')
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
    plt.close(fig)


def create_alignment_video(main_paths, ref_paths, alignment_indices, output_path, loss_val, interval=1000, main_aug=None, ref_aug=None, multiplier=1, chunk_id=0, is_cut_mask=None):
    """
    Creates a comparison video of main video aligned to ref video.
    Left: Main Video (Source)
    Center: Ref Video (Target) - Aligned Frame chosen by alignment indices
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
    
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), tight_layout=True)
    fig.suptitle(f"Loss: {loss_val:.4f} | M={multiplier} C={chunk_id}", fontsize=16)

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

        # Ref Frame (Normal Playback)
        ax[2].cla()
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
            
            ax[2].imshow(norm_ref_img)
            
            t_text = f'Ref Frame {i}'
            if ref_path_norm == "DUSTBIN":
                t_text += " (Dustbin)"
            
            if is_cut_mask is not None and i < len(is_cut_mask) and is_cut_mask[i] > 0.5:
                t_text += " [CUT]"

            ax[2].set_title(t_text)
        else:
            ax[2].text(0.5, 0.5, "End of Ref Video", ha='center')
        
        ax[2].axis('off')

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    try:
        anim.save(output_path, writer='pillow')
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
    
    count = 0
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")
                continue
            
            # Extract info
            sample_type = record.get('type', 'high_loss') # Default for older logs
            
            # Filter by type
            if args.viz_type != 'all' and sample_type != args.viz_type:
                continue

            step = record.get('step', 'unknown')
            
            # Filter by step
            if isinstance(step, int) and step < args.min_step:
                continue

            dataset = record.get('dataset', 'unknown')
            loss = record.get('loss', 0.0)
            main_paths = record.get('frame_paths', [])
            ref_paths = record.get('ref_frame_paths', [])
            main_aug = record.get('aug_params')
            ref_aug = record.get('ref_aug_params')
            chunk_id = record.get('chunk_id', 0)
            multiplier = record.get('multiplier', 1)
            is_cut_mask = record.get('is_cut_mask', None)
            
            # Prefer forward_alignment_indices (Source -> Target), then fallback to alignment_indices (Source -> Target -> Source)
            alignment = record.get('forward_alignment_indices', record.get('alignment_indices', []))
            
            if not main_paths or not ref_paths or not alignment:
                print(f"Skipping record at step {step}: missing paths or alignment info.")
                continue
            
            # Extract top-N data if available
            top5_indices = record.get('forward_top5_indices', None)
            top5_probs = record.get('forward_top5_probs', None)
            
            # Generate output filename
            # format: viz_{type}_step_{step}_m{multiplier}_c{chunk_id}_{dataset}_loss_{loss:.2f}.gif
            if args.show_topn and top5_indices:
                filename = f"viz_topn_{sample_type}_step_{step}_m{multiplier}_c{chunk_id}_{dataset}_loss_{loss:.2f}.gif"
            else:
                filename = f"viz_{sample_type}_step_{step}_m{multiplier}_c{chunk_id}_{dataset}_loss_{loss:.2f}.gif"
            output_path = os.path.join(output_dir, filename)
            
            print(f"Processing sample {count+1}: Type {sample_type}, Step {step}, Dataset {dataset}, Loss {loss:.4f} (M={multiplier}, C={chunk_id})")
            
            # Choose visualization function based on args.show_topn
            if args.show_topn and top5_indices:
                create_alignment_video_with_topn(
                    main_paths, ref_paths, alignment, top5_indices, top5_probs,
                    output_path, loss, interval=args.interval, 
                    main_aug=main_aug, ref_aug=ref_aug, 
                    multiplier=multiplier, chunk_id=chunk_id, 
                    is_cut_mask=is_cut_mask, topn=args.topn
                )
            else:
                create_alignment_video(
                    main_paths, ref_paths, alignment, 
                    output_path, loss, interval=args.interval, 
                    main_aug=main_aug, ref_aug=ref_aug, 
                    multiplier=multiplier, chunk_id=chunk_id, 
                    is_cut_mask=is_cut_mask
                )
            
            count += 1
            if max_samples is not None and count >= max_samples:
                print(f"Reached max samples {max_samples}.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Path to log directory containing high_loss_samples.jsonl')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save GIFs. Defaults to logdir/high_loss_viz')
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples to visualize')
    parser.add_argument('--min_step', type=int, default=0, help='Minimum step to visualize')
    parser.add_argument('--interval', type=int, default=500, help='Frame interval in ms')
    parser.add_argument('--viz_type', type=str, default='all', choices=['all', 'high_loss', 'low_loss', 'periodic_batch'], help='Type of samples to visualize')
    parser.add_argument('--show_topn', action='store_true', help='Visualize top-N alignment candidates with probabilities')
    parser.add_argument('--topn', type=int, default=5, help='Number of top candidates to show (default: 5)')
    
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
            print(f"Found rank-specific files: {[os.path.basename(f) for f in rank_files]}")
            print("Hint: Use high_loss_samples_all.jsonl (merged) or specify a rank-specific file")
        raise FileNotFoundError(
            f"Could not find high_loss_samples file in {args.logdir}\n"
            f"Looked for: {', '.join(possible_files)}"
        )
    
    # Update args for main consumption
    args.log_file = log_file 
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.logdir, 'high_loss_viz')
        
    main(args)
