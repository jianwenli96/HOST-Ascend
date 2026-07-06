# coding=utf-8
"""Util functions."""

import io
import os
import yaml
import numpy as np
import torch
from easydict import EasyDict
from config import CONFIG

def save_checkpoint(logdir, state, is_best=False):
    step = state.get('global_step', 'unknown')
    filename = os.path.join(logdir, f'checkpoint_{step}.pth.tar')
    # Also save as 'checkpoint.pth.tar' for easy resuming
    latest_filename = os.path.join(logdir, 'checkpoint.pth.tar')
    
    torch.save(state, filename)
    torch.save(state, latest_filename)
    
    if is_best:
        import shutil
        shutil.copyfile(filename, os.path.join(logdir, 'model_best.pth.tar'))

def restore_ckpt(logdir, model, optimizer=None, scheduler=None):
    """Restore checkpoint if exists."""
    checkpoint_path = os.path.join(logdir, 'checkpoint.pth.tar')
    fp32_path = os.path.join(logdir, 'fp32_converted', 'pytorch_model.bin')
    
    start_epoch = 0
    global_step = 0
    
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    elif os.path.isfile(fp32_path):
        print("=> loading FP32 converted checkpoint '{}'".format(fp32_path))
        # This is just a state_dict, not a full checkpoint dict
        state_dict = torch.load(fp32_path, map_location='cpu')
        
        # Handle potential prefix issues if model was wrapped differently
        # But based on our inspection, keys are 'cnn...' and 'emb...', which matches Algorithm
        try:
            model.load_state_dict(state_dict, strict=False)
            print("=> loaded FP32 state_dict successfully.")
        except Exception as e:
            print(f"=> Error loading FP32 state_dict: {e}")
            print("=> Attempting to load with strict=False and ignoring prefixes...")
            # Add more robust loading logic if needed
            
    else:
        print("=> no checkpoint found at '{}' or '{}'".format(checkpoint_path, fp32_path))
    
    return start_epoch, global_step

def to_dict(config):
    if isinstance(config, list):
        return [to_dict(c) for c in config]
    elif isinstance(config, EasyDict):
        return dict([(k, to_dict(v)) for k, v in config.items()])
    else:
        return config


def _convert_to_logical_paths(paths, data_dict, idx):
    """
    Convert frame paths to logical string paths for JSON serialization.
    Handles numpy arrays, tensors, and file paths.
    
    Returns list of strings suitable for JSON.
    """
    # Check if we have logical paths stored separately (use '_str' suffix)
    logical_key = f'{paths}_str'
    if logical_key in data_dict and data_dict[logical_key] is not None:
        if isinstance(data_dict[logical_key], list) and idx < len(data_dict[logical_key]):
            return data_dict[logical_key][idx]
    
    # Otherwise use the regular paths
    if paths not in data_dict:
        return []
    
    path_list = data_dict[paths][idx] if isinstance(data_dict[paths], list) else data_dict[paths]
    
    # Convert to strings
    result = []
    for p in path_list:
        if isinstance(p, str):
            result.append(p)
        elif isinstance(p, (np.ndarray, torch.Tensor)):
            # Skip actual data arrays (shouldn't be here if logical exists)
            result.append(f"<array_data>")
        else:
            result.append(str(p))
    
    return result

def log_and_save_high_loss_samples(logdir, step, loss_dict, data, save_all=False, rank=None):
    """
    Check for high loss samples, log them, and save details to a JSONL file
    for later visualization.
    
    Args:
        logdir: Directory to save logs
        step: Current training/evaluation step
        loss_dict: Dictionary containing loss information
        data: Batch data
        save_all: If True, save all samples (useful for evaluation mode)
        rank: Optional rank identifier for multi-GPU evaluation (saves to high_loss_samples_{rank}.jsonl)
    """
    if 'per_sample_loss' not in loss_dict:
        return

    # Metadata for reconstruction
    qwen_meta = data.get('qwen_input', {})
    group_ids = qwen_meta.get('group_ids')
    
    if group_ids is not None:
        # We are in chunked/grouped mode
        unique_groups = torch.unique(group_ids).cpu().tolist()
        num_unique = len(unique_groups)
        dataset_names_all = data.get('dataset_name', ["unknown"] * len(group_ids))
        
        # Aggregate name/paths for each group
        group_data = []
        # In TCC paired mode, we have [Main, Ref] rows. To reconstruct the video,
        # we only need to look at one of these halves to get the unique chunks.
        half_len = len(group_ids) // 2
        for i, g_id in enumerate(unique_groups):
            # Use only the first half of indices for reconstruction
            indices = (group_ids[:half_len] == g_id).nonzero(as_tuple=True)[0]
            
            # Use data from the chunks in this group
            # Sort indices by chunk_ids to ensure correct sequence order
            chunk_ids = qwen_meta['chunk_ids'][:half_len][indices]
            sort_idx = torch.argsort(chunk_ids)
            sorted_indices = indices[sort_idx]
            
            # Aggregate paths
            all_frame_paths = []
            all_ref_frame_paths = []
            for idx in sorted_indices:
                all_frame_paths.extend(data['frame_paths'][idx])
                all_ref_frame_paths.extend(data['ref_frame_paths'][idx])
            
            # Get logical paths if available (for pickle mode) - aggregate all chunks
            all_logical_frame_paths = []
            all_logical_ref_frame_paths = []
            for idx in sorted_indices:
                chunk_logical_paths = _convert_to_logical_paths('frame_paths', data, idx.item())
                chunk_logical_ref_paths = _convert_to_logical_paths('ref_frame_paths', data, idx.item())
                if chunk_logical_paths:
                    all_logical_frame_paths.extend(chunk_logical_paths)
                if chunk_logical_ref_paths:
                    all_logical_ref_frame_paths.extend(chunk_logical_ref_paths)
            
            logical_frame_paths = all_logical_frame_paths if all_logical_frame_paths else None
            logical_ref_frame_paths = all_logical_ref_frame_paths if all_logical_ref_frame_paths else None
            
            g_entry = {
                "dataset_name": dataset_names_all[sorted_indices[0].item()],
                "main_name": data['name'][sorted_indices[0].item()],
                "ref_name": data['ref_name'][sorted_indices[0].item()],
                "frame_paths": logical_frame_paths if logical_frame_paths else all_frame_paths,
                "ref_frame_paths": logical_ref_frame_paths if logical_ref_frame_paths else all_ref_frame_paths,
                "aug_params": data['aug_params'][sorted_indices[0].item()] if 'aug_params' in data else None,
                "ref_aug_params": data['ref_aug_params'][sorted_indices[0].item()] if 'ref_aug_params' in data else None,
                "multiplier": data['multiplier'][sorted_indices[0].item()] if 'multiplier' in data else 1,
                "chunk_id": "merged",
            }
            
            if 'dtw_guidance_loss_per_sample' in loss_dict:
                dtw_ps_vec = loss_dict['dtw_guidance_loss_per_sample']
                if i < dtw_ps_vec.size(0):
                    g_entry['dtw_loss'] = dtw_ps_vec[i].float().item()

            group_data.append(g_entry)
        
        per_sample_loss = loss_dict['per_sample_loss'] # Shape (2 * num_unique,)
        dataset_names = [d['dataset_name'] for d in group_data]
        batch_size_val = num_unique
    else:
        # Legacy/Single-video mode
        per_sample_loss = loss_dict['per_sample_loss']
        dataset_names = data.get('dataset_name', ["unknown"] * (per_sample_loss.size(0) // 2))
        batch_size_val = len(dataset_names)
        
        group_data = []
        for i in range(batch_size_val):
            # Get logical paths if available (for pickle mode)
            logical_frame_paths = _convert_to_logical_paths('frame_paths', data, i)
            logical_ref_frame_paths = _convert_to_logical_paths('ref_frame_paths', data, i)
            
            g_entry = {
                "dataset_name": dataset_names[i],
                "main_name": data['name'][i] if 'name' in data else "unknown",
                "ref_name": data['ref_name'][i] if 'ref_name' in data else "unknown",
                "frame_paths": logical_frame_paths if logical_frame_paths else data['frame_paths'][i],
                "ref_frame_paths": logical_ref_frame_paths if logical_ref_frame_paths else data['ref_frame_paths'][i],
                "aug_params": data['aug_params'][i] if 'aug_params' in data else None,
                "ref_aug_params": data['ref_aug_params'][i] if 'ref_aug_params' in data else None,
                "multiplier": data.get('multiplier', [1] * (2*batch_size_val))[i] if isinstance(data.get('multiplier'), list) else 1,
                "chunk_id": data.get('chunk_id', [0] * (2*batch_size_val))[i] if isinstance(data.get('chunk_id'), list) else 0,
            }
            
            if 'dtw_guidance_loss_per_sample' in loss_dict:
                dtw_ps_vec = loss_dict['dtw_guidance_loss_per_sample']
                if i < dtw_ps_vec.size(0):
                    g_entry['dtw_loss'] = dtw_ps_vec[i].float().item()

            group_data.append(g_entry)

    # Ensure shape matches expected structure (Main, then Ref)
    if per_sample_loss.view(-1).size(0) != 2 * batch_size_val:
        return

    sample_losses = (per_sample_loss[:batch_size_val] + per_sample_loss[batch_size_val:]) / 2.0
    
    # Aggregate Per-Dataset Loss for this Batch
    dataset_loss_accum = {}
    dataset_counts = {}
    
    for idx, name in enumerate(dataset_names):
        l_val = sample_losses[idx].item()
        dataset_loss_accum[name] = dataset_loss_accum.get(name, 0.0) + l_val
        dataset_counts[name] = dataset_counts.get(name, 0) + 1
    
    # Update loss_dict in place for WandB logging
    for name, total_loss in dataset_loss_accum.items():
        loss_dict[f'loss/{name}'] = total_loss / dataset_counts[name]
    
    # Debug High Loss
    high_threshold = CONFIG.LOGGING.HIGH_LOSS_THRESHOLD
    low_threshold = CONFIG.LOGGING.LOW_LOSS_THRESHOLD
    max_low_loss = CONFIG.LOGGING.MAX_LOW_LOSS_SAMPLES
    debug_step_start = CONFIG.LOGGING.DEBUG_STEP_START
    save_batch_interval = getattr(CONFIG.LOGGING, 'SAVE_BATCH_INTERVAL', 0)
    
    # Initialize counter if not present
    if not hasattr(log_and_save_high_loss_samples, "low_loss_count"):
        log_and_save_high_loss_samples.low_loss_count = 0

    if step > debug_step_start or save_all:
        save_full_batch = (save_batch_interval > 0) and (step % save_batch_interval == 0)
        
        if save_full_batch:
            log_and_save_high_loss_samples.low_loss_count = 0
            
        indices_map = {} # idx -> type
        
        # If save_all is True, save all samples (for eval mode)
        if save_all:
            for idx in range(batch_size_val):
                indices_map[idx] = "eval_all"
        else:
            # 1. Identify High Loss
            if (sample_losses > high_threshold).any():
                 high_loss_indices = (sample_losses > high_threshold).nonzero(as_tuple=True)[0]
                 for idx in high_loss_indices:
                     indices_map[idx.item()] = "high_loss"
                     
            # 2. Identify Low Loss
            if (sample_losses < low_threshold).any():
                low_loss_indices = (sample_losses < low_threshold).nonzero(as_tuple=True)[0]
                for idx in low_loss_indices:
                    idx_val = idx.item()
                    if idx_val not in indices_map:
                        indices_map[idx_val] = "low_loss"
            
            # 3. Fill rest if Periodic
            if save_full_batch:
                for idx in range(batch_size_val):
                    if idx not in indices_map:
                        indices_map[idx] = "periodic_batch"
        
        if indices_map:
            # Use rank-specific filename if rank is provided
            if rank is not None:
                jsonl_file = os.path.join(logdir, f"high_loss_samples_{rank}.jsonl")
            else:
                jsonl_file = os.path.join(logdir, "high_loss_samples.jsonl")
            
            import json
            with open(jsonl_file, "a") as f:
                for idx in sorted(indices_map.keys()):
                    s_type = indices_map[idx]
                    s_loss = sample_losses[idx].item()
                    
                    if s_type == "low_loss" and not save_full_batch:
                        if log_and_save_high_loss_samples.low_loss_count >= max_low_loss:
                            continue
                        log_and_save_high_loss_samples.low_loss_count += 1
                        
                    g_info = group_data[idx]
                    
                    # Construct record
                    record = {
                        "step": step,
                        "type": s_type,
                        "dataset": g_info["dataset_name"],
                        "loss": s_loss,
                        "main_video_path": g_info["main_name"],
                        "ref_video_path": g_info["ref_name"],
                        "frame_paths": g_info["frame_paths"],
                        "ref_frame_paths": g_info["ref_frame_paths"],
                        "aug_params": g_info["aug_params"],
                        "ref_aug_params": g_info["ref_aug_params"],
                        "chunk_id": g_info.get("chunk_id", 0),
                        "multiplier": g_info.get("multiplier", 1),
                    }
                    
                    if 'dtw_loss' in g_info:
                        record['dtw_loss'] = g_info['dtw_loss']

                    # Save alignment indices
                    if 'forward_alignment_indices' in loss_dict:
                        # Forward: Main -> Ref (primary based on config)
                        fwd_align_idx = loss_dict['forward_alignment_indices'][idx].cpu().tolist()
                        record["forward_alignment_indices"] = fwd_align_idx
                        # For backward compatibility, also save as alignment_indices
                        record["alignment_indices"] = fwd_align_idx
                    elif 'alignment_indices' in loss_dict:
                        # Fallback for older code
                        align_idx = loss_dict['alignment_indices'][idx].cpu().tolist()
                        record["alignment_indices"] = align_idx
                    
                    # Backward: Ref -> Main (primary based on config)
                    if 'backward_alignment_indices' in loss_dict:
                        bwd_align_idx = loss_dict['backward_alignment_indices'][idx].cpu().tolist()
                        record["backward_alignment_indices"] = bwd_align_idx
                    
                    # Save argmax indices (always available)
                    if 'forward_argmax_indices' in loss_dict:
                        fwd_argmax_idx = loss_dict['forward_argmax_indices'][idx].cpu().tolist()
                        record["forward_argmax_indices"] = fwd_argmax_idx
                    
                    if 'backward_argmax_indices' in loss_dict:
                        bwd_argmax_idx = loss_dict['backward_argmax_indices'][idx].cpu().tolist()
                        record["backward_argmax_indices"] = bwd_argmax_idx
                    
                    # Save DTW indices (if computed)
                    if 'forward_dtw_indices' in loss_dict and loss_dict['forward_dtw_indices'] is not None:
                        fwd_dtw_idx = loss_dict['forward_dtw_indices'][idx].cpu().tolist()
                        record["forward_dtw_indices"] = fwd_dtw_idx
                    
                    if 'backward_dtw_indices' in loss_dict and loss_dict['backward_dtw_indices'] is not None:
                        bwd_dtw_idx = loss_dict['backward_dtw_indices'][idx].cpu().tolist()
                        record["backward_dtw_indices"] = bwd_dtw_idx
                    
                    # Save top-5 alignment candidates and their probabilities
                    # Forward top-5: Main -> Ref (top 5 ref frames for each main frame)
                    if 'forward_top5_indices' in loss_dict:
                        fwd_top5_idx = loss_dict['forward_top5_indices'][idx].cpu().tolist()
                        fwd_top5_prob = loss_dict['forward_top5_probs'][idx].cpu().tolist()
                        record["forward_top5_indices"] = fwd_top5_idx
                        record["forward_top5_probs"] = fwd_top5_prob
                    
                    # Backward top-5: Ref -> Main (top 5 main frames for each ref frame)
                    if 'backward_top5_indices' in loss_dict:
                        bwd_top5_idx = loss_dict['backward_top5_indices'][idx].cpu().tolist()
                        bwd_top5_prob = loss_dict['backward_top5_probs'][idx].cpu().tolist()
                        record["backward_top5_indices"] = bwd_top5_idx
                        record["backward_top5_probs"] = bwd_top5_prob

                    if loss_dict.get('raw_sim12_mr_path'):
                        record['raw_sim12_mr_path'] = loss_dict['raw_sim12_mr_path']
                        record['raw_sim12_rm_path'] = loss_dict['raw_sim12_rm_path']
                        record['raw_sim12_batch_index'] = idx
                    
                    f.write(json.dumps(record) + "\n")
            
            # Print save confirmation
            num_saved = len(indices_map)
            # print(f"[Step {step}] Saved {num_saved} samples to {jsonl_file}")
            
            # Print breakdown by type
            type_counts = {}
            for s_type in indices_map.values():
                type_counts[s_type] = type_counts.get(s_type, 0) + 1
            # print(f"  Breakdown: {type_counts}")

def setup_train_dir(logdir):
    """Setups directory for training."""
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    # Always save current config to config.yml for reference, but DO NOT load from it.
    config_path = os.path.join(logdir, 'config.yml')
    print('Saving current config to %s', config_path)
    with open(config_path, 'w') as config_file:
        config = dict([(k, to_dict(v)) for k, v in CONFIG.items()])
        yaml.safe_dump(config, config_file, default_flow_style=False)

    train_logs_dir = os.path.join(logdir, 'train_logs')
    if not os.path.exists(train_logs_dir):
        os.makedirs(train_logs_dir)

def get_cnn_feats(cnn, data, training, num_steps=None):
    """Passes data through base CNN."""
    if num_steps is None:
        if training:
            num_steps = CONFIG.TRAIN.NUM_FRAMES * CONFIG.DATA.NUM_STEPS
        else:
            num_steps = CONFIG.EVAL.NUM_FRAMES * CONFIG.DATA.NUM_STEPS

    # Check for Qwen input
    if isinstance(data, dict) and 'qwen_input' in data:
        return cnn(data)

    # Qwen path is always active
    raise RuntimeError("qwen_input not found in data. Qwen model requires processor.")
