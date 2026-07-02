# coding=utf-8
"""Util functions."""

import io
import math
import os
import time
import yaml
import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict
from config import CONFIG

def check_nan(tensor, name, context=""):
  if getattr(CONFIG, 'DEBUG', False):
    if tensor is None:
        return
    
    if isinstance(tensor, dict):
        for k, v in tensor.items():
            check_nan(v, f"{name}[{k}]", context)
        return
    
    if not isinstance(tensor, torch.Tensor):
        return

    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
       msg = f"NaN/Inf detected in {name} at {context}. Shape: {tensor.shape}"
       print(msg)
       # print stats
       if tensor.numel() > 0:
           print(f"Stats: min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}")
       raise ValueError(msg)

def register_debug_hooks(model):
    """Registers backward hooks to detect NaNs during backprop."""
    if not getattr(CONFIG, 'DEBUG', False):
        return

    def hook_fn(module, grad_input, grad_output):
        module_name = module.__class__.__name__
        
        # Check grad_output (gradients coming from upper layers/loss)
        if grad_output is not None:
             for i, g in enumerate(grad_output):
                 if g is not None:
                     check_nan(g, f"{module_name}_grad_output_{i}", "backward_hook")

        # Check grad_input (gradients calculated for this layer's inputs)
        if grad_input is not None:
             for i, g in enumerate(grad_input):
                 if g is not None:
                     check_nan(g, f"{module_name}_grad_input_{i}", "backward_hook")

    print("DEBUG: Registering backward hooks on all modules...")
    for name, module in model.named_modules():
        module.register_full_backward_hook(hook_fn)

def get_lr_scheduler(optimizer, optimizer_config, last_epoch=-1):
    """Returns learning rate scheduler based on config."""
    lr_params = optimizer_config.LR
    if lr_params.DECAY_TYPE == 'exp_decay':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_params.EXP_DECAY_STEPS,
            gamma=lr_params.EXP_DECAY_RATE,
            last_epoch=last_epoch
        )
    elif lr_params.DECAY_TYPE == 'manual':
        lr_step_boundaries = [int(x) for x in lr_params.MANUAL_LR_STEP_BOUNDARIES]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_step_boundaries,
            gamma=lr_params.MANUAL_LR_DECAY_RATE,
            last_epoch=last_epoch
        )
    elif lr_params.DECAY_TYPE == 'fixed':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0, last_epoch=last_epoch)
    elif lr_params.DECAY_TYPE == 'poly':
        # PyTorch doesn't have a direct PolynomialDecay, implementing a lambda
        max_iters = CONFIG.TRAIN.MAX_ITERS
        power = 1.0
        lr_lambda = lambda step: (1 - step / max_iters) ** power
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    elif lr_params.DECAY_TYPE == 'warmup_cosine':
        max_iters = CONFIG.TRAIN.MAX_ITERS
        warmup_fraction = lr_params.WARMUP_FRACTION
        warmup_steps = int(max_iters * warmup_fraction)
        initial_lr = lr_params.INITIAL_LR
        end_lr = lr_params.END_LR
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, max_iters - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - cosine_decay) * end_lr + cosine_decay * initial_lr
                return decayed / initial_lr
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    else:
        raise ValueError('Learning rate decay type %s not supported.' % lr_params.DECAY_TYPE)
    
    # Warmup handling could be added here or wrapped around
    return scheduler

def get_optimizer(model_params, optimizer_config):
    """Returns optimizer based on config."""
    learning_rate = optimizer_config.LR.INITIAL_LR
    if optimizer_config.TYPE == 'AdamOptimizer':
        opt = optim.Adam(model_params, lr=learning_rate)
    elif optimizer_config.TYPE == 'MomentumOptimizer':
        opt = optim.SGD(model_params, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError('Optimizer %s not supported.' % optimizer_config.TYPE)
    return opt

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
            
            if 'dustbin_loss_per_sample' in loss_dict:
                 d_loss_vec = loss_dict['dustbin_loss_per_sample']
                 # Using index i because d_loss_vec corresponds to unique groups
                 if i < d_loss_vec.size(0):
                     g_entry['dustbin_loss'] = d_loss_vec[i].float().item()

            if 'dtw_guidance_loss_per_sample' in loss_dict:
                dtw_ps_vec = loss_dict['dtw_guidance_loss_per_sample']
                if i < dtw_ps_vec.size(0):
                    g_entry['dtw_loss'] = dtw_ps_vec[i].float().item()

            if 'is_cut_mask' in loss_dict:
                 cut_masks = loss_dict['is_cut_mask']
                 if i < cut_masks.size(0):
                     # Mask for this sample
                     g_entry['is_cut_mask'] = cut_masks[i].cpu().tolist()

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
            
            if 'dustbin_loss_per_sample' in loss_dict:
                 d_loss_vec = loss_dict['dustbin_loss_per_sample']
                 if i < d_loss_vec.size(0):
                     g_entry['dustbin_loss'] = d_loss_vec[i].item()

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
                    
                    if 'dustbin_loss' in g_info:
                        record['dustbin_loss'] = g_info['dustbin_loss']

                    if 'dtw_loss' in g_info:
                        record['dtw_loss'] = g_info['dtw_loss']

                    if 'is_cut_mask' in g_info:
                        record['is_cut_mask'] = g_info['is_cut_mask']

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
                    
                    # Save real softmax alignment (dustbin excluded)
                    if 'forward_alignment_indices_real' in loss_dict:
                        fwd_align_idx_real = loss_dict['forward_alignment_indices_real'][idx].cpu().tolist()
                        record["forward_alignment_indices_real"] = fwd_align_idx_real
                    
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

class Stopwatch(object):
    """Simple timer for measuring elapsed time."""

    def __init__(self):
        self.reset()

    def elapsed(self):
        return time.time() - self.time

    def done(self, target_interval):
        return self.elapsed() >= target_interval

    def reset(self):
        self.time = time.time()

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

def get_embeddings_dataset(model, iterator, keep_data=False, keep_labels=True, max_embs=None, save_dir=None, normalize_embeddings=False):
    """Extracts embeddings from the dataset."""
    import collections
    
    # Structure: video_name -> {frame_idx: {'emb': ..., 'frame': ..., 'cand_emb': ..., 'cand_frame': ..., 'cand_name': ...}}
    all_video_results = collections.defaultdict(dict)
    
    model.eval()
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    num_videos_processed = 0
    
    with torch.no_grad():
        for data in iterator:
            # data is from DataLoader with TCCCollator
            # If batch_size > 1, some fields are lists, some are shifted into qwen_input
            
            # Check for max_embs limit (based on unique video names seen so far)
            if max_embs and len(all_video_results) >= max_embs:
                # If the first video in this batch is not already being tracked,
                # it means we've finished the allowed number of videos.
                if data['video_name'][0] not in all_video_results:
                    break
            
            # Forward pass
            # get_cnn_feats handles data if 'qwen_input' is present
            cnn_feats = get_cnn_feats(model.cnn, data, training=False)
            
            # embs: (Combined_B_T, D)
            # We need to know the total frames in this batch
            if 'qwen_input' in data:
                total_frames = data['qwen_input']['input_ids'].shape[0]
            else:
                # Fallback for non-qwen
                b, t, n, c, h, w = data['frames'].shape
                total_frames = b * t

            embs = model.emb(cnn_feats, num_frames=total_frames)
            
            # Handle dictionary output (if model.emb returns more than just embeddings)
            mains_tokens = None
            refs_tokens = None
            if isinstance(embs, dict):
                if 'mains_tokens' in embs:
                    mains_tokens = embs['mains_tokens'] # (N_m, P, D)
                if 'refs_tokens' in embs:
                    refs_tokens = embs['refs_tokens']   # (N_r, P, D)
                embs = embs['embeddings']

            if normalize_embeddings:
                embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

            embs_np = embs.cpu().numpy()
            
            # Extract Batch Info
            # In TCCCollator, each sample in batch is split into 2 rows: [Main_Row, Ref_Row]
            # So batch_inputs effectively has 2 * len(data['video_name']) entries.
            # total_frames is sum of these rows.
            
            # Since we use chunked processing where one Row has 'num_c' frames:
            # We need to correctly map embs back to video_name.
            
            # The indices are stored in qwen_input: group_ids and chunk_ids
            # group_ids[i] refers to the index in the original 'batch' that created this row.
            
            q_meta = data['qwen_input']
            group_ids = q_meta['group_ids'] # (2 * B)
            num_mains = q_meta['num_mains'] # (2 * B)
            num_refs = q_meta['num_refs']   # (2 * B)
            
            batch_size = len(data['video_name'])
            
            # Start offset in embs_np (Flattened embeddings)
            # CAUTION: 'embs' is [All_Mains, All_Refs] if coming from models.py with Qwen input
            # BUT 'group_ids' iterates over input rows.
            # We must track separate offsets for Main tokens and Ref tokens if we want to extract them correctly.
            # For flattened embeddings 'embs', models.py returns [Mains... Refs...].
            # This makes matching linear scan of group_ids difficult for 'embs' unless we split 'embs' too.

            # Determine splits for embs
            total_mains = num_mains.sum().item()
            total_refs = num_refs.sum().item()
            
            all_mains_embs = embs_np[:total_mains]
            all_refs_embs = embs_np[total_mains:] # Should be total_refs length
            
            off_m = 0
            off_r = 0
            
            for i in range(len(group_ids)):
                g_idx = group_ids[i].item()
                v_name = data['video_name'][g_idx]

                # If we've reached the limit and this is a new video name, skip it.
                # This prevents "leaking" into the next video in a mixed batch.
                if max_embs and v_name not in all_video_results and len(all_video_results) >= max_embs:
                    # Need to advance offsets to keep sync
                    off_m += num_mains[i].item()
                    off_r += num_refs[i].item()
                    continue

                cand_name = data.get('candidate_name', [None]*batch_size)[g_idx]
                
                n_m = num_mains[i].item()
                n_r = num_refs[i].item()
                
                if n_m > 0:
                    # This is a Main row
                    row_embs = all_mains_embs[off_m : off_m + n_m]
                    row_tokens = mains_tokens[off_m : off_m + n_m] if mains_tokens is not None else None
                    off_m += n_m
                    
                    g_indices = data['global_indices'][g_idx].numpy()
                    # data['frames']: (B, T, NUM_STEPS, C, H, W)
                    main_frames_np = data['frames'][g_idx].numpy() if keep_data else None
                    
                    for t in range(len(g_indices)):
                        idx = g_indices[t]
                        if idx not in all_video_results[v_name]:
                            all_video_results[v_name][idx] = {}
                        all_video_results[v_name][idx]['emb'] = row_embs[t]
                        if row_tokens is not None:
                            all_video_results[v_name][idx]['token'] = row_tokens[t].float().cpu().numpy()
                        if cand_name is not None:
                             all_video_results[v_name]['cand_name'] = cand_name
                        if main_frames_np is not None:
                            all_video_results[v_name][idx]['frame'] = main_frames_np[t]
                            
                elif n_r > 0:
                    # This is a Ref row
                    row_embs = all_refs_embs[off_r : off_r + n_r]
                    row_tokens = refs_tokens[off_r : off_r + n_r] if refs_tokens is not None else None
                    off_r += n_r
                    
                    cand_indices = data['candidate_global_indices'][g_idx].numpy()
                    ref_frames_np = data['candidate_frames'][g_idx].numpy() if keep_data else None
                    
                    if 'cand_results' not in all_video_results[v_name]:
                        all_video_results[v_name]['cand_results'] = {}
                        
                    for t in range(len(cand_indices)):
                        c_idx = cand_indices[t]
                        all_video_results[v_name]['cand_results'][c_idx] = {
                            'emb': row_embs[t],
                            'cand_name': cand_name
                        }
                        if row_tokens is not None:
                            all_video_results[v_name]['cand_results'][c_idx]['token'] = row_tokens[t].float().cpu().numpy()
                        if ref_frames_np is not None:
                            all_video_results[v_name]['cand_results'][c_idx]['frame'] = ref_frames_np[t]

            print(f"Processed batch, total videos in buffer: {len(all_video_results)}")

    # Finalization: Save each video
    index_data = []
    
    # Pre-computation: Ensure we have all candidate videos loaded if we want to compute gates
    # But candidates are often stored inside 'cand_results' of the main video anyway?
    # Yes, cand_results stores embeddings (and now tokens) of the reference video PAIRED with this main video.
    # So we don't need to look up other videos! We have the pair right here: Main(results) vs Ref(results['cand_results']).
    
    for count, (v_name, results) in enumerate(all_video_results.items()):
        if max_embs and count >= max_embs:
            break
            
        # Sort indices
        sorted_indices = sorted([k for k in results.keys() if isinstance(k, (int, np.integer))])
        
        video_embs = np.stack([results[idx]['emb'] for idx in sorted_indices])
        
        result = {'embs': video_embs}
        
        # Stack Tokens if present
        if 'token' in results[sorted_indices[0]]:
             video_tokens = np.stack([results[idx]['token'] for idx in sorted_indices]) # (T_main, P, D)
        else:
             video_tokens = None
        
        if keep_data:
            result['frames'] = np.stack([results[idx]['frame'] for idx in sorted_indices])
            
        cand_tokens = None
        if 'cand_results' in results:
            cand_results = results['cand_results']
            sorted_cand_indices = sorted(cand_results.keys())
            result['candidate_embs'] = np.stack([cand_results[idx]['emb'] for idx in sorted_cand_indices])
            if 'token' in cand_results[sorted_cand_indices[0]]:
                cand_tokens = np.stack([cand_results[idx]['token'] for idx in sorted_cand_indices]) # (T_ref, P, D)
                
            if keep_data:
                result['candidate_frames'] = np.stack([cand_results[idx]['frame'] for idx in sorted_cand_indices])

        # --- Compute Gate for this Pair ---
        if hasattr(model, 'gate') and model.gate is not None and video_tokens is not None and cand_tokens is not None:
            # We have both Main tokens and Reference tokens for this specific pair
            # Prepare inputs
            # Ensure we use the correct dtype matching the model weights (e.g. bfloat16)
            # FIX: Use gate parameters specifically to avoid device mismatch if CNN is on different device
            gate_param = next(model.gate.parameters())
            device = gate_param.device
            dtype = gate_param.dtype
            
            t1 = torch.from_numpy(video_tokens).unsqueeze(0).to(device=device, dtype=dtype) # (1, T_m, P, D)
            t2 = torch.from_numpy(cand_tokens).unsqueeze(0).to(device=device, dtype=dtype)   # (1, T_r, P, D)
            
            # Debug: Token Stats
            # print(f"Gate Input T1: Mean {t1.mean().item():.4f}, Std {t1.std().item():.4f}, Dtype {t1.dtype}")
            
            with torch.no_grad():
                # Process in chunks to avoid CUDA OOM / Configuration errors
                # Total Batch for transformer = B(=1) * Chunk_Tm * Tr
                # We want Chunk_Tm * Tr <= MAX_BATCH
                
                Tr = t2.shape[1]
                if Tr > 0:
                    MAX_BATCH = 128 # Conservative batch size (number of frame pairs)
                    chunk_size = max(1, MAX_BATCH // Tr)
                    
                    gate_outputs = []
                    for i in range(0, t1.shape[1], chunk_size):
                        t1_chunk = t1[:, i : i + chunk_size]
                        # t1_chunk: (1, current_chunk, P, D)
                        # t2: (1, Tr, P, D)
                        # gate returns: (1, current_chunk, Tr, 1)
                        out_chunk = model.gate(t1_chunk, t2) 
                        gate_outputs.append(out_chunk.float().cpu()) # Ensure we move back as float32
                        
                    full_gate = torch.cat(gate_outputs, dim=1)
                    result['gate'] = full_gate.numpy()[0, :, :, 0] # (T_m, T_r)
                else:
                    # No reference frames?
                    result['gate'] = np.zeros((t1.shape[1], 0))

        if save_dir:
            filename = f"video_{count}.npy"
            filepath = os.path.join(save_dir, filename)
            np.save(filepath, result)
            
            index_entry = {
                'id': count,
                'name': str(v_name),
                'file': filename
            }
            if 'cand_results' in results:
                cand_res = results['cand_results']
                first_cand_idx = sorted(cand_res.keys())[0]
                index_entry['candidate_name'] = str(cand_res[first_cand_idx].get('cand_name', 'unknown'))
                
            index_data.append(index_entry)
            print(f"Saved {v_name} to {filename}")
        else:
            embeddings.append(video_embs)
            if keep_data:
                frames.append(result['frames'])
                
    if save_dir:
        import json
        with open(os.path.join(save_dir, 'dataset_index.json'), 'w') as f:
            json.dump(index_data, f, indent=2)
        return None
    else:
        return {'embs': embeddings, 'frames': frames}

