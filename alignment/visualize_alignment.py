# coding=utf-8
"""Visualize alignment based on nearest neighbor in embedding space."""

import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from dtw import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("DTW not found. Falling back to Nearest Neighbor.")

def dist_fn(x, y):
    # Ensure inputs are normalized for distance calculation if config says so.
    # But since we normalize feats before passing to align, this is just L2.
    dist = np.sum((x-y)**2)
    return dist

def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    if max_v - min_v > 0:
        query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame

def align(query_feats, candidate_feats, use_dtw, gate=None, k=3):
    if use_dtw and HAS_DTW:
        # DTW not yet modified to support Gate or Top-K. 
        # Focusing on NN alignment as requested.
        _, _, _, path = dtw(query_feats, candidate_feats, dist=dist_fn)
        _, uix = np.unique(path[0], return_index=True)
        # For DTW, we only have one path, but we return it K times or pad
        nns = path[1][uix]
        topk_indices = np.stack([nns] * k, axis=1) # (N, k)
        topk_probs = np.ones((len(nns), k))
    else:
        # Replicate align_pair_of_sequences logic
        # 1. Calculate Unscaled Similarity (Negative L2 Distance Squared)
        # q: (N, D), c: (M, D)
        q_sq = np.sum(query_feats**2, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(candidate_feats**2, axis=1, keepdims=True).T  # (1, M)
        dist_sq = q_sq + c_sq - 2 * np.dot(query_feats, candidate_feats.T)  # (N, M)
        sim_12 = -1.0 * dist_sq

        # 2. Scale by Temperature (and Channels if standard TCC, but visualization often skips Channel Norm)
        # To strictly match align_pair_of_sequences -> get_scaled_similarity:
        # similarity /= channels (if not normalized exp, but here we assume trained setup)
        # similarity /= temperature
        # Let's assume default temperature 0.1 for TCC
        
        # Check config norm logic
        channels = float(query_feats.shape[-1])
        normalize_embeddings = args.normalize_embeddings
        if not normalize_embeddings:
             sim_12 /= channels
             
        temperature = 0.1
        # Check config if available, but for vis script simpler is better
        sim_12 = sim_12 / temperature

        # 3. Softmax
        # For numerical stability with numpy
        sim_12_max = np.max(sim_12, axis=-1, keepdims=True)
        exp_sim = np.exp(sim_12 - sim_12_max)
        softmaxed_sim_12 = exp_sim / np.sum(exp_sim, axis=-1, keepdims=True)

        # 4. Apply Gate
        if gate is not None:
             if gate.shape != softmaxed_sim_12.shape:
                  print(f"Warning: Gate shape {gate.shape} != Sim shape {softmaxed_sim_12.shape}. Ignoring Gate.")
             else:
                  print("Applying Gate to similarity.")
                  softmaxed_sim_12 = softmaxed_sim_12 * gate
                  # Re-normalize
                  row_sums = np.sum(softmaxed_sim_12, axis=-1, keepdims=True) + 1e-8
                  softmaxed_sim_12 = softmaxed_sim_12 / row_sums
        
        # 5. Get Top-K Nearest Neighbor from Final Distribution
        # Use argsort to get indices of top-k. argsort sorts ascending, so take last k
        topk_indices_asc = np.argsort(softmaxed_sim_12, axis=1)[:, -k:]
        # Flip to get Descending (Best first)
        topk_indices = np.flip(topk_indices_asc, axis=1)
        
        # Get probs using take_along_axis
        topk_probs = np.take_along_axis(softmaxed_sim_12, topk_indices, axis=1)
        
    return topk_indices, topk_probs

def create_video(query_feats, query_frames, candidate_feats, candidate_frames, video_path, use_dtw, interval, gate=None):
    print(f"Query feats shape: {query_feats.shape}")
    print(f"Candidate feats shape: {candidate_feats.shape}")
    print(f"Query frames shape: {query_frames.shape}")
    print(f"Candidate frames shape: {candidate_frames.shape}")
    if gate is not None:
        print(f"Gate shape: {gate.shape}")
        print(f"Gate Mean: {np.mean(gate):.4f}, Max: {np.max(gate):.4f}, Min: {np.min(gate):.4f}")
        print(f"Gate Sparsity (<0.5): {np.mean(gate < 0.5):.4f}")
        print(f"Gate Sparsity (<0.1): {np.mean(gate < 0.1):.4f}")
        print(f"Gate Sparsity (<0.01): {np.mean(gate < 0.01):.4f}")
    
    # Align
    k = 3 # Top-K
    topk_indices, topk_probs = align(query_feats, candidate_feats, use_dtw, gate=gate, k=k)
    print(f"Aligned Indices shape: {topk_indices.shape}")
    
    # Create animation
    # Reduced figsize and DPI for better compression/smaller GIF size
    # 1 col for query, k cols for candidates
    fig, ax = plt.subplots(ncols=1+k, figsize=(4*(1+k), 4), tight_layout=True)
    
    def update(i):
        if i >= len(query_frames):
            return
        
        # Query Frame in Col 0
        ax[0].cla()
        q_frame = query_frames[i]
        # Handle Context: (NUM_STEPS, C, H, W) -> Take last frame
        if q_frame.ndim == 4:
            q_frame = q_frame[-1]
            
        # Transpose (C, H, W) -> (H, W, C)
        q_frame = np.transpose(q_frame, (1, 2, 0))
        q_frame = unnorm(q_frame)
        ax[0].imshow(q_frame)
        ax[0].set_title(f'Query Frame {i}')
        ax[0].axis('off')
        
        # Candidate Frames in Col 1..k
        if i < len(topk_indices):
            for rank in range(k):
                 ax_idx = rank + 1
                 ax[ax_idx].cla()
                 
                 c_idx = topk_indices[i, rank]
                 prob_val = topk_probs[i, rank]
                 
                 if c_idx < len(candidate_frames):
                    c_frame = candidate_frames[c_idx]
                    if c_frame.ndim == 4:
                        c_frame = c_frame[-1]
                    c_frame = np.transpose(c_frame, (1, 2, 0))
                    c_frame = unnorm(c_frame)
                    ax[ax_idx].imshow(c_frame)
                    ax[ax_idx].set_title(f'Rank {rank+1}: {c_idx}\nConf: {prob_val:.4f}')
                 else:
                    pass
                 ax[ax_idx].axis('off')
        
    anim = FuncAnimation(fig, update, frames=len(query_frames), interval=interval)
    try:
        # Use lower DPI for compression
        anim.save(video_path, writer='pillow', dpi=60)
        print(f"Saved video to {video_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        import traceback
        traceback.print_exc()
    plt.close(fig)

def visualize(args):
    import json
    
    embs_dir = args.embs_path
    if embs_dir.endswith('.npy'):
        embs_dir = os.path.dirname(embs_dir)
        
    index_path = os.path.join(embs_dir, 'dataset_index.json')
    if not os.path.exists(index_path):
        print(f"Index file not found at {index_path}. Cannot visualize batch.")
        return
        
    with open(index_path, 'r') as f:
        dataset_index = json.load(f)
        
    # Output dir
    vis_dir = args.video_path
    # If video_path looks like a file (ends with extension), take dirname
    if os.path.splitext(vis_dir)[1]:
         vis_dir = os.path.dirname(vis_dir)
         
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    if not dataset_index:
        print("Dataset index is empty.")
        return

    # Check if first entry has candidate info
    first_entry = dataset_index[0]
    first_data = np.load(os.path.join(embs_dir, first_entry['file']), allow_pickle=True).item()
    has_candidate = 'candidate_embs' in first_data
    
    if has_candidate:
        print("Detected paired embeddings (Query + Candidate). Visualizing pairs...")
        for entry in dataset_index:
            data = np.load(os.path.join(embs_dir, entry['file']), allow_pickle=True).item()
            
            query_embs = data['embs']
            query_frames = data['frames']
            cand_embs = data['candidate_embs']
            cand_frames = data['candidate_frames']
            gate = data.get('gate', None)
            
            if args.normalize_embeddings:
                query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
                cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)
                
            output_filename = f"alignment_{entry['id']}.gif"
            output_path = os.path.join(vis_dir, output_filename)
            
            print(f"Visualizing pair {entry['id']}...")
            create_video(query_embs, query_frames, cand_embs, cand_frames, output_path, args.use_dtw, args.interval, gate=gate)
            
    else:
        print("No paired embeddings found. Using Reference vs All mode.")
        
        # Load Reference
        ref_idx = args.reference_video
        ref_entry = next((item for item in dataset_index if item['id'] == ref_idx), None)
        
        if not ref_entry:
            # Fallback to first one
            ref_entry = dataset_index[0]
            print(f"Reference video {ref_idx} not found. Using {ref_entry['id']} instead.")
            
        ref_data = np.load(os.path.join(embs_dir, ref_entry['file']), allow_pickle=True).item()
        ref_embs = ref_data['embs']
        ref_frames = ref_data['frames']
        
        if args.normalize_embeddings:
            ref_embs = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)
            
        # Iterate over candidates
        for entry in dataset_index:
            if entry['id'] == ref_entry['id']:
                continue
                
            print(f"Visualizing alignment between {ref_entry['id']} and {entry['id']}...")
            
            cand_data = np.load(os.path.join(embs_dir, entry['file']), allow_pickle=True).item()
            cand_embs = cand_data['embs']
            cand_frames = cand_data['frames']
            
            if args.normalize_embeddings:
                cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)
                
            output_filename = f"alignment_{ref_entry['id']}_to_{entry['id']}.gif"
            output_path = os.path.join(vis_dir, output_filename)
            
            create_video(ref_embs, ref_frames, cand_embs, cand_frames, output_path, args.use_dtw, args.interval)

if __name__ == '__main__':
    from config import CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument('--embs_path', type=str, required=True, help='Path to embeddings directory.')
    parser.add_argument('--video_path', type=str, required=True, help='Output video path (directory will be used).')
    parser.add_argument('--use_dtw', action='store_true', help='Use DTW.')
    parser.add_argument('--reference_video', type=int, default=0, help='Reference video index.')
    parser.add_argument('--candidate_video', type=int, default=None, help='Candidate video index (ignored in batch mode).')
    parser.add_argument('--normalize_embeddings', type=int, default=int(CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS), 
                        help='Normalize embeddings (0 or 1). Default matches training config.')
    parser.add_argument('--interval', type=int, default=30, help='Interval in ms.')

    parser.add_argument('--alsologtostderr', action='store_true', help='Log to stderr (dummy).')
    
    args = parser.parse_args()
    # Convert int to bool if needed, though visualize uses it as truthy
    args.normalize_embeddings = bool(args.normalize_embeddings)
    visualize(args)
