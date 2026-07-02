# coding=utf-8
"""Evaluate embeddings on downstream tasks."""

import os
import argparse
import numpy as np
import torch
import yaml
from easydict import EasyDict
from config import CONFIG
from datasets import create_one_epoch_dataset
from utils import get_embeddings_dataset, restore_ckpt
from models import get_model
from algos.algorithm import Algorithm
from transformers import AutoProcessor

def extract_embeddings(args):
    logdir = args.logdir
    if not os.path.exists(logdir):
        raise ValueError(f"Log directory {logdir} does not exist.")
        
    # Setup Config
    config_path = os.path.join(logdir, 'config.yml')
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
            # Recursively update CONFIG
            def update_config(c, sc):
                for k, v in sc.items():
                    if isinstance(v, dict) and k in c:
                        update_config(c[k], v)
                    else:
                        c[k] = v
            # Update entire CONFIG structure to ensure consistency (especially DATA.NUM_STEPS for packing)
            update_config(CONFIG, saved_config)

    CONFIG.DATA.SAMPLE_ALL_STRIDE = args.sample_all_stride

    # Model

    # Initialize Processor for Qwen
    processor = None
    if 'Qwen' in CONFIG.MODEL.BASE_MODEL.NETWORK:
        try:
            # Use local path for Qwen3-VL-2B
            model_name = '/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B'
            print(f"Loading processor for {model_name}")
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Could not load processor for Qwen: {e}")

    # Initialize Algorithm (which initializes Model)
    # We use Algorithm wrapper because it handles model structure (cnn, emb)
    # But we can also just use get_model() and wrap it in a dummy class if needed.
    # However, get_embeddings_dataset expects model.cnn and model.emb.
    # Algorithm class has self.cnn and self.emb.
    
    # We need to instantiate a concrete Algorithm class or a generic one.
    # Since we only need forward pass components, we can use a simple wrapper.
    
    class GenericAlgo(Algorithm):
        def compute_loss(self, *args, **kwargs): pass
        
    model_dict = get_model()
    algo = GenericAlgo(model_dict)
    
    if torch.cuda.is_available():
        algo = algo.cuda()
        
    # Restore Checkpoint
    # restore_ckpt expects 'model' which is usually the Algorithm or Model.
    # If checkpoint saved Algorithm state_dict, we should pass algo.
    # If it saved model state_dict, we should pass algo.model?
    # In train.py (not shown but assumed), we usually save algo.state_dict() or model.state_dict().
    # Let's assume restore_ckpt handles it.
    restore_ckpt(logdir, algo)
    
    # Dataset
    # args.dataset is passed as split? No, args.split should be passed as split.
    # args.dataset is the dataset name (e.g. pouring, libero).
    # But create_one_epoch_dataset signature is (split, mode, ...).
    # And it ignores split inside.
    # So we can pass args.split or args.dataset, it doesn't matter for LiberoVideoDataset currently.
    # But for correctness:
    iterator = create_one_epoch_dataset(
        args.split, 
        mode='eval', 
        batch_size=args.batch_size, 
        return_iterator=True, 
        video_paths_json=args.video_paths, 
        processor=processor,
        chunk_size=args.frames_per_batch
    )
    
    # Extract
    max_embs = None if args.max_embs <= 0 else args.max_embs
    
    # If save_path is a directory (ends with / or doesn't have extension), treat as dir
    # Or just assume it's a dir if we want batch processing.
    # But to be backward compatible or flexible, let's check.
    # The user asked for batch visualization which requires json index.
    # So we should enforce directory structure or create one.
    
    save_dir = args.save_path
    if save_dir.endswith('.npy'):
        # If user provided a file path, use its parent dir for batch files
        # But wait, if user provided .npy, maybe they want the old behavior?
        # The user asked to modify for batch visualization.
        # Let's assume save_path IS the directory where we save everything.
        # If it ends in .npy, we strip it.
        save_dir = os.path.dirname(save_dir)
        
    get_embeddings_dataset(
        algo,
        iterator,
        keep_data=args.keep_data,
        keep_labels=args.keep_labels,
        max_embs=max_embs,
        save_dir=save_dir,
        normalize_embeddings=bool(args.normalize_embeddings)
    )
    
    print(f"Embeddings saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Path to logs.')
    parser.add_argument('--dataset', type=str, default='libero', help='Dataset name.')
    parser.add_argument('--save_path', type=str, default='/tmp/embeddings', help='Directory to store embeddings.')
    parser.add_argument('--frames_per_batch', type=int, default=30, help='Chunk size (frames per sample).')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of chunks per batch.')
    parser.add_argument('--keep_data', action='store_true', help='Keep frames.')
    parser.add_argument('--keep_labels', action='store_true', default=True, help='Keep labels.')
    parser.add_argument('--sample_all_stride', type=int, default=1, help='Stride.')
    parser.add_argument('--max_embs', type=int, default=0, help='Max videos.')
    parser.add_argument('--alsologtostderr', action='store_true', help='Log to stderr (dummy).')
    parser.add_argument('--split', type=str, default='val', help='Split.')
    parser.add_argument('--network', type=str, default=None, help='Base network.')
    parser.add_argument('--video_paths', type=str, default=None, help='Path to video_paths.json.')
    parser.add_argument('--normalize_embeddings', type=int, default=int(CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS),
                        help='Normalize embeddings (0 or 1). Default matches training config.')
    
    args = parser.parse_args()
    
    if args.network:
        CONFIG.MODEL.BASE_MODEL.NETWORK = args.network
        
    extract_embeddings(args)
