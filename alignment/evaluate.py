# coding=utf-8
"""Evaluate embeddings on downstream tasks."""

import os
import argparse
import torch
import numpy as np
import yaml
from config import CONFIG
from datasets import create_dataset
from models import get_model
from utils import restore_ckpt, log_and_save_high_loss_samples
from algos.algorithm import Algorithm
from algos.alignment import Alignment
from transformers import AutoProcessor
# from algos.deterministic_alignment import DeterministicAlignment # If it exists

def get_algo(algo_name):
    if algo_name == 'alignment':
        return Alignment
    # elif algo_name == 'deterministic_alignment':
    #     return DeterministicAlignment
    else:
        raise ValueError(f"Unknown algorithm {algo_name}")

def evaluate(args):
    # Only evaluate on rank 0 when using distributed training
    # torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE env vars
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank != 0:
        print(f"LOCAL_RANK={local_rank}: Skipping evaluation (only rank 0 evaluates)")
        return
    
    # resume_dir: where to load checkpoint from
    # logdir: where to save evaluation results (if not specified, use resume_dir)
    resume_dir = args.resume_dir if args.resume_dir else args.logdir
    logdir = args.logdir if args.logdir else args.resume_dir
    
    if not resume_dir or not os.path.exists(resume_dir):
        raise ValueError(f"Resume directory {resume_dir} does not exist.")
    
    # Create logdir if it doesn't exist (for saving evaluation results)
    if logdir and not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
        print(f"Created evaluation output directory: {logdir}")
    
    print(f"Loading checkpoint from: {resume_dir}")
    print(f"Saving evaluation results to: {logdir}")
    
    # Apply command line overrides to CONFIG
    if args.eval_chunk_probs:
        probs = [float(x) for x in args.eval_chunk_probs.split(',')]
        CONFIG.EVAL.CHUNK_PROBS = probs
        print(f"Override EVAL.CHUNK_PROBS to {probs}")
    
    if args.batch_size:
        CONFIG.EVAL.BATCH_SIZE = args.batch_size
        print(f"Override EVAL.BATCH_SIZE to {args.batch_size}")
    
    # Enable high loss sample saving during evaluation
    CONFIG.LOGGING.DEBUG_STEP_START = 0
    print(f"Set DEBUG_STEP_START to 0 for evaluation")
        
    # Setup Config (load from resume_dir)
    config_path = os.path.join(resume_dir, 'config.yml')
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
            # Only update meaningful keys that might have been changed
            if 'MODEL' in saved_config: update_config(CONFIG.MODEL, saved_config['MODEL'])
            if 'DATA' in saved_config: update_config(CONFIG.DATA, saved_config['DATA'])
            if 'TRAINING_ALGO' in saved_config: CONFIG.TRAINING_ALGO = saved_config['TRAINING_ALGO']

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

    # Model
    model = get_model()
    # if torch.cuda.is_available():
    #     model = model.cuda()
        
    # Restore
    # restore_ckpt expects model, optimizer, scheduler.
    # But here we only need model.
    # However, restore_ckpt loads state_dict into model.
    # If checkpoint has 'state_dict' key which is algo.state_dict(), we should pass algo.
    # But we haven't created algo yet.
    # Let's create algo first.
    
    # Algorithm
    AlgoClass = get_algo(CONFIG.TRAINING_ALGO)
    # AlgoClass expects model dict or module.
    # get_model returns a dict {'cnn': ..., 'emb': ...}
    algo = AlgoClass(model)
    if torch.cuda.is_available():
        algo = algo.cuda()

    start_epoch, global_step = restore_ckpt(resume_dir, algo)
    
    # Convert gate module to match Qwen model dtype
    if hasattr(algo, 'gate') and 'Qwen' in CONFIG.MODEL.BASE_MODEL.NETWORK:
        # Get dtype from Qwen model
        if hasattr(algo, 'cnn') and hasattr(algo.cnn, 'base_model'):
            qwen_dtype = next(algo.cnn.base_model.parameters()).dtype
            print(f"Converting gate module to {qwen_dtype} to match Qwen output dtype...")
            algo.gate = algo.gate.to(qwen_dtype)
        
    # Datasets
    # Note: We only use one split for evaluation since we don't have actual train/val split
    eval_loader = create_dataset('train', mode='eval', batch_size=CONFIG.EVAL.BATCH_SIZE, return_iterator=False, video_paths_json=args.video_paths, processor=processor)
    
    # Evaluation Loop (AlgoLoss)
    print(f"Evaluating at step {global_step}...")
    print(f"Dataset: {len(eval_loader.dataset)} samples, Batch size: {eval_loader.batch_size}, Expected batches: {len(eval_loader)}")
    
    for split in ['eval']:
        iterator = iter(eval_loader)
        avg_loss = 0.0
        num_iters = min(CONFIG.EVAL.VAL_ITERS, len(eval_loader)) if hasattr(eval_loader, '__len__') else CONFIG.EVAL.VAL_ITERS
        
        for i in range(num_iters):
            data = next(iterator)
                
            # Prepare data
            steps = data['chosen_steps']
            seq_lens = data['seq_lens']
            
            # Evaluate
            loss, loss_dict = algo.evaluate_one_iter(data, steps, seq_lens, global_step)
            avg_loss += loss.item()
            
            # Save all samples during evaluation
            log_and_save_high_loss_samples(logdir, global_step, loss_dict, data, save_all=True)
            
            if (i + 1) % 5 == 0 or i == num_iters - 1:
                print(f"  Progress: {i+1}/{num_iters} batches, avg loss so far: {avg_loss/(i+1):.4f}")
            
        avg_loss /= num_iters
        print(f"Iter[{global_step}] {split} loss: {avg_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=None, help='Path to logs.')
    parser.add_argument('--resume_dir', type=str, default=None, help='Path to checkpoint directory (alias for --logdir).')
    parser.add_argument('--visualize', action='store_true', help='Visualize.')
    parser.add_argument('--max_embs', type=int, default=0, help='Max embs.')
    parser.add_argument('--nocontinuous_eval', action='store_true', help='No continuous eval.')
    parser.add_argument('--alsologtostderr', action='store_true', help='Log to stderr (dummy).')
    parser.add_argument('--network', type=str, default='Resnet50_pretrained', help='Base network.')
    parser.add_argument('--video_paths', type=str, default=None, help='Path to video_paths.json.')
    parser.add_argument('--eval_chunk_probs', type=str, default=None, 
                        help='Chunk probabilities for eval [1x,2x,4x] (e.g., "0,0,1" for 4x multiplier).')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for evaluation.')
    
    args = parser.parse_args()
    
    if not args.logdir and not args.resume_dir:
        parser.error("Either --logdir or --resume_dir must be specified.")
    
    if args.network:
        CONFIG.MODEL.BASE_MODEL.NETWORK = args.network
        
    evaluate(args)
