# coding=utf-8
"""Evaluation code based on train.py - exact same data loading and forward pass."""

import os
import json
import torch
import deepspeed
from multiprocessing import Manager
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from transformers import AutoProcessor

from algorithms import get_algo
from config import CONFIG, apply_eval_overrides
from datasets import create_dataset
from utils import log_and_save_high_loss_samples

flags.DEFINE_string('logdir', None, 'Path to save evaluation results.')
flags.DEFINE_string('resume_dir', None, 'Path to checkpoint directory to load model from.')
flags.DEFINE_string('network', 'Qwen3-VL-2B', 'Base network to use (must contain "Qwen3-VL").')
flags.DEFINE_string('video_paths', None, 'Comma-separated list of paths to video_paths.json.')
flags.DEFINE_integer('local_rank', -1, 'Local rank for distributed training')
flags.DEFINE_string('ds_config', 'scripts/ds_config_zero3.json', 'Path to DeepSpeed config json.')
flags.DEFINE_string('eval_chunk_probs', None, 'Chunk probabilities for eval [1x,2x,4x] (e.g., "0,0,1" for 4x multiplier).')
flags.DEFINE_integer('batch_size', None, 'Batch size for evaluation.')
flags.DEFINE_integer('num_iters', None, 'Number of evaluation iterations. If None, evaluates entire dataset.')
flags.DEFINE_integer('num_align_frames', None, 'Number of frames to use for alignment. Must match training.')
flags.DEFINE_boolean(
    'eval_contiguous_shard',
    False,
    'If True, each rank processes a contiguous block of dataset indices (overrides CONFIG.EVAL.CONTIGUOUS_DISTRIBUTED_SHARD when True).',
)

FLAGS = flags.FLAGS

def evaluate(argv):
  """Evaluates model using the same logic as training."""

  # Apply all eval-mode config defaults (augmentation off, logging tweaks).
  # FLAGS-based overrides below take priority over anything set here.
  apply_eval_overrides()

  # FLAGS-based overrides (highest priority)
  if FLAGS.network:
      CONFIG.MODEL.BASE_MODEL.NETWORK = FLAGS.network

  if FLAGS.eval_chunk_probs:
      probs = [float(x) for x in FLAGS.eval_chunk_probs.split(',')]
      CONFIG.EVAL.CHUNK_PROBS = probs
      print(f"Override EVAL.CHUNK_PROBS to {probs}")

  if FLAGS.batch_size:
      CONFIG.EVAL.BATCH_SIZE = FLAGS.batch_size
      print(f"Override EVAL.BATCH_SIZE to {FLAGS.batch_size}")

  if FLAGS.num_align_frames is not None:
      CONFIG.TRAIN.NUM_ALIGN_FRAMES = FLAGS.num_align_frames
      print(f"Override NUM_ALIGN_FRAMES to {FLAGS.num_align_frames}")

  # DeepSpeed Setup
  deepspeed.init_distributed()
  local_rank = int(os.environ.get("LOCAL_RANK", -1))
  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()
  
  torch.cuda.set_device(local_rank)
  device = torch.device('cuda', local_rank)
  
  is_master = (rank == 0)
  
  if is_master:
      print(f"Multi-GPU evaluation: Using {world_size} GPUs")
  print(f"[Rank {rank}] Initialized")

  # Setup log directory
  if FLAGS.logdir:
      logdir = FLAGS.logdir
  elif FLAGS.resume_dir:
      # If no logdir specified, save results in resume_dir
      logdir = FLAGS.resume_dir
  else:
      raise ValueError("Either --logdir or --resume_dir must be specified.")
  
  os.makedirs(logdir, exist_ok=True)
  
  if is_master:
      print(f"Loading checkpoint from: {FLAGS.resume_dir}")
      print(f"Saving evaluation results to: {logdir}")

  algo = get_algo(CONFIG.TRAINING_ALGO)

  # Setup DeepSpeed (for loading checkpoint and inference)
  ds_config_path = FLAGS.ds_config
  if is_master:
      logging.info(f"Loading DeepSpeed config from: {ds_config_path}")
      
  with open(ds_config_path, 'r') as f:
      ds_config = json.load(f)
  
  # Modify DS config for inference
  ds_config['train_batch_size'] = CONFIG.EVAL.BATCH_SIZE * world_size
  ds_config['train_micro_batch_size_per_gpu'] = CONFIG.EVAL.BATCH_SIZE
  ds_config['gradient_accumulation_steps'] = 1
  
  # Disable optimizer for inference
  if 'optimizer' in ds_config:
      del ds_config['optimizer']
  if 'scheduler' in ds_config:
      del ds_config['scheduler']
  
  # For inference, we can disable ZeRO-3 to avoid parameter sharding issues
  # This allows loading checkpoint from different world size
  if 'zero_optimization' in ds_config and ds_config['zero_optimization'].get('stage', 0) == 3:
      if is_master:
          logging.info("Disabling ZeRO-3 for inference (setting stage=0)")
      ds_config['zero_optimization']['stage'] = 0

  if is_master:
      logging.info(f"Evaluation DeepSpeed Config: {json.dumps(ds_config, indent=2)}")

  # Load checkpoint before DeepSpeed initialization (to avoid ZeRO world size mismatch)
  if not FLAGS.resume_dir:
      raise ValueError("--resume_dir must be specified for evaluation.")
  
  # Try to load PyTorch model file directly
  checkpoint_files = [
      os.path.join(FLAGS.resume_dir, 'checkpoint.pth.tar'),
      os.path.join(FLAGS.resume_dir, 'fp32_converted', 'pytorch_model.bin'),
      os.path.join(FLAGS.resume_dir, 'pytorch_model.bin'),
  ]
  
  checkpoint_path = None
  for path in checkpoint_files:
      if os.path.exists(path):
          checkpoint_path = path
          break
  
  if checkpoint_path is None:
      raise ValueError(f"No checkpoint file found in {FLAGS.resume_dir}. Tried: {checkpoint_files}")
  
  if is_master:
      logging.info(f"Loading checkpoint from: {checkpoint_path}")
  
  # Load checkpoint
  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  
  # Extract state dict
  if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
      global_step = checkpoint.get('global_step', 0)
  elif 'model_state_dict' in checkpoint:
      state_dict = checkpoint['model_state_dict']
      global_step = checkpoint.get('global_step', 0)
  else:
      # Assume the checkpoint is the state dict itself
      state_dict = checkpoint
      global_step = 0
  
  if is_master:
      logging.info(f"Checkpoint global_step: {global_step}")
  
  # Load state dict into model
  missing_keys, unexpected_keys = algo.load_state_dict(state_dict, strict=False)
  if is_master:
      if missing_keys:
          logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
      if unexpected_keys:
          logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
      logging.info("Successfully loaded model weights")
  
  # Initialize DeepSpeed for inference (without loading checkpoint)
  model_engine, _, _, _ = deepspeed.initialize(
      args=FLAGS,
      model=algo,
      model_parameters=algo.parameters(),
      config=ds_config
  )

  # Initialize Processor for Qwen
  processor = None
  if 'Qwen' in CONFIG.MODEL.BASE_MODEL.NETWORK:
      try:
          # TODO(open-source): internal-cluster path; eval-only, not exercised by the
          # verified training run. Same planned fix as models.py/train.py. See
          # OPEN_SOURCE_PATH_TODOS.md.
          model_name = '/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B'
          if is_master:
              logging.info(f"Loading processor for {model_name}")
          processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
      except Exception as e:
          if is_master:
              logging.warning(f"Could not load processor for Qwen: {e}")

  # Setup Dataset - use eval mode with distributed sampling
  contiguous_shard = bool(CONFIG.EVAL.CONTIGUOUS_DISTRIBUTED_SHARD or FLAGS.eval_contiguous_shard)
  eval_loader = create_dataset('train', mode='eval',
                               batch_size=CONFIG.EVAL.BATCH_SIZE,
                               return_iterator=False,  # Don't use infinite iterator
                               distributed=True,  # Enable distributed sampling for multi-GPU
                               video_paths_json=FLAGS.video_paths,
                               processor=processor,
                               contiguous_distributed_eval=contiguous_shard)

  # Set model to eval mode
  model_engine.eval()

  # --- Shared ref-cache index: lets DataLoader workers know which refs are cached ---
  # Keys = same int hashes used by RefEmbeddingCache (zlib.crc32); values = True.
  # Workers read this dict to skip ref image loading on cache hit.
  _mgr = Manager()
  _ref_cache_index = _mgr.dict()
  model_engine.module._ref_cache.index = _ref_cache_index      # sync puts/evicts → index
  if hasattr(eval_loader, 'collate_fn'):
      eval_loader.collate_fn._ref_cache_index = _ref_cache_index  # worker can read
  
  # Determine number of iterations
  if FLAGS.num_iters:
      num_iters = FLAGS.num_iters
  else:
      num_iters = len(eval_loader)
  
  # Each rank processes a subset of data
  if contiguous_shard and hasattr(eval_loader.sampler, '_start'):
      s = eval_loader.sampler
      print(f"[Rank {rank}] Contiguous shard: dataset indices [{s._start}, {s._end}) ({len(eval_loader)} batches)")
  print(f"[Rank {rank}] Dataset: {len(eval_loader.dataset)} samples, Batch size: {CONFIG.EVAL.BATCH_SIZE}, Batches per rank: {len(eval_loader)}")
  print(f"[Rank {rank}] Will evaluate {num_iters} batches")
  
  if is_master:
      pbar = tqdm(total=num_iters, dynamic_ncols=True, desc=f"Rank {rank}")
  
  avg_loss = 0.0
  
  with torch.no_grad():  # No gradients needed for evaluation
      for i, data in enumerate(eval_loader):
          if i >= num_iters:
              break
          
          steps = data['chosen_steps']
          seq_lens = data['seq_lens']
          
          # Data preparation (same as training)
          for k, v in data.items():
              if isinstance(v, torch.Tensor):
                  data[k] = v.to(device)
                  if model_engine.fp16_enabled() and data[k].dtype == torch.float32:
                      data[k] = data[k].half()
                  elif model_engine.bfloat16_enabled() and data[k].dtype == torch.float32:
                      data[k] = data[k].to(torch.bfloat16)
          
          if 'ref_chosen_steps' in data and 'ref_seq_lens' in data:
              steps = torch.cat([data['chosen_steps'], data['ref_chosen_steps']], dim=0)
              seq_lens = torch.cat([data['seq_lens'], data['ref_seq_lens']], dim=0)
          
          steps = steps.to(device)
          seq_lens = seq_lens.to(device)
          
          # Forward pass (same as training)
          embs = model_engine(data, steps, seq_lens, training=False)
          
          # Compute loss (same as training)
          loss, loss_dict = model_engine.module.compute_loss(embs, steps, seq_lens, global_step,
                                            training=False, frame_labels=data.get('frame_labels'),
                                            seq_labels=data.get('seq_labels'),
                                            metadata=data)
          
          avg_loss += loss.item()
          
          # Save all samples during evaluation - each rank saves to its own file
          log_and_save_high_loss_samples(logdir, global_step, loss_dict, data, save_all=True, rank=rank)
          
          if is_master:
              pbar.update(1)
              cache = model_engine.module._ref_cache
              last = "HIT" if cache.last_hit else "MISS"
              _, _, rate = cache.stats()
              pbar.set_postfix(loss=loss.item(), avg_loss=avg_loss/(i+1), cache=f"{last}|{rate:.2f}")
  
  avg_loss /= num_iters
  
  # Gather average loss from all ranks
  avg_loss_tensor = torch.tensor([avg_loss], device=device)
  torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
  global_avg_loss = avg_loss_tensor.item() / world_size
  
  if is_master:
      pbar.close()
  
  print(f"[Rank {rank}] Local average loss: {avg_loss:.4f}")
  print(f"[Rank {rank}] Results saved to: {logdir}/high_loss_samples_{rank}.jsonl")
  
  # Wait for all ranks to finish
  torch.distributed.barrier()

  # Ref cache stats
  if hasattr(model_engine.module, '_ref_cache'):
      cache = model_engine.module._ref_cache
      hits, misses, rate = cache.stats()
      print(f"[Rank {rank}] Ref cache: hits={hits}, misses={misses}, hit_rate={rate:.3f}")

  if is_master:
      print(f"\n=== Evaluation Complete ===")
      print(f"Global Average Loss: {global_avg_loss:.4f}")
      print(f"Results saved to:")
      for r in range(world_size):
          print(f"  - {logdir}/high_loss_samples_{r}.jsonl")
      
      # Optionally merge all jsonl files
      print(f"\nMerging results from all ranks...")
      merged_file = os.path.join(logdir, "high_loss_samples_all.jsonl")
      with open(merged_file, 'w') as outf:
          for r in range(world_size):
              rank_file = os.path.join(logdir, f"high_loss_samples_{r}.jsonl")
              if os.path.exists(rank_file):
                  with open(rank_file, 'r') as inf:
                      outf.write(inf.read())
      print(f"Merged results saved to: {merged_file}")


if __name__ == '__main__':
  app.run(evaluate)
