# coding=utf-8
"""Training code based on PyTorch."""

import os
import json
import re
import torch
import deepspeed
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from transformers import AutoProcessor

from algos.alignment import Alignment
from config import CONFIG
from datasets import create_dataset
from utils import restore_ckpt, setup_train_dir, save_checkpoint, to_dict, log_and_save_high_loss_samples

flags.DEFINE_string('logdir', '/tmp/alignment_logs', 'Path to logs.')
flags.DEFINE_string('resume_dir', None, 'Path to checkpoint directory to resume from. If None, starts from scratch.')
flags.DEFINE_string('pretrain_weights', None,
    'Path to a single pytorch state-dict file (.pt / .bin) for warm-starting the '
    'model. Loaded with strict=False. '
    'For ZeRO-3 sharded checkpoints consolidate first with: '
    'python zero_to_fp32.py <ckpt_dir> pytorch_model.bin')
flags.DEFINE_boolean(
    'force_train', False, 'Continue with training even when '
    'train_logs exist. Useful if one has to resume training. '
    'By default switched off to prevent overwriting existing '
    'experiments.')
flags.DEFINE_string('network', 'Qwen3-VL-Embedding-8B', 'Base network to use (must contain "Qwen3-VL").')
flags.DEFINE_string('video_paths', None, 'Comma-separated list of paths to video_paths.json.')
flags.DEFINE_integer('local_rank', -1, 'Local rank for distributed training')
flags.DEFINE_integer('gradient_accumulation_steps', 1, 'Gradient accumulation steps.')
flags.DEFINE_string('ds_config', 'scripts/ds_config.json', 'Path to DeepSpeed config json.')
flags.DEFINE_integer('save_interval', None, 'Number of steps between saving checkpoints.')
flags.DEFINE_integer('max_iters', None, 'Total number of training steps.')
flags.DEFINE_integer('num_align_frames', None, 'Number of frames to use for alignment.')

FLAGS = flags.FLAGS

def train(argv):
  """Trains model and evaluates on relevant downstream tasks."""
  if FLAGS.save_interval is not None:
      CONFIG.CHECKPOINT.SAVE_INTERVAL = FLAGS.save_interval
  
  if FLAGS.max_iters is not None:
      CONFIG.DS_CONFIG.scheduler.params.total_num_steps = FLAGS.max_iters
      CONFIG.TRAIN.MAX_ITERS = FLAGS.max_iters

  if FLAGS.num_align_frames is not None:
      CONFIG.TRAIN.NUM_ALIGN_FRAMES = FLAGS.num_align_frames

  if FLAGS.network:
      CONFIG.MODEL.BASE_MODEL.NETWORK = FLAGS.network

  deepspeed.init_distributed()
  local_rank = int(os.environ.get("LOCAL_RANK", -1))
  torch.cuda.set_device(local_rank)
  device = torch.device('cuda', local_rank)
  
  # Use global rank to determine master (only one process across all nodes)
  # local_rank is 0 for the first GPU on EACH node, so using it for is_master 
  # causes multiple wandb runs in multi-node training.
  is_master = (torch.distributed.get_rank() == 0)

  CONFIG.LOGDIR = FLAGS.logdir
  logdir = CONFIG.LOGDIR
  
  if is_master:
      setup_train_dir(logdir)
      import wandb
      wandb.init(project=os.environ.get("WANDB_PROJECT", "tcc_experiment"), 
                 dir=logdir,
                 config=to_dict(CONFIG))

  if is_master:
      logging.info('Using device: %s', device)

  algo = Alignment()
  # DeepSpeed will handle moving the model to device

  # Warm-start from a single pytorch state-dict file (strict=False).
  # MUST be done BEFORE deepspeed.initialize() so that DeepSpeed's fp32 master copy
  # is built from the pretrained weights, not from random initialization.
  # (With bf16 training, deepspeed.initialize() creates fp32_master = copy(bf16_params).
  #  If load_state_dict were called AFTER initialize(), the fp32_master would remain
  #  random, and after the first optimizer step it would overwrite the pretrained bf16
  #  weights — causing loss to jump to ~1.0 from step 2 onwards.)
  if FLAGS.pretrain_weights:
      if is_master:
          logging.info(f"Loading pretrain weights (strict=False) from: {FLAGS.pretrain_weights}")
      raw = torch.load(FLAGS.pretrain_weights, map_location='cpu')
      # DeepSpeed ZeRO-2 consolidated files wrap the state dict under 'module'
      state_dict = raw.get('module', raw)
      missing, unexpected = algo.load_state_dict(state_dict, strict=False)
      if is_master:
          if missing:
              logging.warning(f"pretrain_weights: missing keys: {missing[:10]}")
          if unexpected:
              logging.warning(f"pretrain_weights: unexpected extra keys: {unexpected[:10]}")
      del raw, state_dict  # free CPU memory before DeepSpeed moves model to GPU

  ds_config_path = FLAGS.ds_config
  if is_master:
      logging.info(f"Loading DeepSpeed config from: {ds_config_path}")
      
  with open(ds_config_path, 'r') as f:
      ds_config = json.load(f)
  
  if FLAGS.gradient_accumulation_steps != ds_config.get('gradient_accumulation_steps', 1):
      ds_config['gradient_accumulation_steps'] = FLAGS.gradient_accumulation_steps
      if is_master:
          logging.info(f"Overriding gradient_accumulation_steps to {FLAGS.gradient_accumulation_steps}")

  if is_master:
      logging.info("Overriding DeepSpeed config with values from config.py")

  # We only update keys that exist in CONFIG.DS_CONFIG
  # This allows ds_config.json to keep infrastructure settings (ZeRO, fp16, etc.)
  # while config.py controls training hyperparameters.
  
  def update_dict(d, u):
      for k, v in u.items():
          if isinstance(v, dict):
              d[k] = update_dict(d.get(k, {}), v)
          else:
              d[k] = v
      return d

  if 'DS_CONFIG' in CONFIG:
      ds_overrides = to_dict(CONFIG.DS_CONFIG)
      
      # Special handling for scheduler and optimizer: 
      # If defined in config, replace entirely because params depend on type.
      # Recursive merge would mix params from different types (e.g. WarmupDecayLR vs WarmupCosineLR).
      if 'scheduler' in ds_overrides:
          ds_config['scheduler'] = ds_overrides.pop('scheduler')
      
      if 'optimizer' in ds_overrides:
          ds_config['optimizer'] = ds_overrides.pop('optimizer')

      ds_config = update_dict(ds_config, ds_overrides)

  if is_master:
      logging.info(f"Final DeepSpeed Config: {json.dumps(ds_config, indent=2)}")

  if is_master:
      logging.info(f"Final DeepSpeed Config: {json.dumps(ds_config, indent=2)}")

  model_engine, optimizer, _, scheduler = deepspeed.initialize(
      args=FLAGS,
      model=algo,
      model_parameters=algo.parameters(),
      config=ds_config
  )
  
  load_path = None
  if FLAGS.resume_dir:
      if is_master:
          logging.info(f"Attempting to resume from {FLAGS.resume_dir}...")
      load_path, client_state = model_engine.load_checkpoint(FLAGS.resume_dir)
  
  if load_path is not None:
      # client_state is the dictionary we passed to save_checkpoint (if any)
      # DeepSpeed automatically restores model, optimizer, scheduler
      if is_master:
          logging.info(f"Resumed from checkpoint: {load_path}")
      
      if client_state and 'step' in client_state:
          global_step = client_state['step']
      else:
          # Fallback to parsing the tag/path
          try:
              # Handle potential trailing slash or full path issues
              # load_path might be .../1000 or .../global_step1000 or .../1000/
              path_clean = load_path.rstrip('/')
              tag = os.path.basename(path_clean)
              
              if tag.isdigit():
                  global_step = int(tag)
              elif tag.startswith('global_step'):
                  global_step = int(tag.split('global_step')[-1])
              else:
                   match = re.search(r'(\d+)$', tag)
                   if match:
                       global_step = int(match.group(1))
                   else:
                       if is_master:
                           logging.warning(f"Could not parse global_step from tag: {tag}")
                       global_step = 0
          except Exception as e:
              if is_master:
                  logging.warning(f"Error parsing global_step: {e}")
              global_step = 0
          
      if is_master:
          logging.info(f"Resumed global_step: {global_step}")
  else:
      global_step = 0
      if is_master:
          if FLAGS.resume_dir:
              logging.warning(f"Resume directory provided ({FLAGS.resume_dir}) but no checkpoint found/loaded.")
          logging.info("Starting from scratch.")

  processor = None
  if 'Qwen' in CONFIG.MODEL.BASE_MODEL.NETWORK:
      try:
          # Use local path for Qwen3-VL-Embedding-8B
          # TODO(open-source): internal-cluster path, load-bearing for the verified
          # training run — do not remove without re-verifying end-to-end. Same fix as
          # models.py (env var + public HF fallback). See OPEN_SOURCE_PATH_TODOS.md.
          model_name = '/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B'
          if is_master:
              logging.info(f"Loading processor for {model_name}")
          processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
      except Exception as e:
          if is_master:
              logging.warning(f"Could not load processor for Qwen: {e}")

  if getattr(CONFIG.JOINTS, 'USE_JOINTS', False) and processor is not None:
      num_bins = CONFIG.JOINTS.NUM_BINS
      joint_tokens = [f"<|joint_{i}|>" for i in range(num_bins)]
      num_added = processor.tokenizer.add_tokens(joint_tokens, special_tokens=True)
      if is_master:
          logging.info(
              f"[joints] Added {num_added} joint tokens to tokenizer "
              f"(IDs {processor.tokenizer.convert_tokens_to_ids(joint_tokens[0])}"
              f"–{processor.tokenizer.convert_tokens_to_ids(joint_tokens[-1])})"
          )
      # No resize needed: 151669+192=151861 < vocab_size=151936

  train_loader = create_dataset('train', mode='train',
                                batch_size=model_engine.train_micro_batch_size_per_gpu(),
                                return_iterator=True,
                                distributed=True, # DeepSpeed is always distributed
                                video_paths_json=FLAGS.video_paths,
                                processor=processor)

  max_iters = CONFIG.TRAIN.MAX_ITERS
  
  if is_master:
      logging.info('Starting training...')
      # Fix: Set smoothing=0.0 to see instantaneous loss
      pbar = tqdm(total=max_iters, initial=global_step, dynamic_ncols=True, smoothing=0.0)

  current_step = global_step
  for data in train_loader:
      if current_step >= max_iters:
          break
          
      steps = data['chosen_steps']
      seq_lens = data['seq_lens']

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

      embs = model_engine(data, steps, seq_lens, training=True)

      loss, loss_dict = model_engine.module.compute_loss(embs, steps, seq_lens, current_step,
                                        training=True, frame_labels=data.get('frame_labels'),
                                        seq_labels=data.get('seq_labels'),
                                        metadata=data)

      if is_master:
          log_and_save_high_loss_samples(logdir, current_step, loss_dict, data)

      model_engine.backward(loss)

      model_engine.step()
      if model_engine.is_gradient_accumulation_boundary():
          current_step += 1
          if is_master:
              pbar.update(1)

      # DeepSpeed handles scheduler.step() automatically

      if is_master:
          pbar.set_description(f"Step {current_step}/{max_iters}")
          
          postfix = {'loss': loss.item(), 'lr': model_engine.get_lr()[0]}
          pbar.set_postfix(postfix)
      
      if is_master and current_step % CONFIG.LOGGING.REPORT_INTERVAL == 0:
          log_dict = {'loss': loss.item(), 'lr': model_engine.get_lr()[0]}

          if loss_dict:
              for k, v in loss_dict.items():
                  # Skip logging large tensors (like per_sample_loss or alignment_indices) to wandb scalars
                  if k in CONFIG.LOGGING.EXCLUDE_KEYS:
                      continue
                  
                  if isinstance(v, torch.Tensor):
                      if v.numel() == 1:
                          log_dict[k] = v.item()
                  else:
                      log_dict[k] = v
                      
          grad_norm = model_engine.get_global_grad_norm()
          if grad_norm is not None:
              if isinstance(grad_norm, torch.Tensor):
                  grad_norm = grad_norm.item()
              log_dict['grad_norm'] = grad_norm
          wandb.log(log_dict, step=current_step)
          
      if current_step % CONFIG.CHECKPOINT.SAVE_INTERVAL == 0 and current_step > 0:
          client_state = {'step': current_step}
          model_engine.save_checkpoint(logdir, current_step, client_state=client_state)

          # Save Qwen model in HF format for easy loading
          if 'Qwen' in CONFIG.MODEL.BASE_MODEL.NETWORK and is_master:
              save_path = os.path.join(logdir, f"hf_checkpoint-{current_step}")
              try:
                  # Access the underlying Qwen model
                  # model_engine -> Algorithm -> BaseModel -> Qwen
                  # Note: Depending on DeepSpeed config, we might need to unwrap differently
                  if hasattr(model_engine.module, 'model') and hasattr(model_engine.module.model, 'base_model'):
                      qwen_model = model_engine.module.model.base_model
                      qwen_model.save_pretrained(save_path)
                      if processor is not None:
                          processor.save_pretrained(save_path)
                      logging.info(f"Saved Qwen model in HF format to {save_path}")
              except Exception as e:
                  logging.error(f"Failed to save Qwen model in HF format: {e}")

  if is_master:
      pbar.close()


if __name__ == '__main__':
  app.run(train)
