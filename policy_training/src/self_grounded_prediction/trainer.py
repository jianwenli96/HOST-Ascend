import logging
import json
import inspect
import math
import os
import re
import textwrap
from math import ceil
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class PerGroupCosineAnnealingLR(CosineAnnealingLR):
    """CosineAnnealingLR with per-param-group `eta_min`.

    PyTorch's stock `CosineAnnealingLR.eta_min` is a single scalar shared by
    every param group, which collapses the lr ratio between groups (e.g. base
    DiT vs visual encoder with `lr_scale=10`) to 1.0 at the end of cosine.
    This subclass accepts `eta_min` as either a scalar (broadcast) or a list
    of length `len(optimizer.param_groups)`, and computes lr in closed form
    against each group's `base_lr` so the per-group ratio is preserved across
    the entire schedule (lr_t = eta_min + (base_lr - eta_min) * cos_factor).

    State dict adds `_per_group_eta_min`. Loading a state dict from the stock
    scalar-only scheduler is supported: the scalar is broadcast.
    """

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        if isinstance(eta_min, (list, tuple)):
            if len(eta_min) != len(optimizer.param_groups):
                raise ValueError(
                    f"`eta_min` list length {len(eta_min)} does not match "
                    f"len(param_groups)={len(optimizer.param_groups)}"
                )
            eta_min_list = [float(x) for x in eta_min]
        else:
            eta_min_list = [float(eta_min)] * len(optimizer.param_groups)
        # Must be set BEFORE super().__init__: the parent constructor calls
        # `self.step()` which calls our overridden `get_lr()`.
        self._per_group_eta_min = eta_min_list
        super().__init__(
            optimizer, T_max=T_max, eta_min=eta_min_list[0], last_epoch=last_epoch
        )

    def get_lr(self):
        t = max(min(int(self.last_epoch), int(self.T_max)), 0)
        cos_t = 0.5 * (1.0 + math.cos(math.pi * t / max(int(self.T_max), 1)))
        return [
            eta_min + (base_lr - eta_min) * cos_t
            for base_lr, eta_min in zip(self.base_lrs, self._per_group_eta_min)
        ]

    def state_dict(self):
        sd = super().state_dict()
        sd["_per_group_eta_min"] = list(self._per_group_eta_min)
        return sd

    def load_state_dict(self, state_dict):
        eta_min_list = state_dict.pop("_per_group_eta_min", None)
        super().load_state_dict(state_dict)
        if eta_min_list is None:
            n_groups = len(self.optimizer.param_groups)
            self._per_group_eta_min = [float(self.eta_min)] * n_groups
        else:
            self._per_group_eta_min = [float(x) for x in eta_min_list]


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, extra_val_datasets=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # extra_val_datasets: dict[str, Dataset] | None — named extra val sets, e.g. {"seen": ds}
        self.extra_val_datasets = extra_val_datasets if extra_val_datasets is not None else {}
        self.cfg = cfg
        # Per-action-key slices for per-key loss logging (training + eval)
        self.action_component_slices = (
            train_dataset.get_action_component_slices()
            if hasattr(train_dataset, 'get_action_component_slices')
            else None
        ) or None  # treat empty dict as None
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # Staged unfreeze: gradually unfreeze params when switching action representation.
        _su = cfg.get("staged_unfreeze", None)
        if _su is not None and _su.get("enabled", False):
            self._staged_unfreeze_enabled = True
            self._stage2_step = int(_su.stage2_step)
            self._stage3_step = int(_su.stage3_step)
            if self._stage2_step >= self._stage3_step:
                raise ValueError(
                    f"staged_unfreeze.stage2_step ({self._stage2_step}) must be < "
                    f"stage3_step ({self._stage3_step})"
                )
        else:
            self._staged_unfreeze_enabled = False
        self._current_unfreeze_stage = None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown") \
                if self.accelerator.state.deepspeed_plugin is not None else "N/A",
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)

        # Pre-load .pt weights BEFORE optimizer/DeepSpeed init so fp32 master
        # weights are built from the correct checkpoint, not from pretrained.
        self._pre_loaded_pt = False
        if self.resume:
            resume_path = Path(str(self.resume))
            if resume_path.is_file() and str(resume_path).endswith(".pt"):
                logger.info("Pre-loading .pt weights before accelerator.prepare(): %s", self.resume)
                self.model.load_checkpoint(str(resume_path), optimizer=None)
                self._pre_loaded_pt = True

        # Base param group: DiT + optional proprio/progress encoders/decoders.
        base_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            base_params.extend(list(proprio_encoder.parameters()))
        progress_encoder = getattr(self.model, "progress_encoder", None)
        if progress_encoder is not None:
            base_params.extend(list(progress_encoder.parameters()))
        progress_decoder = getattr(self.model, "progress_decoder", None)
        if progress_decoder is not None:
            base_params.extend(list(progress_decoder.parameters()))

        # Visual encoder gets its own param group with `learning_rate * lr_scale`.
        visual_encoder = getattr(self.model, "visual_encoder", None)
        ve_lr_scale = 1.0
        ve_cfg_block = (
            cfg.model.get("visual_encoder", None) if hasattr(cfg, "model") else None
        )
        if ve_cfg_block is not None and "lr_scale" in ve_cfg_block:
            ve_lr_scale = float(ve_cfg_block["lr_scale"])
        if visual_encoder is None and ve_lr_scale != 1.0:
            raise ValueError(
                f"`model.visual_encoder.lr_scale={ve_lr_scale}` is set but the "
                "model has no `visual_encoder` (set `model.visual_encoder.enabled=true`)."
            )

        param_groups = [
            {"params": base_params, "lr": self.learning_rate},
        ]
        if visual_encoder is not None:
            ve_params = list(visual_encoder.parameters())
            param_groups.append(
                {"params": ve_params, "lr": self.learning_rate * ve_lr_scale}
            )
            logger.info(
                "Optimizer param groups: base lr=%.3e (n=%d), visual_encoder lr=%.3e "
                "(n=%d, lr_scale=%g)",
                self.learning_rate,
                len(base_params),
                self.learning_rate * ve_lr_scale,
                len(ve_params),
                ve_lr_scale,
            )
        else:
            logger.info(
                "Optimizer param groups: base lr=%.3e (n=%d), no visual_encoder",
                self.learning_rate,
                len(base_params),
            )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        # WANDB_MODE (set by the launch script, e.g. to fall back to offline logging
        # when no WANDB_API_KEY is configured) must take precedence over the static
        # Hydra default `wandb.mode: online` — otherwise an explicit `mode=` kwarg
        # here would silently shadow the env var and wandb.init() would still try
        # an interactive login.
        env_wandb_mode = os.environ.get("WANDB_MODE")
        if env_wandb_mode is not None and env_wandb_mode != self.cfg.wandb.mode:
            logger.warning(
                "WANDB_MODE=%s in the environment overrides cfg.wandb.mode=%s",
                env_wandb_mode, self.cfg.wandb.mode,
            )
        wandb_mode = env_wandb_mode if env_wandb_mode is not None else self.cfg.wandb.mode

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=wandb_mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        collate_fn = getattr(dataset, 'collate_fn', None)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            timeout=60 if self.num_workers > 0 else 0,
            persistent_workers=self.num_workers > 0,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            min_lr_ratio = float(getattr(self.cfg, 'min_lr_ratio', 0.01))
            per_group_eta_min = [g["lr"] * min_lr_ratio for g in self.optimizer.param_groups]
            main_scheduler = PerGroupCosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=per_group_eta_min,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if self._pre_loaded_pt:
            logger.info(".pt weights were pre-loaded before prepare(); skipping duplicate load.")
            return
        raise RuntimeError(
            f"Resume path is a file but was not pre-loaded (expected .pt): {resume}"
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        progress_encoder = getattr(model, "progress_encoder", None)
        if progress_encoder is not None:
            progress_encoder.train()
            progress_encoder.requires_grad_(True)
        progress_decoder = getattr(model, "progress_decoder", None)
        if progress_decoder is not None:
            progress_decoder.train()
            progress_decoder.requires_grad_(True)
        visual_encoder = getattr(model, "visual_encoder", None)
        if visual_encoder is not None:
            visual_encoder.train()
            visual_encoder.requires_grad_(True)

    def _apply_staged_freeze(self, step: int):
        """Set requires_grad based on staged unfreeze schedule.

        Stage 1 [0, stage2_step):  action I/O layers only (action_encoder, head,
                                   time/text embeddings) + proprio/progress encoders.
        Stage 2 [stage2_step, stage3_step):  full action expert (+ proprio/progress).
        Stage 3 [stage3_step, inf):  everything (identical to _apply_dit_only_train_mode).
        """
        if not self._staged_unfreeze_enabled:
            return

        if step < self._stage2_step:
            stage = 1
        elif step < self._stage3_step:
            stage = 2
        else:
            stage = 3

        # Skip if already in this stage.
        if self._current_unfreeze_stage == stage:
            return
        self._current_unfreeze_stage = stage

        model = self.accelerator.unwrap_model(self.model)

        # Reset to standard dit-only-train baseline (unfreezes all of dit + encoders).
        self._apply_dit_only_train_mode(model)

        if stage == 1:
            # Freeze action DiT transformer blocks (keep action_encoder, head,
            # time_embedding, time_projection, text_embedding trainable).
            for name, param in model.dit.named_parameters():
                if name.startswith("mixtures.action.blocks."):
                    param.requires_grad_(False)
                elif name.startswith("mixtures.video."):
                    param.requires_grad_(False)
            ve = getattr(model, "visual_encoder", None)
            if ve is not None:
                ve.requires_grad_(False)
                ve.eval()
        elif stage == 2:
            # Freeze video expert + visual encoder only; full action expert trainable.
            for name, param in model.dit.named_parameters():
                if name.startswith("mixtures.video."):
                    param.requires_grad_(False)
            ve = getattr(model, "visual_encoder", None)
            if ve is not None:
                ve.requires_grad_(False)
                ve.eval()
        # stage 3: _apply_dit_only_train_mode already unfroze everything.

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Staged unfreeze: entering stage %d at step %d — %d / %d params trainable (%.1f%%)",
            stage, step, trainable, total, 100.0 * trainable / total,
        )

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        task_video = sample.get("task_video", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        # task_video: dataset returns [C, T, H, W]; add batch dim → [1, C, T, H, W]
        if task_video is not None:
            if not isinstance(task_video, torch.Tensor):
                raise TypeError(f"`sample['task_video']` must be a torch.Tensor, got {type(task_video)}")
            if task_video.ndim == 4:
                task_video = task_video.unsqueeze(0)
            if task_video.ndim != 5:
                raise ValueError(f"`sample['task_video']` must be [C,T,H,W] or [B,C,T,H,W], got {tuple(task_video.shape)}")

        # task_video_start / task_video_end: visualization-only indices from custom_collate_fn (lists)
        _tvs = sample.get("task_video_start")
        task_video_start = int(_tvs[0]) if isinstance(_tvs, (list, tuple)) else int(_tvs or 0)

        _tve = sample.get("task_video_end")
        if isinstance(_tve, (list, tuple)):
            _tve = _tve[0]
        task_video_end = int(_tve) if _tve is not None else None

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "task_video": task_video,
            "action_horizon": action_horizon,
            "task_video_start": task_video_start,
            "task_video_end": task_video_end,
            "task_video_dropped": sample.get("task_video_dropped", False),
            "agent_episode_dir": sample.get("agent_episode_dir"),
            "task_episode_dir": sample.get("task_episode_dir"),
            "progress_gt": sample.get("progress_gt", None).unsqueeze(0)
                if sample.get("progress_gt") is not None else None,
        }

    @torch.no_grad()
    def evaluate(self):
        return self._evaluate_dataset(self.val_dataset, name_suffix="")

    @torch.no_grad()
    def _evaluate_dataset(self, dataset, name_suffix: str):
        # Single-pass eval: use the sample as-is from dataset (respects drop probs).
        # Mode is detected from task_video_dropped + context_mask state.
        # name_suffix: appended to mp4 filename (e.g. "_seen") to distinguish extra val datasets.
        _METRIC_COLS = 11  # psnr_rg, ssim_rg, psnr_rd, ssim_rd, action_l2, action_l1, progress_l1, psnr_rg_oracle, ssim_rg_oracle, action_l2_oracle, action_l1_oracle

        if dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(dataset[eval_index])

        # 1. Training loss (mode-independent)
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]  # [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        task_video_raw = sample.get("task_video", None)
        task_video = task_video_raw[0] if task_video_raw is not None else None  # [C, T, H, W]
        task_video_start = int(sample.get("task_video_start") or 0)
        task_video_end_val = sample.get("task_video_end")
        task_video_end = int(task_video_end_val) if task_video_end_val is not None else None

        orig_context      = sample["context"][0]       if sample.get("context")      is not None else None
        orig_context_mask = sample["context_mask"][0]  if sample.get("context_mask") is not None else None

        # 2. VAE reconstruction metrics (mode-independent, compute once)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        gt_video_batch  = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents     = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)
        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )
        psnr_dg = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_dg = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        # 3. Action component slices (for per-key metrics, same across modes)
        action_slices = (
            dataset.get_action_component_slices()
            if hasattr(dataset, 'get_action_component_slices')
            else None
        ) or {}
        per_key_names = list(action_slices.keys())  # fixed order across all ranks

        # 4. Detect mode from dataset's drop decisions
        task_video_dropped_list = sample.get("task_video_dropped", [False])
        _tv_dropped = task_video_dropped_list[0] if isinstance(task_video_dropped_list, list) else bool(task_video_dropped_list)
        _ctx_is_zero = (orig_context_mask is not None and orig_context_mask.sum() == 0)

        if _tv_dropped:
            mode_name = "txt_only"
        elif _ctx_is_zero:
            mode_name = "tv_only"
        else:
            mode_name = "tv+txt"

        # 5. Single-pass inference (no mode loop)

        infer_kwargs = {
            "input_image":          input_image,
            "num_frames":           num_frames,
            "action":               action,
            "action_horizon":       sample["action_horizon"],
            "proprio":              proprio,
            "text_cfg_scale":       1.0,
            "action_cfg_scale":     1.0,
            "num_inference_steps":  self.eval_num_inference_steps,
            "seed":                 42,
            "tiled":                False,
            "task_video":           task_video,
            "task_video_dropped":   _tv_dropped,
        }
        if orig_context is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = orig_context
            infer_kwargs["context_mask"] = orig_context_mask
        else:
            infer_kwargs["prompt"] = prompt

        pred        = model.infer(**infer_kwargs)
        pred_video  = pred["video"]
        pred_action = pred.get("action", None)

        # Oracle run: inference with GT progress (skip Stage 1)
        pred_oracle = None
        progress_gt_batch = sample.get("progress_gt", None)  # [B, 2] in [0, 1]
        if progress_gt_batch is not None and not _tv_dropped:
            _pg0 = progress_gt_batch[0].float() * 2.0 - 1.0  # [2] → [-1, 1]
            oracle_kwargs = dict(infer_kwargs)
            oracle_kwargs["progress_override"] = _pg0
            pred_oracle = model.infer(**oracle_kwargs)

        # Video metrics
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            f"[{mode_name}] Eval pred/GT video shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )
        psnr_rg = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rg = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)
        psnr_rd = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rd = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        # Action metrics
        action_l2 = action_l1 = -1.0
        per_key_l1_vals = [-1.0] * len(per_key_names)
        per_key_l2_vals = [-1.0] * len(per_key_names)
        if action is not None and pred_action is not None:
            pred_btd = (pred_action if pred_action.ndim == 3 else pred_action.unsqueeze(0)).detach().float()
            gt_btd   = (action      if action.ndim == 3      else action.unsqueeze(0)).detach().float()
            if pred_btd.shape != gt_btd.shape:
                raise ValueError(
                    f"[{mode_name}] Eval pred/GT action shape mismatch: "
                    f"pred={tuple(pred_btd.shape)} vs gt={tuple(gt_btd.shape)}"
                )
            action_diff  = pred_btd - gt_btd
            action_l1    = action_diff.abs().mean().item()
            action_l2    = action_diff.pow(2).mean().item()
            per_key_l1_vals = [
                action_diff[..., s:e].abs().mean().item()
                for (s, e) in action_slices.values()
            ]
            per_key_l2_vals = [
                action_diff[..., s:e].pow(2).mean().item()
                for (s, e) in action_slices.values()
            ]

        # Oracle metrics
        psnr_rg_oracle = ssim_rg_oracle = -1.0
        action_l2_oracle = action_l1_oracle = -1.0
        if pred_oracle is not None:
            pred_oracle_video_tensor = pil_frames_to_video_tensor(pred_oracle["video"])
            psnr_rg_oracle = video_psnr(pred=pred_oracle_video_tensor, target=gt_video_tensor)
            ssim_rg_oracle = video_ssim(pred=pred_oracle_video_tensor, target=gt_video_tensor)
            pred_oracle_action = pred_oracle.get("action", None)
            if action is not None and pred_oracle_action is not None:
                _po = (pred_oracle_action if pred_oracle_action.ndim == 3 else pred_oracle_action.unsqueeze(0)).detach().float()
                _ga = (action if action.ndim == 3 else action.unsqueeze(0)).detach().float()
                _od = _po - _ga
                action_l1_oracle = _od.abs().mean().item()
                action_l2_oracle = _od.pow(2).mean().item()

        # Stitched video: [task_video_gt | pred | vae_recon | gt | task_video_pred]
        T_total = gt_video_tensor.shape[1]
        T_col = T_total
        if task_video is not None and not _tv_dropped:
            task_v01 = (task_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
            _end = task_video_end if task_video_end is not None else task_v01.shape[1]
            task_v01 = task_v01[:, task_video_start:_end]
            T_task_col = task_v01.shape[1]
            if T_task_col > 0 and T_task_col != T_col:
                indices = torch.linspace(0, T_task_col - 1, T_col).long()
                task_v01 = task_v01[:, indices]
            task_col_frames = []
            for t in range(task_v01.shape[1]):
                frame_np = (task_v01[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                task_col_frames.append(Image.fromarray(frame_np))
            task_col_tensor = pil_frames_to_video_tensor(task_col_frames)
        else:
            task_col_tensor = torch.zeros_like(gt_video_tensor[:, :T_col])

        # 5th column: task_video window from predicted progress
        pred_progress = pred.get("progress", None)
        if task_video is not None and not _tv_dropped and pred_progress is not None:
            _pp = pred_progress.float().cpu()  # [2] in [-1, 1]
            _pp_01 = (_pp + 1.0) * 0.5        # → [0, 1]
            T_full = task_video.shape[1]       # [C, T, H, W] dim 1 = temporal
            p_start = int(round(_pp_01[0].item() * max(T_full - 1, 1)))
            p_end   = int(round(_pp_01[1].item() * max(T_full - 1, 1))) + 1
            p_start = max(0, min(p_start, T_full - 1))
            p_end   = min(p_end, T_full)
            if p_start >= p_end:
                p_end = p_start + 1  # degenerate → single frame
            ptv01 = (task_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
            ptv01 = ptv01[:, p_start:p_end]
            T_ptv = ptv01.shape[1]
            if T_ptv > 0 and T_ptv != T_col:
                idx = torch.linspace(0, T_ptv - 1, T_col).long()
                ptv01 = ptv01[:, idx]
            pred_task_frames = []
            for t in range(ptv01.shape[1]):
                frame_np = (ptv01[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                pred_task_frames.append(Image.fromarray(frame_np))
            pred_task_col = pil_frames_to_video_tensor(pred_task_frames)
        else:
            pred_task_col = torch.zeros_like(gt_video_tensor[:, :T_col])

        stitched = torch.cat(
            [task_col_tensor, pred_video_tensor, vae_video_tensor, gt_video_tensor, pred_task_col],
            dim=3,
        ).contiguous()
        _n_cols = 5

        _col_W = gt_video_tensor.shape[3]
        _full_W = _n_cols * _col_W
        _line_h = 11
        _chars_per_line = max(20, _full_W // 7)
        _wrapped_lines = textwrap.wrap(prompt or "", width=_chars_per_line) or [""]
        _text_block_h = len(_wrapped_lines) * _line_h + 6

        stitched_frames = []
        _vid_H = stitched.shape[2]
        _vid_W = stitched.shape[3]
        for t in range(stitched.shape[1]):
            frame_np = (stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            canvas = Image.new("RGB", (_vid_W, _vid_H + _text_block_h), (0, 0, 0))
            canvas.paste(Image.fromarray(frame_np), (0, 0))
            draw = ImageDraw.Draw(canvas)
            for i, line in enumerate(_wrapped_lines):
                draw.text((4, _vid_H + 3 + i * _line_h), line, fill=(255, 255, 255))
            draw.rectangle([(0, 0), (len(mode_name) * 6 + 8, 13)], fill=(0, 0, 0))
            draw.text((4, 2), mode_name, fill=(255, 255, 255))
            # Frame counter (top-right)
            _T_total_stitched = stitched.shape[1]
            _fc_text = f"{t+1}/{_T_total_stitched}"
            _fc_w = len(_fc_text) * 6 + 8
            draw.rectangle([(_vid_W - _fc_w, 0), (_vid_W, 13)], fill=(0, 0, 0))
            draw.text((_vid_W - _fc_w + 4, 2), _fc_text, fill=(255, 255, 255))
            stitched_frames.append(canvas)

        mp4_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}{name_suffix}_{mode_name}.mp4",
        )
        save_mp4(stitched_frames, mp4_path, fps=8)

        _agent_dirs = sample.get("agent_episode_dir") or [None]
        _task_dirs  = sample.get("task_episode_dir")  or [None]
        json_path = mp4_path.replace(".mp4", ".json")
        with open(json_path, "w") as _f:
            json.dump({
                "agent_episode_dir": _agent_dirs[0] if isinstance(_agent_dirs, (list, tuple)) else _agent_dirs,
                "task_episode_dir":  _task_dirs[0]  if isinstance(_task_dirs,  (list, tuple)) else _task_dirs,
                "mode": mode_name,
            }, _f, indent=2)

        # Oracle MP4: same layout with oracle predictions
        if pred_oracle is not None:
            oracle_stitched = torch.cat(
                [task_col_tensor, pred_oracle_video_tensor, vae_video_tensor, gt_video_tensor, task_col_tensor],
                dim=3,
            ).contiguous()
            _oracle_mode = f"{mode_name}_oracle"
            oracle_frames = []
            for t in range(oracle_stitched.shape[1]):
                frame_np = (oracle_stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                canvas = Image.new("RGB", (_vid_W, _vid_H + _text_block_h), (0, 0, 0))
                canvas.paste(Image.fromarray(frame_np), (0, 0))
                draw = ImageDraw.Draw(canvas)
                for i, line in enumerate(_wrapped_lines):
                    draw.text((4, _vid_H + 3 + i * _line_h), line, fill=(255, 255, 255))
                draw.rectangle([(0, 0), (len(_oracle_mode) * 6 + 8, 13)], fill=(0, 0, 0))
                draw.text((4, 2), _oracle_mode, fill=(255, 255, 255))
                _fc_text = f"{t+1}/{oracle_stitched.shape[1]}"
                _fc_w = len(_fc_text) * 6 + 8
                draw.rectangle([(_vid_W - _fc_w, 0), (_vid_W, 13)], fill=(0, 0, 0))
                draw.text((_vid_W - _fc_w + 4, 2), _fc_text, fill=(255, 255, 255))
                oracle_frames.append(canvas)
            oracle_mp4_path = mp4_path.replace(".mp4", "_oracle.mp4")
            save_mp4(oracle_frames, oracle_mp4_path, fps=8)

        # Progress head eval
        progress_l1 = -1.0
        pred_progress = pred.get("progress", None)
        progress_gt = sample.get("progress_gt", None)
        if pred_progress is not None and progress_gt is not None:
            _gt_normed = progress_gt[0].float().cpu() * 2.0 - 1.0 if progress_gt.ndim > 1 else progress_gt.float().cpu() * 2.0 - 1.0
            _pred = pred_progress.float().cpu() if isinstance(pred_progress, torch.Tensor) else torch.tensor(pred_progress)
            progress_l1 = float((_pred - _gt_normed).abs().mean().item())
            logger.info(
                "[eval] progress head: pred=[%.4f, %.4f] gt=[%.4f, %.4f] L1=%.4f",
                float(_pred[0]), float(_pred[1]),
                float(_gt_normed[0]), float(_gt_normed[1]),
                progress_l1,
            )

        # Action + progress comparison plot
        if action is not None and pred_action is not None:
            try:
                import sys as _sys
                _eval_dir = os.path.join(os.path.dirname(__file__), "..", "..", "eval", "real_openloop")
                if _eval_dir not in _sys.path:
                    _sys.path.insert(0, _eval_dir)
                from action_viz import save_action_plot
                # Strip mask dim only (last), keep action + progress
                _gt_act = action if action.ndim == 2 else action[0]        # [T, action_dim+2]
                _pred_act = pred_action if pred_action.ndim == 2 else pred_action[0]
                _gt_np = _gt_act[:, :-1].detach().float().cpu().numpy()    # [T, action_dim+1]
                _pred_np = _pred_act[:, :-1].detach().float().cpu().numpy()

                # Append progress head prediction as extra dims (broadcast to T steps)
                if pred_progress is not None and progress_gt is not None:
                    _gt_normed_np = _gt_normed.numpy()   # [2]: [start, end]
                    _pred_prog_np = _pred.numpy()         # [2]: [start, end]
                    T = _gt_np.shape[0]
                    # Broadcast [start, end] → [T] via linspace for visualization
                    _gt_prog_T = np.linspace(_gt_normed_np[0], _gt_normed_np[1], T).reshape(T, 1)
                    _pred_prog_T = np.linspace(_pred_prog_np[0], _pred_prog_np[1], T).reshape(T, 1)
                    _gt_np = np.concatenate([_gt_np, _gt_prog_T], axis=1)
                    _pred_np = np.concatenate([_pred_np, _pred_prog_T], axis=1)

                _action_png = mp4_path.replace(".mp4", "_action.png")
                save_action_plot(
                    _action_png, _pred_np, _gt_np,
                    title=f"Action (normed) step={self.global_step} {mode_name}",
                )
            except Exception as _e:
                logging.warning("[eval] action plot failed: %s", _e)

        # 6. Gather metrics across ranks
        metric_vals = (
            [psnr_rg, ssim_rg, psnr_rd, ssim_rd, action_l2, action_l1, progress_l1,
             psnr_rg_oracle, ssim_rg_oracle, action_l2_oracle, action_l1_oracle]
            + per_key_l1_vals
            + per_key_l2_vals
        )
        flat_vals = [float(val_loss), float(psnr_dg), float(ssim_dg)] + metric_vals

        local_metrics = torch.tensor(flat_vals, device=self.accelerator.device, dtype=torch.float32).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local_metrics)
        mean_vals = gathered.mean(dim=0).tolist()

        if was_dit_training:
            self._set_dit_only_train_mode()

        # 7. Build result dict
        result = {
            "val_loss": float(mean_vals[0]),
            "psnr_dg":  float(mean_vals[1]),
            "ssim_dg":  float(mean_vals[2]),
            "video_paths": {mode_name: mp4_path},
            "eval_mode": mode_name,
        }
        vals = mean_vals[3:]
        result[f"{mode_name}/psnr_rg"] = float(vals[0])
        result[f"{mode_name}/ssim_rg"] = float(vals[1])
        result[f"{mode_name}/psnr_rd"] = float(vals[2])
        result[f"{mode_name}/ssim_rd"] = float(vals[3])
        if vals[4] >= 0.0:
            result[f"{mode_name}/action_l2"] = float(vals[4])
        if vals[5] >= 0.0:
            result[f"{mode_name}/action_l1"] = float(vals[5])
        if vals[6] >= 0.0:
            result[f"{mode_name}/progress_l1"] = float(vals[6])
        if vals[7] >= 0.0:
            result[f"{mode_name}/psnr_rg_oracle"] = float(vals[7])
        if vals[8] >= 0.0:
            result[f"{mode_name}/ssim_rg_oracle"] = float(vals[8])
        if vals[9] >= 0.0:
            result[f"{mode_name}/action_l2_oracle"] = float(vals[9])
        if vals[10] >= 0.0:
            result[f"{mode_name}/action_l1_oracle"] = float(vals[10])
        for ki, key in enumerate(per_key_names):
            result[f"{mode_name}/eval_action_l1/{key}"] = float(vals[_METRIC_COLS + ki])
            result[f"{mode_name}/eval_action_l2/{key}"] = float(vals[_METRIC_COLS + len(per_key_names) + ki])
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()
        self._apply_staged_freeze(self.global_step)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            _t_data_start = time.time()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            data_load_time = time.time() - _t_data_start

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(
                        sample,
                        action_component_slices=self.action_component_slices,
                    )
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self._apply_staged_freeze(self.global_step)
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    # Assert all ranks return the same loss_dict keys (step=1 only).
                    # all_gather_object uses pickle and is ~10s per call — do NOT run every step.
                    # Mismatched keys cause collective mismatch deadlock — fail loudly on first step.
                    if self.global_step == 1:
                        all_keys_per_rank = [None] * self.accelerator.num_processes
                        torch.distributed.all_gather_object(all_keys_per_rank, set(loss_dict.keys()))
                        if len(set(frozenset(k) for k in all_keys_per_rank)) != 1:
                            raise RuntimeError(
                                f"loss_dict key mismatch across ranks at step {self.global_step}: "
                                f"{all_keys_per_rank}. All ranks must return identical keys from training_loss()."
                            )

                    # ── Step 1: gather raw per-rank values for every metric ──────────────
                    gathered_raw = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        gathered_raw[key] = self.accelerator.gather(metric_tensor)  # [num_processes]

                    # ── Step 2: overall metrics (nanmean across all ranks, existing behaviour) ─
                    global_loss_metrics = {}
                    for key, gathered in gathered_raw.items():
                        valid = gathered[~gathered.isnan()]
                        global_loss_metrics[key] = float(valid.mean().item()) if valid.numel() > 0 else float("nan")

                    # ── Step 3: split by task_video presence ──
                    # has_task_video: 1.0 = all have TV, nan = all dropped, 0.5 = mixed (skip)
                    rho_gathered = gathered_raw.get("has_task_video")
                    if rho_gathered is not None:
                        has_tv_mask = (rho_gathered == 1.0)   # pure with-TV ranks only
                        no_tv_mask  = rho_gathered.isnan()    # pure without-TV ranks only
                        # mixed (0.5) ranks are excluded from both
                        for prefix, mask in (("with_task_video", has_tv_mask), ("without_task_video", no_tv_mask)):
                            if not mask.any():
                                continue
                            for key, gathered in gathered_raw.items():
                                vals = gathered[mask]
                                valid = vals[~vals.isnan()]
                                if valid.numel() > 0:
                                    global_loss_metrics[f"{prefix}/{key}"] = float(valid.mean().item())
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    data_load_time_tensor = torch.tensor(data_load_time, device=loss.device, dtype=torch.float32)
                    global_data_load_time = float(self.accelerator.gather(data_load_time_tensor).max().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        logger.info(
                            "[train] epoch=%d step=%d/%d loss=%.4f lr=%.2e speed=%.2f step/s, %.2f samples/s data=%.2fs eta=%s",
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            global_data_load_time,
                            eta_str,
                        )

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            "performance/data_load_time_s": global_data_load_time,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            logger.info(
                                "[eval] step=%d val_loss=%.4f psnr_dg=%.4f ssim_dg=%.4f",
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_dg"],
                                metrics["ssim_dg"],
                            )
                            # WandB: log all float metrics; skip video path dicts
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_dg":  float(metrics["psnr_dg"]),
                                "eval/ssim_dg":  float(metrics["ssim_dg"]),
                            }
                            for key, val in metrics.items():
                                if key == "video_paths":
                                    continue
                                if isinstance(val, float):
                                    eval_payload[f"eval/{key}"] = val
                            self._wandb_log(eval_payload)

                        # Extra val datasets (e.g. "seen")
                        for ds_name, extra_ds in self.extra_val_datasets.items():
                            extra_metrics = self._evaluate_dataset(extra_ds, name_suffix=f"_{ds_name}")
                            self.accelerator.wait_for_everyone()
                            if extra_metrics is not None and self.accelerator.is_main_process:
                                extra_payload = {}
                                for key, val in extra_metrics.items():
                                    if key == "video_paths":
                                        continue
                                    if isinstance(val, float):
                                        extra_payload[f"eval_{ds_name}/{key}"] = val
                                self._wandb_log(extra_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
