import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


# NOTE: This function instantiates FastWAMJoint (NOT FastWAM base).
# FastWAMJoint overrides _build_mot_attention_mask so that action tokens attend
# to ALL video tokens in self-attention (FastWAM base only allows first frame).
# All model configs with _target_: fastwam.runtime.create_fastwam_joint go through here.
def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    task_video_conditioning_mode: str = "prepend",
    log_cross_attn_weights: bool = False,
    action_prediction_type: str = "velocity",
    video_prediction_type: str = "velocity",
    progress_scheduler=None,
    progress_prediction_type: str = "velocity",
    # True (default): flow-matching noise on progress token + loss_progress computed.
    # False: clean GT progress as conditioning token, loss_progress skipped.
    # Set in model YAML: noise_to_progress_token: false
    noise_to_progress_token: bool = True,
    use_noisy_progress_group: bool = False,
    num_inference_steps_progress: int | None = None,
    noise_clean_progress_token: bool = False,
    noise_clean_progress_prob: float = 0.5,
    noise_clean_progress_scale: float = 0.1,
    action_self_attn_to_task_video: bool = False,
    visual_encoder=None,
):
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(visual_encoder, DictConfig):
        visual_encoder = OmegaConf.to_container(visual_encoder, resolve=True)
    if visual_encoder is not None and not isinstance(visual_encoder, dict):
        raise ValueError(f"`visual_encoder` must be dict-like or None, got {type(visual_encoder)}")
    visual_encoder_config = None
    if visual_encoder is not None and bool(visual_encoder.get("enabled", False)):
        visual_encoder_config = {
            k: v for k, v in visual_encoder.items() if k not in ("enabled", "lr_scale")
        }

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    if isinstance(progress_scheduler, DictConfig):
        progress_scheduler = OmegaConf.to_container(progress_scheduler, resolve=True)
    if progress_scheduler is None:
        progress_scheduler = {}
    if not isinstance(progress_scheduler, dict):
        raise ValueError(f"`progress_scheduler` must be dict-like, got {type(progress_scheduler)}")

    return FastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_progress=float(loss.get("lambda_progress", 1.0)),
        task_video_conditioning_mode=str(task_video_conditioning_mode),
        log_cross_attn_weights=bool(log_cross_attn_weights),
        action_prediction_type=str(action_prediction_type),
        video_prediction_type=str(video_prediction_type),
        progress_prediction_type=str(progress_prediction_type),
        progress_num_train_timesteps=int(progress_scheduler.get("num_train_timesteps", 1000)),
        progress_train_shift=float(progress_scheduler.get("train_shift", 3.0)),
        progress_infer_shift=float(progress_scheduler.get("infer_shift", 3.0)),
        noise_to_progress_token=bool(noise_to_progress_token),
        use_noisy_progress_group=bool(use_noisy_progress_group),
        num_inference_steps_progress=(int(num_inference_steps_progress) if num_inference_steps_progress is not None else None),
        noise_clean_progress_token=bool(noise_clean_progress_token),
        noise_clean_progress_prob=float(noise_clean_progress_prob),
        noise_clean_progress_scale=float(noise_clean_progress_scale),
        action_self_attn_to_task_video=bool(action_self_attn_to_task_video),
        visual_encoder_config=visual_encoder_config,
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    train_stats_path = data_cfg.train.get("pretrained_norm_stats")
    default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")

    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)

    extra_val_datasets = {}
    if data_cfg.get("extra_val") is not None:
        for name, extra_cfg in data_cfg.extra_val.items():
            ev_stats = extra_cfg.get("pretrained_norm_stats") if hasattr(extra_cfg, "get") else None
            pretrained_norm_stats = ev_stats or train_stats_path or default_stats_path
            logger.info("Building extra_val['%s'] with pretrained_norm_stats: %s", name, pretrained_norm_stats)
            extra_val_datasets[name] = instantiate(extra_cfg, pretrained_norm_stats=pretrained_norm_stats)

    return train_ds, val_ds, extra_val_datasets


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    # Print full resolved config on rank 0
    is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
    if is_rank0:
        logger.info("=" * 80)
        logger.info("Resolved training config:")
        logger.info("=" * 80)
        logger.info("\n%s", OmegaConf.to_yaml(cfg, resolve=True))
        logger.info("=" * 80)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)

    # task_video/text drop is now handled at dataset side (per-sample),
    # no longer synced to model.

    train_ds, val_ds, extra_val_datasets = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        extra_val_datasets=extra_val_datasets,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
