from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from self_grounded_prediction.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .visual_encoder import FirstFrameVisualEncoder

logger = get_logger(__name__)


class SelfGroundedPredictor(torch.nn.Module):
    """MoT world model with video/action experts.

    WARNING — THIS IS THE BASE CLASS AND IS NOT USED DIRECTLY IN PRODUCTION.
    Production configs (_target_: self_grounded_prediction.runtime.create_self_grounded_predictor) instantiate
    SelfGroundedPredictorJoint (self_grounded_predictor_joint.py), which overrides _build_mot_attention_mask.

    The key behavioral difference is in _build_mot_attention_mask:
        SelfGroundedPredictor (this class):  action → first-frame video tokens only
        SelfGroundedPredictorJoint (subclass, used in prod): action → ALL video tokens
    """

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_progress: float = 1.0,
        task_video_conditioning_mode: str = "prepend",
        log_cross_attn_weights: bool = False,
        action_prediction_type: str = "velocity",
        video_prediction_type: str = "velocity",
        progress_prediction_type: str = "velocity",
        progress_num_train_timesteps: int = 1000,
        progress_train_shift: float = 3.0,
        progress_infer_shift: float = 3.0,
        # -----------------------------------------------------------------------
        # noise_to_progress_token: whether to apply flow-matching noise to the
        # progress token during training.
        #   True  (default) — noise added via train_progress_scheduler;
        #                      loss_progress is computed normally.
        #   False           — clean GT progress used as conditioning token;
        #                      loss_progress is skipped (target_progress=None).
        # ⚠️  Only affects the progress token path. Action/video noise unchanged.
        # Configure via model YAML: noise_to_progress_token: false
        # -----------------------------------------------------------------------
        noise_to_progress_token: bool = True,
        use_noisy_progress_group: bool = False,
        num_inference_steps_progress: int | None = None,
        # -----------------------------------------------------------------------
        # noise_clean_progress_token: add bounded Gaussian noise to the clean
        # progress token during training for robustness.  Requires
        # use_noisy_progress_group=True.  No loss on clean progress — purely
        # conditioning.  t_mod stays timestep=0 (no train-inference gap).
        # -----------------------------------------------------------------------
        noise_clean_progress_token: bool = False,
        noise_clean_progress_prob: float = 0.5,
        noise_clean_progress_scale: float = 0.1,
        action_self_attn_to_task_video: bool = False,
        visual_encoder_config: Optional[dict] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        # Progress flow matching: encoder maps noisy progress [B,2] → token [B,1,D],
        # decoder maps output token [B,D] → predicted velocity/sample [B,2].
        D = self.video_expert.hidden_dim
        self.progress_encoder = nn.Sequential(
            nn.Linear(2, D),
            nn.GELU(),
            nn.Linear(D, D),
        ).to(torch_dtype)
        self.progress_decoder = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2),
        ).to(torch_dtype)
        self.train_progress_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=progress_num_train_timesteps,
            shift=progress_train_shift,
        )
        self.infer_progress_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=progress_num_train_timesteps,
            shift=progress_infer_shift,
        )
        self.progress_prediction_type = str(progress_prediction_type)
        assert self.progress_prediction_type in ("velocity", "sample"), (
            f"progress_prediction_type must be 'velocity' or 'sample', "
            f"got '{self.progress_prediction_type}'"
        )
        self.noise_to_progress_token = bool(noise_to_progress_token)

        self.use_noisy_progress_group = bool(use_noisy_progress_group)
        self.num_inference_steps_progress = (
            int(num_inference_steps_progress) if num_inference_steps_progress is not None else None
        )
        if self.use_noisy_progress_group:
            assert not self.noise_to_progress_token, (
                "use_noisy_progress_group requires noise_to_progress_token=False "
                "(clean progress for main group, noisy progress handled by isolated group)"
            )
            assert task_video_conditioning_mode == "prepend_cross_attn", (
                f"use_noisy_progress_group requires task_video_conditioning_mode='prepend_cross_attn', "
                f"got '{task_video_conditioning_mode}'"
            )

        self.noise_clean_progress_token = bool(noise_clean_progress_token)
        self.noise_clean_progress_prob = float(noise_clean_progress_prob)
        self.noise_clean_progress_scale = float(noise_clean_progress_scale)
        if self.noise_clean_progress_token:
            assert self.use_noisy_progress_group, (
                "noise_clean_progress_token requires use_noisy_progress_group=True"
            )

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_progress = float(loss_lambda_progress)
        # task_video/text drop is now handled at dataset side (per-sample)
        self.task_video_conditioning_mode = str(task_video_conditioning_mode)
        assert self.task_video_conditioning_mode in ("prepend", "cross_attn", "prepend_cross_attn"), (
            f"task_video_conditioning_mode must be 'prepend', 'cross_attn', or 'prepend_cross_attn', "
            f"got '{self.task_video_conditioning_mode}'"
        )
        self.action_prediction_type = str(action_prediction_type)
        assert self.action_prediction_type in ("velocity", "sample"), (
            f"action_prediction_type must be 'velocity' or 'sample', "
            f"got '{self.action_prediction_type}'"
        )
        self.video_prediction_type = str(video_prediction_type)
        assert self.video_prediction_type in ("velocity", "sample"), (
            f"video_prediction_type must be 'velocity' or 'sample', "
            f"got '{self.video_prediction_type}'"
        )
        self.log_cross_attn_weights = bool(log_cross_attn_weights)
        self.action_self_attn_to_task_video = bool(action_self_attn_to_task_video)

        # --- Trainable visual encoder (off by default) ---
        if visual_encoder_config:
            ve_cfg = dict(visual_encoder_config)
            self.visual_encoder = FirstFrameVisualEncoder(
                text_dim=self.text_dim,
                backbone_name=str(ve_cfg.pop("backbone_name", "dinov2_vits14")),
                num_cameras=int(ve_cfg.pop("num_cameras", 2)),
                camera_input_size=int(ve_cfg.pop("camera_input_size", 224)),
                proj_hidden=int(ve_cfg.pop("proj_hidden", 1024)),
                dropout=float(ve_cfg.pop("dropout", 0.0)),
                torch_dtype=torch_dtype,
                backbone_local_repo=ve_cfg.pop("backbone_local_repo", None),
                backbone_weights_path=ve_cfg.pop("backbone_weights_path", None),
                siglip_local_weights_path=ve_cfg.pop("siglip_local_weights_path", None),
            )
            if ve_cfg:
                raise ValueError(f"Unknown `visual_encoder_config` keys: {list(ve_cfg)}")
        else:
            self.visual_encoder = None

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_progress: float = 1.0,
        task_video_conditioning_mode: str = "prepend",
        log_cross_attn_weights: bool = False,
        action_prediction_type: str = "velocity",
        video_prediction_type: str = "velocity",
        progress_prediction_type: str = "velocity",
        progress_num_train_timesteps: int = 1000,
        progress_train_shift: float = 3.0,
        progress_infer_shift: float = 3.0,
        noise_to_progress_token: bool = True,  # see __init__ docblock for semantics
        use_noisy_progress_group: bool = False,
        num_inference_steps_progress: int | None = None,
        noise_clean_progress_token: bool = False,
        noise_clean_progress_prob: float = 0.5,
        noise_clean_progress_scale: float = 0.1,
        action_self_attn_to_task_video: bool = False,
        visual_encoder_config: Optional[dict] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for SelfGroundedPredictor.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for SelfGroundedPredictor.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_progress=loss_lambda_progress,
            task_video_conditioning_mode=task_video_conditioning_mode,
            log_cross_attn_weights=log_cross_attn_weights,
            action_prediction_type=action_prediction_type,
            video_prediction_type=video_prediction_type,
            progress_prediction_type=progress_prediction_type,
            progress_num_train_timesteps=progress_num_train_timesteps,
            progress_train_shift=progress_train_shift,
            progress_infer_shift=progress_infer_shift,
            noise_to_progress_token=noise_to_progress_token,
            use_noisy_progress_group=use_noisy_progress_group,
            num_inference_steps_progress=num_inference_steps_progress,
            noise_clean_progress_token=noise_clean_progress_token,
            noise_clean_progress_prob=noise_clean_progress_prob,
            noise_clean_progress_scale=noise_clean_progress_scale,
            action_self_attn_to_task_video=action_self_attn_to_task_video,
            visual_encoder_config=visual_encoder_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # Zero out padding embeddings (same as pre-computed instruction.pt).
        # Keep original mask so padding positions have mask=False, matching
        # training-time collator behavior (pad_context_sequence sets mask=False
        # for padded positions).  The old `mask = torch.ones_like(mask)` caused
        # cross-attn to attend to zero-embedding padding tokens, diluting the
        # real text signal and creating a train-infer mismatch.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _append_visual_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        first_frame_rgb: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.visual_encoder is None or first_frame_rgb is None:
            return context, context_mask
        if first_frame_rgb.ndim != 4 or first_frame_rgb.shape[1] != 3:
            raise ValueError(
                f"`first_frame_rgb` must be [B, 3, H, W], got {tuple(first_frame_rgb.shape)}"
            )
        rgb = first_frame_rgb.to(device=self.device, dtype=context.dtype)
        visual_tokens = self.visual_encoder(rgb).to(dtype=context.dtype)  # [B, N, D]
        visual_mask = torch.ones(
            (context_mask.shape[0], visual_tokens.shape[1]),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat([context, visual_tokens], dim=1),
            torch.cat([context_mask, visual_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "SelfGroundedPredictor training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for SelfGroundedPredictor training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        # Raw first-frame RGB in [-1, 1] for trainable visual encoder.
        first_frame_rgb = None
        if self.visual_encoder is not None:
            first_frame_rgb = input_video[:, :, 0]  # [B, 3, H, W]

        # Task video: VAE encode as clean conditioning latents (no noise will be added)
        task_video_latents = None
        task_video_raw = sample.get("task_video", None)
        if task_video_raw is not None:
            tv = task_video_raw
            if tv.ndim == 4:
                tv = tv.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
            tv = tv.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            task_video_latents = self._encode_video_latents(tv, tiled=tiled)

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        if self.visual_encoder is not None:
            context, context_mask = self._append_visual_to_context(
                context=context,
                context_mask=context_mask,
                first_frame_rgb=first_frame_rgb,
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        progress_gt = sample.get("progress_gt", None)
        if progress_gt is not None:
            progress_gt = progress_gt.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "first_frame_rgb": first_frame_rgb,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "task_video_latents": task_video_latents,
            "task_video_dropped": sample.get("task_video_dropped"),
            "progress_gt": progress_gt,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        task_video_seq_len: int = 0,
        progress_seq_len: int = 0,
        noisy_group_seq_len: int = 0,
    ) -> torch.Tensor:
        """Build attention mask for [progress | task_video | video | noisy_group | action].

        NOTE: SelfGroundedPredictorJoint (self_grounded_predictor_joint.py) overrides this method so that
        action attends to ALL video tokens. This base version restricts action
        to first-frame video only. In production we use SelfGroundedPredictorJoint.

        When noisy_group_seq_len > 0, an isolated noisy group is inserted between
        agent video and action. The noisy group self-attends (full) but is completely
        blocked from all other tokens (bidirectional isolation).

        Attention rules (base SelfGroundedPredictor):
            progress -> task_video/agent_video: full; -> action/noisy: blocked
            task_video -> progress/task_video: full; -> rest: blocked
            video -> progress/task_video/video: allowed; -> noisy/action: blocked
            noisy_group -> noisy_group: FULL; -> rest: BLOCKED
            rest -> noisy_group: BLOCKED
            action -> progress/first-frame-video/action: allowed; -> rest: blocked
        """
        total = (progress_seq_len + task_video_seq_len + video_seq_len
                 + noisy_group_seq_len + action_seq_len)
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        p_end   = progress_seq_len
        tv_end  = p_end + task_video_seq_len
        v_end   = tv_end + video_seq_len
        ng_end  = v_end + noisy_group_seq_len
        a_start = ng_end

        # progress -> progress: full
        if p_end > 0:
            mask[:p_end, :p_end] = True
        # progress -> task_video: full
        if p_end > 0 and tv_end > p_end:
            mask[:p_end, p_end:tv_end] = True
        # progress -> agent_video: full
        if p_end > 0:
            mask[:p_end, tv_end:v_end] = True
        # progress -> noisy_group/action: blocked (default False)

        # task_video -> progress: full
        if p_end > 0 and tv_end > p_end:
            mask[p_end:tv_end, :p_end] = True
        # task_video -> task_video: full
        if tv_end > p_end:
            mask[p_end:tv_end, p_end:tv_end] = True

        # video -> progress: full
        if p_end > 0:
            mask[tv_end:v_end, :p_end] = True
        # video -> task_video: can attend
        if tv_end > p_end:
            mask[tv_end:v_end, p_end:tv_end] = True

        # video -> video: original mask
        mask[tv_end:v_end, tv_end:v_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # noisy group -> noisy group: FULL (completely isolated from rest)
        if noisy_group_seq_len > 0:
            mask[v_end:ng_end, v_end:ng_end] = True

        # action -> progress: full
        if p_end > 0:
            mask[a_start:, :p_end] = True
        # action -> task_video: blocked

        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[a_start:, tv_end:tv_end + first_frame_tokens] = True

        # action -> action: full
        mask[a_start:, a_start:] = True

        return mask

    def _compute_task_video_attn_stats(self, attn_weights, task_video_seq_len, video_seq_len, progress_seq_len=0):
        """Compute agent_video → task_video attention ratio from last-layer weights.

        Args:
            attn_weights: [B, H, S_total, S_total] detached attention weights
            task_video_seq_len: number of task_video tokens
            video_seq_len: number of agent_video tokens
            progress_seq_len: number of progress tokens prepended before task_video (0 or 1)

        Returns:
            dict with rho_video_to_task (float): mean attention from agent_video to task_video
        """
        if task_video_seq_len == 0 or attn_weights is None:
            return {}

        tv_end = progress_seq_len + task_video_seq_len
        v_start = tv_end
        v_end = tv_end + video_seq_len

        # agent_video queries attending to task_video keys: [B, H, S_video, S_task]
        # task_video is in columns [progress_seq_len : tv_end]
        attn_to_task = attn_weights[:, :, v_start:v_end, progress_seq_len:tv_end]
        # Sum over task_video keys per query token, then average over everything
        rho = attn_to_task.sum(dim=-1).mean().item()

        return {"rho_video_to_task": rho}

    def _build_progress_freq(self, f_task: int, f_agent: int, device: torch.device) -> torch.Tensor:
        """Build 3D RoPE frequency for the progress token.

        Assigns the progress token temporal position f_task + f_agent (one past all
        video frames visible in self-attention), height=0, width=0. This gives a
        clearly identifiable, non-conflicting position in the video expert's freq space.

        Returns:
            progress_freq: [1, 1, freq_dim] complex tensor matching video token freq format.
        """
        f_total = f_task + f_agent
        return torch.cat([
            self.video_expert.freqs[0][f_total],   # temporal component
            self.video_expert.freqs[1][0],          # height component
            self.video_expert.freqs[2][0],          # width component
        ], dim=-1).view(1, 1, -1).to(device)

    def _prepend_progress_token(
        self,
        latents_progress: torch.Tensor,
        combined_video_tokens: torch.Tensor,
        combined_video_freqs: torch.Tensor,
        combined_video_t_mod: torch.Tensor,
        combined_video_context_mask: torch.Tensor,
        video_pre: dict,
        f_task_meta: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode noisy progress [B, 2] and prepend its token to the video expert sequence.

        Assigns 3D RoPE position (f_task_meta + f_agent, 0, 0) — one past all video frames.
        Reuses the first agent video token's t_mod and context_mask for the progress token.

        Returns:
            (combined_video_tokens, combined_video_freqs, combined_video_t_mod,
             combined_video_context_mask) with the progress token prepended at position 0.
        """
        B = latents_progress.shape[0]
        f_agent_meta = video_pre["meta"]["grid_size"][0]
        progress_token = self.progress_encoder(latents_progress).unsqueeze(1)  # [B, 1, D]
        assert progress_token.shape == (B, 1, combined_video_tokens.shape[-1]), (
            f"progress_token shape mismatch: {progress_token.shape}"
        )
        progress_freq = self._build_progress_freq(f_task_meta, f_agent_meta, combined_video_tokens.device)
        return (
            torch.cat([progress_token,                       combined_video_tokens],       dim=1),
            torch.cat([progress_freq,                        combined_video_freqs],        dim=0),
            torch.cat([video_pre["t_mod"][:, :1, :],         combined_video_t_mod],        dim=1),
            torch.cat([video_pre["context_mask"][:, :1, :],  combined_video_context_mask], dim=1),
        )

    def _compute_t_mod_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Compute single-token time modulation from a timestep using video expert's time embedding.

        Used for the noisy_progress_token so its t_mod reflects the actual progress
        noise level (timestep_progress), not the video noise level. This ensures:
        - The model knows the noise level of the progress content it needs to denoise.
        - Train-inference consistency: both use the progress scheduler's timesteps.
        - Clear distinction from clean_progress_token (which uses timestep=0 t_mod).

        Args:
            timestep: [B] scalar timestep per sample.

        Returns:
            t_mod: [B, 1, 6, D] time modulation tensor for one token.
        """
        from .wan_video_dit import sinusoidal_embedding_1d
        t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, timestep)  # [B, freq_dim*2]
        t = self.video_expert.time_embedding(t_emb)  # [B, D]
        t_mod = self.video_expert.time_projection(t)  # [B, 6*D]
        t_mod = t_mod.unflatten(1, (6, self.video_expert.hidden_dim))  # [B, 6, D]
        return t_mod.unsqueeze(1)  # [B, 1, 6, D]

    def _add_clean_progress_noise(self, progress: torch.Tensor) -> torch.Tensor:
        """Add bounded Gaussian noise to clean progress for robustness training.

        Args:
            progress: [B, 2] normalized progress in [-1, 1].

        Returns:
            noisy_progress: [B, 2] clamped to [-1, 1].
        """
        noise = torch.randn_like(progress) * self.noise_clean_progress_scale
        return (progress + noise).clamp(-1.0, 1.0)

    def _build_noisy_group(
        self,
        latents_noisy_progress: torch.Tensor,
        timestep_progress: torch.Tensor,
        task_video_pre: dict,
        video_pre: dict,
        video_tokens_per_frame: int,
        f_task_meta: int,
        f_agent_meta: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the isolated noisy group: [noisy_p(1) | tv_noisy(S_tv) | ff_noisy(S_1f)].

        All conditioning tokens (task video, first frame) are **detached** from the
        computational graph to prevent gradient flow from progress loss back to pre_dit.
        The noisy progress token is NOT detached — ``progress_encoder`` receives gradients.

        t_mod assignment:
        - noisy_progress: from ``timestep_progress`` (reflects actual noise level)
        - tv_noisy: from task_video_pre (timestep=0, clean conditioning)
        - ff_noisy: from video_pre first frame (timestep=0 in separated_timestep mode)

        Args:
            latents_noisy_progress: [B, 2] noised progress values.
            timestep_progress: [B] progress noise level for t_mod computation.
            task_video_pre: pre_dit output of task video (timestep=0).
            video_pre: pre_dit output of agent video (first frame is clean).
            video_tokens_per_frame: number of tokens per spatial frame (H*W).
            f_task_meta: number of task video frames (for RoPE position).

        Returns:
            (tokens, freqs, t_mod, context_mask) ready to append to combined video sequence.
            tokens:  [B, 1+S_tv+S_1f, D]
            freqs:   [1+S_tv+S_1f, 1, freq_dim]
            t_mod:   [B, 1+S_tv+S_1f, 6, D]
            cmask:   [B, 1+S_tv+S_1f, L]
        """
        tpf = video_tokens_per_frame

        # 1. Encode noisy progress → token (NOT detached: encoder gets gradients)
        noisy_p_token = self.progress_encoder(latents_noisy_progress).unsqueeze(1)  # [B, 1, D]

        # 2. Noisy progress t_mod from timestep_progress (reflects actual noise level).
        #    Distinguishes from clean_progress which uses timestep=0 t_mod.
        noisy_p_t_mod = self._compute_t_mod_from_timestep(timestep_progress)  # [B, 1, 6, D]

        # 3. Clone task_video tokens (already clean, timestep=0) — DETACHED
        tv_tokens = task_video_pre["tokens"].detach().clone()          # [B, S_tv, D]
        tv_freqs  = task_video_pre["freqs"].clone()                    # [S_tv, 1, freq_dim]
        tv_t_mod  = task_video_pre["t_mod"].detach().clone()           # [B, S_tv, 6, D]
        tv_cmask  = task_video_pre["context_mask"].detach().clone()    # [B, S_tv, L]

        # 4. Clone first frame tokens from video_pre (clean content) — DETACHED
        ff_tokens = video_pre["tokens"][:, :tpf, :].detach().clone()           # [B, S_1f, D]
        ff_freqs  = video_pre["freqs"][:tpf].clone()                           # [S_1f, 1, freq_dim]
        ff_t_mod  = video_pre["t_mod"][:, :tpf].detach().clone()               # [B, S_1f, 6, D]
        ff_cmask  = video_pre["context_mask"][:, :tpf, :].detach().clone()     # [B, S_1f, L]

        # 5. Noisy progress RoPE position — same as clean_progress: (f_task + f_agent, 0, 0)
        noisy_p_freq = self._build_progress_freq(f_task_meta, f_agent_meta, noisy_p_token.device)  # [1, 1, freq_dim]
        noisy_p_cmask = video_pre["context_mask"][:, :1, :].detach().clone()   # [B, 1, L]

        # 6. Concatenate: [noisy_p | tv_noisy | ff_noisy]
        tokens = torch.cat([noisy_p_token, tv_tokens, ff_tokens], dim=1)
        freqs  = torch.cat([noisy_p_freq,  tv_freqs,  ff_freqs],  dim=0)
        t_mod  = torch.cat([noisy_p_t_mod, tv_t_mod,  ff_t_mod],  dim=1)
        cmask  = torch.cat([noisy_p_cmask, tv_cmask,  ff_cmask],  dim=1)

        return tokens, freqs, t_mod, cmask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False, action_component_slices: dict | None = None):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video,
            prediction_type=self.video_prediction_type,
        )

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action,
            prediction_type=self.action_prediction_type,
        )

        # Progress token: optionally add noise (flow-matching) or use clean GT as conditioning.
        progress_gt = inputs.get("progress_gt")
        latents_progress = None
        target_progress = None
        timestep_progress = None
        if progress_gt is not None:
            progress_gt_normed = progress_gt * 2.0 - 1.0  # [B, 2] → [-1, 1]
            if self.noise_to_progress_token:
                noise_progress = torch.randn_like(progress_gt_normed)
                timestep_progress = self.train_progress_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=progress_gt_normed.dtype,
                )
                latents_progress = self.train_progress_scheduler.add_noise(
                    progress_gt_normed, noise_progress, timestep_progress
                )  # [B, 2]
                target_progress = self.train_progress_scheduler.training_target(
                    progress_gt_normed, noise_progress, timestep_progress,
                    prediction_type=self.progress_prediction_type,
                )  # [B, 2]
            else:
                latents_progress = progress_gt_normed  # clean conditioning; no flow-matching loss
                if self.noise_clean_progress_token and torch.rand(1).item() < self.noise_clean_progress_prob:
                    latents_progress = self._add_clean_progress_noise(progress_gt_normed)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        # Task video: clean conditioning (timestep=0, no noise)
        # Per-sample drop decision made at dataset side; dropped samples have zeros tensor
        # + attention mask blocks video → task_video for those samples.
        task_video_latents = inputs.get("task_video_latents")
        task_video_dropped = inputs.get("task_video_dropped")  # list[bool], bool, or None
        # Normalize to list[bool] for uniform indexing
        if task_video_dropped is not None and not isinstance(task_video_dropped, (list, tuple)):
            task_video_dropped = [bool(task_video_dropped)] * batch_size
        # (text drop is now handled at dataset side via context_mask=zeros)
        task_video_seq_len = 0
        text_context_len = video_pre["context_mask"].shape[2]  # L (text tokens, for cross-attn ratio metric)

        if task_video_latents is not None:
            task_video_pre = self.video_expert.pre_dit(
                x=task_video_latents,
                timestep=torch.zeros((batch_size,), device=self.device, dtype=input_latents.dtype),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=True,  # all frames get timestep=0
            )

            if self.task_video_conditioning_mode == "prepend":
                # Existing behavior: prepend task_video tokens to self-attn sequence
                task_video_seq_len = task_video_pre["tokens"].shape[1]
                combined_video_tokens = torch.cat([task_video_pre["tokens"], video_pre["tokens"]], dim=1)
                combined_video_freqs = torch.cat([task_video_pre["freqs"], video_pre["freqs"]], dim=0)
                combined_video_t_mod = torch.cat([task_video_pre["t_mod"], video_pre["t_mod"]], dim=1)
                combined_video_context = video_pre["context"]
                combined_video_context_mask = torch.cat(
                    [task_video_pre["context_mask"], video_pre["context_mask"]], dim=1
                )

            elif self.task_video_conditioning_mode == "cross_attn":
                # Separate cross-attn branches: task_video and text are handled by dedicated
                # cross-attn layers (task_video first, then text). Self-attn sequence is
                # [progress | agent_video | action]; task_video_seq_len stays 0.
                tv_tokens = task_video_pre["tokens"]  # [B, S_task, hidden_dim]
                tv_freqs  = task_video_pre["freqs"]   # RoPE freqs for task video K
                combined_video_tokens = video_pre["tokens"]
                combined_video_freqs = video_pre["freqs"]
                combined_video_t_mod = video_pre["t_mod"]
                # Text context unchanged — no concat with task_video
                combined_video_context = video_pre["context"]       # [B, L, hidden_dim]
                combined_video_context_mask = video_pre["context_mask"]  # [B, S_vid, L]
                # tv_attn_mask_vid built after progress prepend (sequence length changes)

            elif self.task_video_conditioning_mode == "prepend_cross_attn":
                # Token layout same as prepend: [task_video | agent_video] in self-attn.
                # Additionally, each layer extracts [progress|task_video] from video expert
                # output as dynamic cross-attn context for both experts.
                task_video_seq_len = task_video_pre["tokens"].shape[1]
                combined_video_tokens = torch.cat([task_video_pre["tokens"], video_pre["tokens"]], dim=1)
                combined_video_freqs = torch.cat([task_video_pre["freqs"], video_pre["freqs"]], dim=0)
                combined_video_t_mod = torch.cat([task_video_pre["t_mod"], video_pre["t_mod"]], dim=1)
                combined_video_context = video_pre["context"]
                combined_video_context_mask = torch.cat(
                    [task_video_pre["context_mask"], video_pre["context_mask"]], dim=1
                )
                # k_freqs for cross-attn sliced from combined_video_freqs after _prepend_progress_token

            else:
                raise ValueError(
                    f"Unknown task_video_conditioning_mode: '{self.task_video_conditioning_mode}'"
                )
        else:
            combined_video_tokens = video_pre["tokens"]
            combined_video_freqs = video_pre["freqs"]
            combined_video_t_mod = video_pre["t_mod"]
            combined_video_context = video_pre["context"]
            combined_video_context_mask = video_pre["context_mask"]

        # Prepend progress token to video expert sequence (when progress_gt is available).
        # Progress token is assigned 3D RoPE position (f_task+f_agent, 0, 0) — unique, past all frames.
        # Shape: combined_video_tokens [B, 1+S_video, D], combined_video_freqs [1+S_video, 1, d]
        progress_seq_len = 0
        if latents_progress is not None:
            progress_seq_len = 1
            f_task_meta = task_video_pre["meta"]["grid_size"][0] if task_video_latents is not None else 0
            (
                combined_video_tokens,
                combined_video_freqs,
                combined_video_t_mod,
                combined_video_context_mask,
            ) = self._prepend_progress_token(
                latents_progress=latents_progress,
                combined_video_tokens=combined_video_tokens,
                combined_video_freqs=combined_video_freqs,
                combined_video_t_mod=combined_video_t_mod,
                combined_video_context_mask=combined_video_context_mask,
                video_pre=video_pre,
                f_task_meta=f_task_meta,
            )

        # Noisy progress group: build isolated tokens and append to video expert sequence.
        # Must be after _prepend_progress_token and task_video_pre.
        noisy_group_seq_len = 0
        target_noisy_p = None
        timestep_noisy_p = None
        if (self.use_noisy_progress_group
                and task_video_latents is not None
                and progress_gt is not None):
            progress_gt_normed = progress_gt * 2.0 - 1.0  # [B, 2] → [-1, 1]
            noise_p = torch.randn_like(progress_gt_normed)
            timestep_noisy_p = self.train_progress_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=progress_gt_normed.dtype,
            )
            latents_noisy_p = self.train_progress_scheduler.add_noise(
                progress_gt_normed, noise_p, timestep_noisy_p,
            )
            target_noisy_p = self.train_progress_scheduler.training_target(
                progress_gt_normed, noise_p, timestep_noisy_p,
                prediction_type=self.progress_prediction_type,
            )
            tpf = int(video_pre["meta"]["tokens_per_frame"])
            f_task_meta_ng = task_video_pre["meta"]["grid_size"][0]
            ng_tokens, ng_freqs, ng_t_mod, ng_cmask = self._build_noisy_group(
                latents_noisy_progress=latents_noisy_p,
                timestep_progress=timestep_noisy_p,
                task_video_pre=task_video_pre,
                video_pre=video_pre,
                video_tokens_per_frame=tpf,
                f_task_meta=f_task_meta_ng,
                f_agent_meta=video_pre["meta"]["grid_size"][0],
            )
            noisy_group_seq_len = ng_tokens.shape[1]  # 1 + S_tv + S_1f
            combined_video_tokens = torch.cat([combined_video_tokens, ng_tokens], dim=1)
            combined_video_freqs  = torch.cat([combined_video_freqs,  ng_freqs],  dim=0)
            combined_video_t_mod  = torch.cat([combined_video_t_mod,  ng_t_mod],  dim=1)
            combined_video_context_mask = torch.cat([combined_video_context_mask, ng_cmask], dim=1)

        # Build task_video attention mask for cross_attn mode (after progress prepend so dims are correct)
        tv_attn_mask_vid: Optional[torch.Tensor] = None
        tv_attn_mask_act: Optional[torch.Tensor] = None
        if (self.task_video_conditioning_mode == "cross_attn"
                and task_video_latents is not None):
            S_vid_final = combined_video_tokens.shape[1]
            B_ = batch_size
            tv_attn_mask_vid = torch.ones(
                (B_, S_vid_final, tv_tokens.shape[1]),
                dtype=torch.bool, device=tv_tokens.device,
            )
            if task_video_dropped is not None:
                for b in range(B_):
                    if task_video_dropped[b]:
                        tv_attn_mask_vid[b] = False

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        # Build action-side task video attention mask for cross_attn mode
        if (self.task_video_conditioning_mode == "cross_attn"
                and task_video_latents is not None):
            B_ = batch_size
            S_action = action_tokens.shape[1]
            tv_attn_mask_act = torch.ones(
                (B_, S_action, tv_tokens.shape[1]),
                dtype=torch.bool, device=tv_tokens.device,
            )
            if task_video_dropped is not None:
                for b in range(B_):
                    if task_video_dropped[b]:
                        tv_attn_mask_act[b] = False

        # Build masks and freqs for prepend_cross_attn mode.
        # Must be after _prepend_progress_token (combined_video_freqs[0] = progress freq)
        # and after action_pre (action_tokens available for S_action).
        _pac_mode = (
            self.task_video_conditioning_mode == "prepend_cross_attn"
            and task_video_latents is not None
        )
        pac_k_freqs: Optional[torch.Tensor] = None
        pac_slice: Optional[tuple] = None
        if _pac_mode:
            # [progress|task_video] dynamic context length
            pac_ctx_len = progress_seq_len + task_video_seq_len
            # combined_video_freqs after _prepend_progress_token:
            #   [0]        = progress freq  (f_tv+f_ag, 0, 0)
            #   [1:1+tv]   = task_video_pre["freqs"]  (identical to cross_attn mode tv_freqs)
            pac_k_freqs = combined_video_freqs[:pac_ctx_len]  # [p+tv, 1, freq_dim]
            pac_slice = (0, pac_ctx_len)

            S_vid_final = combined_video_tokens.shape[1]  # p + tv + S_agent
            S_action = action_tokens.shape[1]
            B_ = batch_size

            # [B, Q_vid, K_ctx]: each video token can attend to [p|tv] unless dropped
            tv_attn_mask_vid = torch.ones(
                (B_, S_vid_final, pac_ctx_len), dtype=torch.bool,
                device=combined_video_tokens.device,
            )
            if task_video_dropped is not None:
                for b in range(B_):
                    if task_video_dropped[b]:
                        tv_attn_mask_vid[b] = False
            # Block task-video cross-attention for noisy group tokens
            if noisy_group_seq_len > 0:
                ng_start_in_vid = progress_seq_len + task_video_seq_len + video_pre["tokens"].shape[1]
                tv_attn_mask_vid[:, ng_start_in_vid:, :] = False

            # [B, Q_act, K_ctx]: each action token can attend to [p|tv] unless dropped
            tv_attn_mask_act = torch.ones(
                (B_, S_action, pac_ctx_len), dtype=torch.bool,
                device=combined_video_tokens.device,
            )
            if task_video_dropped is not None:
                for b in range(B_):
                    if task_video_dropped[b]:
                        tv_attn_mask_act[b] = False

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            task_video_seq_len=task_video_seq_len,
            progress_seq_len=progress_seq_len,
            noisy_group_seq_len=noisy_group_seq_len,
        )
        # Per-sample attention mask: expand [S,S] → [B,1,S,S] and block
        # progress/video → task_video attention for dropped samples
        if (task_video_dropped is not None
                and task_video_seq_len > 0
                and any(task_video_dropped)):
            B = batch_size
            p_end = progress_seq_len
            tv_end = p_end + task_video_seq_len
            v_end = tv_end + video_tokens.shape[1]
            ng_end = v_end + noisy_group_seq_len
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()
            _drop_mask = torch.tensor(task_video_dropped, dtype=torch.bool, device=attention_mask.device)
            for b in range(B):
                if _drop_mask[b]:
                    attention_mask[b, :, :, p_end:tv_end] = False  # nothing attends to task_video
                    if noisy_group_seq_len > 0:
                        attention_mask[b, :, :, v_end:ng_end] = False  # block noisy group as keys
                        attention_mask[b, :, v_end:ng_end, :] = False  # noisy group attends to nothing

        _cross_attn_tv_static = (
            self.task_video_conditioning_mode == "cross_attn"
            and task_video_latents is not None
        )
        _cross_attn_tv_dynamic = _pac_mode  # prepend_cross_attn: dynamic per-layer extraction
        tokens_out = self.mot(
            embeds_all={
                "video": combined_video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": combined_video_freqs,
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": combined_video_context,
                    "mask": combined_video_context_mask,
                    **(
                        {
                            "task_video_context": tv_tokens,
                            "task_video_mask":    tv_attn_mask_vid,
                            "q_freqs":            combined_video_freqs,
                            "task_video_freqs":   tv_freqs,
                            **({"task_video_dropped_mask": task_video_dropped} if task_video_dropped is not None else {}),
                        }
                        if _cross_attn_tv_static else {}
                    ),
                    **(
                        {
                            # Signal dynamic extraction inside _apply_expert_post_block
                            "prepend_cross_attn_slice": pac_slice,
                            "task_video_mask":          tv_attn_mask_vid,
                            "q_freqs":                  combined_video_freqs,
                            "task_video_freqs":         pac_k_freqs,
                            **({"task_video_dropped_mask": task_video_dropped} if task_video_dropped is not None else {}),
                            # Zero out task-video cross-attn output for noisy group tokens
                            # to match inference Stage 1 (which skips task-video CA entirely).
                            **(
                                {
                                    "noisy_group_zero_range": (
                                        progress_seq_len + task_video_seq_len + video_pre["tokens"].shape[1],
                                        progress_seq_len + task_video_seq_len + video_pre["tokens"].shape[1] + noisy_group_seq_len,
                                    ),
                                }
                                if noisy_group_seq_len > 0 else {}
                            ),
                        }
                        if _cross_attn_tv_dynamic else {}
                    ),
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                    **(
                        {
                            "task_video_context": tv_tokens,
                            "task_video_mask":    tv_attn_mask_act,
                            "q_freqs":            action_pre["freqs"],
                            "task_video_freqs":   tv_freqs,
                            **({"task_video_dropped_mask": task_video_dropped} if task_video_dropped is not None else {}),
                        }
                        if _cross_attn_tv_static else {}
                    ),
                    **(
                        {
                            # task_video_context injected per-layer by MoT.forward
                            "task_video_mask":    tv_attn_mask_act,
                            "q_freqs":            action_pre["freqs"],
                            "task_video_freqs":   pac_k_freqs,
                            **({"task_video_dropped_mask": task_video_dropped} if task_video_dropped is not None else {}),
                        }
                        if _cross_attn_tv_dynamic else {}
                    ),
                },
            },
            t_mod_all={
                "video": combined_video_t_mod,
                "action": action_pre["t_mod"],
            },
            prepend_cross_attn_slice=pac_slice if _cross_attn_tv_dynamic else None,
        )

        # Split combined video output: [progress | task_video | agent_video | noisy_group]
        # progress token is at position 0 (if present), then task_video, then agent_video,
        # then noisy group (if present).
        video_out_full = tokens_out["video"]
        if progress_seq_len > 0:
            # Extract clean progress output token (position 0)
            progress_out_token = video_out_full[:, 0, :]  # [B, D]
            video_out_rest = video_out_full[:, 1:]        # [B, tv+video+noisy_group, D]
        else:
            progress_out_token = None
            video_out_rest = video_out_full

        # Extract noisy progress output (if noisy group exists)
        noisy_progress_out = None
        if noisy_group_seq_len > 0:
            # video_out_rest layout: [tv(S_tv) | video(S_v) | noisy_p(1) | tv_noisy(S_tv) | ff_noisy(S_1f)]
            noisy_p_idx = task_video_seq_len + video_pre["tokens"].shape[1]
            noisy_progress_out = video_out_rest[:, noisy_p_idx, :]  # [B, D]
            # Trim noisy group from video_out_rest to get [tv | video] only
            video_out_rest = video_out_rest[:, :noisy_p_idx]

        # Discard task_video output; only agent_video output feeds pred_video
        video_out = video_out_rest[:, task_video_seq_len:]  # [B, S_agent, D]

        pred_video = self.video_expert.post_dit(video_out, video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        # Progress flow matching loss:
        # - Original mode (noise_to_progress_token=True): decode clean progress output token
        # - Noisy group mode (use_noisy_progress_group=True): decode noisy progress output token
        loss_progress = torch.tensor(0.0, device=loss_video.device)
        if noisy_progress_out is not None and target_noisy_p is not None:
            # Noisy group mode: loss from the isolated noisy progress token
            pred_progress_fm = self.progress_decoder(noisy_progress_out)  # [B, 2]
            assert pred_progress_fm.shape == (batch_size, 2), (
                f"pred_progress_fm shape mismatch: {pred_progress_fm.shape}"
            )
            progress_weight = self.train_progress_scheduler.training_weight(timestep_noisy_p).to(
                pred_progress_fm.device, dtype=pred_progress_fm.dtype
            )
            loss_progress_per_sample = F.mse_loss(
                pred_progress_fm.float(), target_noisy_p.float(), reduction="none"
            ).mean(dim=-1)  # [B]
            # Zero loss for samples where task_video was dropped
            if task_video_dropped is not None:
                tv_valid = torch.tensor(
                    [not d for d in task_video_dropped],
                    dtype=loss_progress_per_sample.dtype,
                    device=loss_progress_per_sample.device,
                )
                loss_progress_per_sample = loss_progress_per_sample * tv_valid
            loss_progress = (loss_progress_per_sample * progress_weight).mean()
        elif progress_out_token is not None and target_progress is not None:
            # Original mode: decode from main group's progress output token
            pred_progress_fm = self.progress_decoder(progress_out_token)  # [B, 2]
            assert pred_progress_fm.shape == (batch_size, 2), (
                f"pred_progress_fm shape mismatch: {pred_progress_fm.shape}"
            )
            progress_weight = self.train_progress_scheduler.training_weight(timestep_progress).to(
                pred_progress_fm.device, dtype=pred_progress_fm.dtype
            )
            loss_progress_per_sample = F.mse_loss(
                pred_progress_fm.float(), target_progress.float(), reduction="none"
            ).mean(dim=-1)  # [B]
            loss_progress = (loss_progress_per_sample * progress_weight).mean()

        loss_total = (self.loss_lambda_video * loss_video
                      + self.loss_lambda_action * loss_action
                      + self.loss_lambda_progress * loss_progress)
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_progress": self.loss_lambda_progress * float(loss_progress.detach().item()),
            # has_task_video sentinel: 1.0 = all have TV, nan = all dropped, 0.5 = mixed (skip classification)
            "has_task_video": (
                float("nan") if (task_video_dropped is not None and all(task_video_dropped))
                else 1.0 if (task_video_dropped is None or not any(task_video_dropped))
                else 0.5
            ),
            "rho_video_to_task": float("nan"),  # overwritten by attn_stats when task_video present (prepend mode only)
        }
        # Pre-declare cross-attn metric keys so all ranks always have the same key set,
        # regardless of whether gradient checkpointing suppressed weight capture this step.
        if self.task_video_conditioning_mode == "cross_attn" and self.log_cross_attn_weights:
            loss_dict["rho_text_attend"]       = float("nan")
            loss_dict["rho_task_video_attend"] = float("nan")

        # Per-key weighted flow MSE (logging only, no backward)
        if action_component_slices:
            for name, (start, end) in action_component_slices.items():
                key_pred  = pred_action.float()[..., start:end]   # [B, T, dim]
                key_tgt   = target_action.float()[..., start:end]
                key_mse_t = F.mse_loss(key_pred, key_tgt, reduction="none").mean(dim=2)  # [B, T]
                if action_is_pad is not None:
                    valid = (~action_is_pad).to(device=key_mse_t.device, dtype=key_mse_t.dtype)
                    valid_sum = valid.sum(dim=1).clamp(min=1.0)
                    key_loss_per_sample = (key_mse_t * valid).sum(dim=1) / valid_sum  # [B]
                else:
                    key_loss_per_sample = key_mse_t.mean(dim=1)  # [B]
                key_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                    key_loss_per_sample.device, dtype=key_loss_per_sample.dtype
                )
                loss_dict[f"loss_action/{name}"] = float(
                    (key_loss_per_sample * key_weight).mean().detach().item()
                )

        # Attention stats (last layer, training only)
        if task_video_latents is not None:
            if self.task_video_conditioning_mode == "prepend":
                # Self-attn rho: agent_video → task_video
                last_attn = getattr(self.mot, '_last_attn_weights', None)
                if last_attn is not None and task_video_seq_len > 0:
                    attn_stats = self._compute_task_video_attn_stats(
                        last_attn, task_video_seq_len, video_tokens.shape[1],
                        progress_seq_len=progress_seq_len,
                    )
                    loss_dict.update(attn_stats)

            elif self.task_video_conditioning_mode == "cross_attn" and self.log_cross_attn_weights:
                # Cross-attn rho: agent_video → text (text-only context, no concat)
                # Only populated when log_cross_attn_weights=True AND not gradient-checkpointing
                cross_weights = getattr(self.mot, '_last_video_cross_attn_weights', None)
                if cross_weights is not None:
                    # cross_weights: [B, H, S_vid, L] (text context only)
                    loss_dict["rho_text_attend"] = cross_weights.sum(dim=-1).mean().item()

        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        task_video_pre: Optional[dict] = None,
        latents_progress: Optional[torch.Tensor] = None,
        task_video_dropped: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = latents_video.shape[0]
        # [B, 48, T_lat, H/8, W/8] → [B, S_video, hidden_dim]
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        # Assemble video tokens according to task_video_conditioning_mode
        task_video_seq_len = 0
        tv_attn_mask_vid: Optional[torch.Tensor] = None
        tv_attn_mask_act: Optional[torch.Tensor] = None
        if task_video_pre is not None:
            if self.task_video_conditioning_mode == "prepend":
                # Prepend task_video tokens to self-attn sequence
                task_video_seq_len = task_video_pre["tokens"].shape[1]
                combined_video_tokens = torch.cat([task_video_pre["tokens"], video_pre["tokens"]], dim=1)
                combined_video_freqs = torch.cat([task_video_pre["freqs"], video_pre["freqs"]], dim=0)
                combined_video_t_mod = torch.cat([task_video_pre["t_mod"], video_pre["t_mod"]], dim=1)
                combined_video_context = video_pre["context"]
                combined_video_context_mask = torch.cat(
                    [task_video_pre["context_mask"], video_pre["context_mask"]], dim=1
                )

            elif self.task_video_conditioning_mode == "cross_attn":
                # Separate cross-attn branches: task_video and text handled independently.
                tv_tokens = task_video_pre["tokens"]  # [B, S_task, hidden_dim]
                tv_freqs  = task_video_pre["freqs"]
                combined_video_tokens = video_pre["tokens"]
                combined_video_freqs = video_pre["freqs"]
                combined_video_t_mod = video_pre["t_mod"]
                # Text context unchanged — no concat
                combined_video_context = video_pre["context"]
                combined_video_context_mask = video_pre["context_mask"]
                # tv_attn_mask_vid built after progress prepend

            elif self.task_video_conditioning_mode == "prepend_cross_attn":
                task_video_seq_len = task_video_pre["tokens"].shape[1]
                combined_video_tokens = torch.cat([task_video_pre["tokens"], video_pre["tokens"]], dim=1)
                combined_video_freqs  = torch.cat([task_video_pre["freqs"],  video_pre["freqs"]],  dim=0)
                combined_video_t_mod  = torch.cat([task_video_pre["t_mod"],  video_pre["t_mod"]],  dim=1)
                combined_video_context = video_pre["context"]
                combined_video_context_mask = torch.cat(
                    [task_video_pre["context_mask"], video_pre["context_mask"]], dim=1
                )

            else:
                raise ValueError(
                    f"Unknown task_video_conditioning_mode: '{self.task_video_conditioning_mode}'"
                )
        else:
            combined_video_tokens = video_pre["tokens"]
            combined_video_freqs = video_pre["freqs"]
            combined_video_t_mod = video_pre["t_mod"]
            combined_video_context = video_pre["context"]
            combined_video_context_mask = video_pre["context_mask"]

        # Prepend progress token to video expert sequence (when latents_progress provided)
        progress_seq_len = 0
        if latents_progress is not None:
            progress_seq_len = 1
            f_task_meta = task_video_pre["meta"]["grid_size"][0] if task_video_pre is not None else 0
            (
                combined_video_tokens,
                combined_video_freqs,
                combined_video_t_mod,
                combined_video_context_mask,
            ) = self._prepend_progress_token(
                latents_progress=latents_progress,
                combined_video_tokens=combined_video_tokens,
                combined_video_freqs=combined_video_freqs,
                combined_video_t_mod=combined_video_t_mod,
                combined_video_context_mask=combined_video_context_mask,
                video_pre=video_pre,
                f_task_meta=f_task_meta,
            )

        # Build cross-attn masks for task_video (after progress prepend, so dimensions are correct)
        tv_attn_mask_vid: Optional[torch.Tensor] = None
        tv_attn_mask_act: Optional[torch.Tensor] = None
        _cross_attn_tv_static = (
            self.task_video_conditioning_mode == "cross_attn"
            and task_video_pre is not None
        )
        if _cross_attn_tv_static:
            S_vid_final = combined_video_tokens.shape[1]
            S_action = action_pre["tokens"].shape[1]
            tv_attn_mask_vid = torch.ones(
                (B, S_vid_final, tv_tokens.shape[1]),
                dtype=torch.bool, device=tv_tokens.device,
            )
            tv_attn_mask_act = torch.ones(
                (B, S_action, tv_tokens.shape[1]),
                dtype=torch.bool, device=tv_tokens.device,
            )
            if task_video_dropped:
                tv_attn_mask_vid[:] = False
                tv_attn_mask_act[:] = False

        # prepend_cross_attn masks
        _pac_mode = (
            self.task_video_conditioning_mode == "prepend_cross_attn"
            and task_video_pre is not None
        )
        pac_k_freqs: Optional[torch.Tensor] = None
        pac_slice: Optional[tuple] = None
        if _pac_mode:
            pac_ctx_len = progress_seq_len + task_video_seq_len
            pac_k_freqs = combined_video_freqs[:pac_ctx_len]
            pac_slice = (0, pac_ctx_len)
            S_vid_final = combined_video_tokens.shape[1]
            S_action = action_pre["tokens"].shape[1]
            tv_attn_mask_vid = torch.ones(
                (B, S_vid_final, pac_ctx_len),
                dtype=torch.bool, device=combined_video_tokens.device,
            )
            tv_attn_mask_act = torch.ones(
                (B, S_action, pac_ctx_len),
                dtype=torch.bool, device=combined_video_tokens.device,
            )
            if task_video_dropped:
                tv_attn_mask_vid[:] = False
                tv_attn_mask_act[:] = False
        _cross_attn_tv_dynamic = _pac_mode

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            task_video_seq_len=task_video_seq_len,
            progress_seq_len=progress_seq_len,
        )
        # Block self-attn to task_video for dropped samples (matches training_loss 4D mask)
        if task_video_dropped and task_video_seq_len > 0:
            p_end = progress_seq_len
            tv_end = p_end + task_video_seq_len
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()
            attention_mask[:, :, :, p_end:tv_end] = False

        tokens_out = self.mot(
            embeds_all={
                "video": combined_video_tokens,
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": combined_video_freqs,
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": combined_video_context,
                    "mask": combined_video_context_mask,
                    **(
                        {
                            "task_video_context": tv_tokens,
                            "task_video_mask":    tv_attn_mask_vid,
                            "q_freqs":            combined_video_freqs,
                            "task_video_freqs":   tv_freqs,
                            **({"task_video_dropped_mask": [task_video_dropped]} if task_video_dropped else {}),
                        }
                        if _cross_attn_tv_static else {}
                    ),
                    **(
                        {
                            "prepend_cross_attn_slice": pac_slice,
                            "task_video_mask":          tv_attn_mask_vid,
                            "q_freqs":                  combined_video_freqs,
                            "task_video_freqs":         pac_k_freqs,
                            **({"task_video_dropped_mask": [task_video_dropped]} if task_video_dropped else {}),
                        }
                        if _cross_attn_tv_dynamic else {}
                    ),
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                    **(
                        {
                            "task_video_context": tv_tokens,
                            "task_video_mask":    tv_attn_mask_act,
                            "q_freqs":            action_pre["freqs"],
                            "task_video_freqs":   tv_freqs,
                            **({"task_video_dropped_mask": [task_video_dropped]} if task_video_dropped else {}),
                        }
                        if _cross_attn_tv_static else {}
                    ),
                    **(
                        {
                            "task_video_mask":    tv_attn_mask_act,
                            "q_freqs":            action_pre["freqs"],
                            "task_video_freqs":   pac_k_freqs,
                            **({"task_video_dropped_mask": [task_video_dropped]} if task_video_dropped else {}),
                        }
                        if _cross_attn_tv_dynamic else {}
                    ),
                },
            },
            t_mod_all={
                "video": combined_video_t_mod,
                "action": action_pre["t_mod"],
            },
            prepend_cross_attn_slice=pac_slice if _cross_attn_tv_dynamic else None,
        )

        # Split [progress | task_video | agent_video] output; only agent_video feeds pred_video
        video_out_full = tokens_out["video"]
        if progress_seq_len > 0:
            progress_out_token = video_out_full[:, 0, :]  # [B, D]
            video_out = video_out_full[:, 1 + task_video_seq_len:]  # [B, S_agent, D]
        else:
            progress_out_token = None
            video_out = video_out_full[:, task_video_seq_len:]  # [B, S_agent, D]

        pred_video = self.video_expert.post_dit(video_out, video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # Decode progress token → predicted velocity/sample
        if progress_out_token is not None:
            pred_progress = self.progress_decoder(progress_out_token)  # [B, 2]
        else:
            raise RuntimeError(
                "`_predict_joint_noise` expected a progress token but `latents_progress` was None. "
                "Pass `latents_progress` or handle the no-progress case at the call site."
            )

        return pred_video, pred_action, pred_progress

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def _infer_progress_stage1(
        self,
        task_video_pre: dict,
        ff_infer_pre: dict,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int,
        f_agent_meta: int,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Stage 1: Denoise progress using only the isolated noisy group.

        Runs a mini denoising loop through ``MoT.forward_single_expert("video", ...)``
        with only the noisy group tokens: [noisy_p | tv_noisy | ff_noisy].
        Text cross-attention is allowed; task-video cross-attention is skipped
        (no ``prepend_cross_attn_slice`` in context_payload).

        Args:
            task_video_pre: Cached pre_dit output for task video (timestep=0).
            ff_infer_pre: Cached pre_dit output for first frame (timestep=0).
            context: [B, L, D] text embeddings.
            context_mask: [B, L] text mask.
            num_inference_steps: Number of denoising steps for progress.
            sigma_shift: Optional override for scheduler shift.
            seed: Optional seed for reproducibility.

        Returns:
            predicted_progress: [B, 2] denoised progress in [-1, 1].
        """
        B = 1
        rand_device = self.device
        progress_generator = None
        if seed is not None:
            progress_generator = torch.Generator(device=rand_device).manual_seed(seed)
        latents_progress = torch.randn(
            (B, 2), generator=progress_generator,
            device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_ts, infer_deltas = self.infer_progress_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_progress.dtype,
            shift_override=sigma_shift,
        )
        f_task_meta = task_video_pre["meta"]["grid_size"][0]
        tpf = int(ff_infer_pre["meta"]["tokens_per_frame"])
        _n_steps = len(infer_ts)

        for step_i, (step_t, step_delta) in enumerate(zip(infer_ts, infer_deltas)):
            is_last = (step_i == _n_steps - 1)
            timestep_p = step_t.unsqueeze(0).to(
                dtype=latents_progress.dtype, device=self.device,
            )

            ng_tokens, ng_freqs, ng_t_mod, ng_cmask = self._build_noisy_group(
                latents_noisy_progress=latents_progress,
                timestep_progress=timestep_p,
                task_video_pre=task_video_pre,
                video_pre=ff_infer_pre,
                video_tokens_per_frame=tpf,
                f_task_meta=f_task_meta,
                f_agent_meta=f_agent_meta,
            )

            # All-to-all self-attention within group
            S = ng_tokens.shape[1]
            attn_mask = torch.ones((S, S), dtype=torch.bool, device=self.device)

            # Text CA only — no task-video CA (no pac_slice in payload)
            # Use projected context (hidden_dim) from task_video_pre, not raw text (text_dim).
            context_payload = {
                "context": task_video_pre["context"],
                "mask": ng_cmask,
            }

            out = self.mot.forward_single_expert(
                expert_name="video",
                tokens=ng_tokens,
                freqs=ng_freqs,
                t_mod=ng_t_mod,
                context_payload=context_payload,
                attention_mask=attn_mask,
            )

            # Noisy progress is at position 0
            pred_progress = self.progress_decoder(out[:, 0, :])  # [B, 2]

            latents_progress = self.infer_progress_scheduler.step(
                pred_progress, step_delta, latents_progress,
                timestep=timestep_p,
                prediction_type=self.progress_prediction_type,
                is_last_step=is_last,
            )

        return latents_progress  # [B, 2]

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        task_video: Optional[torch.Tensor] = None,
        progress_override: Optional[torch.Tensor] = None,
        task_video_dropped: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        # Skip infer_action consistency check when using progress override (oracle mode)
        if progress_override is not None:
            test_action_with_infer_action = False
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        progress_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_progress = torch.randn(
            (1, 2),
            generator=progress_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        if self.visual_encoder is not None:
            context, context_mask = self._append_visual_to_context(
                context=context,
                context_mask=context_mask,
                first_frame_rgb=input_image,
            )

        # Pre-compute task_video tokens once (timestep=0, constant across all denoising steps).
        # Mirrors training_loss(): task_video_pre is computed with t=0 (clean conditioning).
        task_video_pre_cache = None
        if task_video is not None:
            tv = task_video if task_video.ndim == 5 else task_video.unsqueeze(0)
            tv = tv.to(device=self.device, dtype=self.torch_dtype)
            task_video_latents = self._encode_video_latents(tv, tiled=tiled)
            task_video_pre_cache = self.video_expert.pre_dit(
                x=task_video_latents,
                timestep=torch.zeros((1,), device=self.device, dtype=self.torch_dtype),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=True,
            )

        # Two-stage inference with noisy progress group:
        # Stage 1: denoise progress using isolated noisy group, then
        # Stage 2: use predicted progress as fixed clean conditioning for video+action.
        if self.use_noisy_progress_group and task_video_pre_cache is not None:
            if progress_override is not None:
                # Oracle mode: skip Stage 1, use provided progress directly ([-1, 1] range)
                latents_progress = progress_override.to(device=self.device, dtype=self.torch_dtype)
                if latents_progress.ndim == 1:
                    latents_progress = latents_progress.unsqueeze(0)  # [1, 2]
            elif task_video_dropped:
                # Skip Stage 1 — training zeroes progress loss for dropped samples,
                # so the model never learned to predict progress from zero task_video.
                # Use [-1, -1] matching training's progress_gt_normed for dropped samples.
                latents_progress = torch.tensor(
                    [[-1.0, -1.0]], device=self.device, dtype=self.torch_dtype
                )
            else:
                # Pre-compute first frame encoding at timestep=0 for noisy group
                ff_infer_pre_cache = self.video_expert.pre_dit(
                    x=first_frame_latents,
                    timestep=torch.zeros((1,), device=self.device, dtype=self.torch_dtype),
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=True,
                )
                # Stage 1: denoise progress
                _n_steps_progress = self.num_inference_steps_progress or num_inference_steps
                latents_progress = self._infer_progress_stage1(
                    task_video_pre=task_video_pre_cache,
                    ff_infer_pre=ff_infer_pre_cache,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=_n_steps_progress,
                    f_agent_meta=latent_t,
                    sigma_shift=sigma_shift,
                    seed=seed,
                )
            # Stage 2: standard video+action denoising with fixed clean progress
            infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            _n_steps = len(infer_timesteps_video)
            for _step_i, (
                step_t_video, step_delta_video,
                step_t_action, step_delta_action,
            ) in enumerate(zip(
                infer_timesteps_video, infer_deltas_video,
                infer_timesteps_action, infer_deltas_action,
            )):
                timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                _is_last = (_step_i == _n_steps - 1)

                pred_video_posi, pred_action_posi, _ = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=action,
                    task_video_pre=task_video_pre_cache,
                    latents_progress=latents_progress,  # fixed clean progress from Stage 1
                    task_video_dropped=task_video_dropped,
                )

                latents_video = self.infer_video_scheduler.step(
                    pred_video_posi, step_delta_video, latents_video,
                    timestep=timestep_video, prediction_type=self.video_prediction_type,
                    is_last_step=_is_last,
                )
                latents_action = self.infer_action_scheduler.step(
                    pred_action_posi, step_delta_action, latents_action,
                    timestep=timestep_action, prediction_type=self.action_prediction_type,
                    is_last_step=_is_last,
                )
                # NOTE: latents_progress NOT updated — it is the fixed clean prediction from Stage 1
                latents_video[:, :, 0:1] = first_frame_latents.clone()
        else:
            # Original path: synchronized 3-stream denoising
            _progress_is_fixed = (progress_override is not None)
            if _progress_is_fixed:
                # Oracle mode: use provided progress as fixed conditioning
                latents_progress = progress_override.to(device=self.device, dtype=self.torch_dtype)
                if latents_progress.ndim == 1:
                    latents_progress = latents_progress.unsqueeze(0)  # [1, 2]
            if task_video_dropped and not self.noise_to_progress_token:
                # Training uses clean fixed progress for dropped samples when
                # noise_to_progress_token=False; match at inference.
                latents_progress = torch.tensor(
                    [[-1.0, -1.0]], device=self.device, dtype=self.torch_dtype
                )
                _progress_is_fixed = True
            infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            infer_timesteps_progress, infer_deltas_progress = self.infer_progress_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_progress.dtype,
                shift_override=sigma_shift,
            )
            _n_steps = len(infer_timesteps_video)
            for _step_i, (
                step_t_video, step_delta_video,
                step_t_action, step_delta_action,
                step_t_progress, step_delta_progress,
            ) in enumerate(zip(
                infer_timesteps_video, infer_deltas_video,
                infer_timesteps_action, infer_deltas_action,
                infer_timesteps_progress, infer_deltas_progress,
            )):
                timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                _is_last = (_step_i == _n_steps - 1)

                pred_video_posi, pred_action_posi, pred_progress_posi = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=action,
                    task_video_pre=task_video_pre_cache,
                    latents_progress=latents_progress,
                    task_video_dropped=task_video_dropped,
                )

                latents_video = self.infer_video_scheduler.step(
                    pred_video_posi, step_delta_video, latents_video,
                    timestep=timestep_video, prediction_type=self.video_prediction_type,
                    is_last_step=_is_last,
                )
                latents_action = self.infer_action_scheduler.step(
                    pred_action_posi, step_delta_action, latents_action,
                    timestep=timestep_action, prediction_type=self.action_prediction_type,
                    is_last_step=_is_last,
                )
                if not _progress_is_fixed:
                    latents_progress = self.infer_progress_scheduler.step(
                        pred_progress_posi, step_delta_progress, latents_progress,
                        timestep=step_t_progress.unsqueeze(0).to(dtype=latents_progress.dtype, device=self.device),
                        prediction_type=self.progress_prediction_type,
                        is_last_step=_is_last,
                    )
                latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
            "progress": latents_progress[0].detach().to(device="cpu", dtype=torch.float32),  # [2]
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        # NOTE: Train-inference gap — this action-only path does not inject the progress token.
        # During training, the progress token participates in video expert self-attention whenever
        # progress_gt is present. Use infer_joint for production inference to avoid this gap.
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        if self.visual_encoder is not None:
            context, context_mask = self._append_visual_to_context(
                context=context,
                context_mask=context_mask,
                first_frame_rgb=input_image,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        _n_steps_action = len(infer_timesteps_action)
        for _step_i, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action)
        ):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action,
                timestep=timestep_action, prediction_type=self.action_prediction_type,
                is_last_step=(_step_i == _n_steps_action - 1),
            )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        task_video: Optional[torch.Tensor] = None,
        progress_override: Optional[torch.Tensor] = None,
        task_video_dropped: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            task_video=task_video,
            progress_override=progress_override,
            task_video_dropped=task_video_dropped,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        payload["progress_encoder"] = self.progress_encoder.state_dict()
        payload["progress_decoder"] = self.progress_decoder.state_dict()
        if self.visual_encoder is not None:
            payload["visual_encoder"] = self.visual_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            result = self.mot.load_state_dict(payload["mot"], strict=False)
            if result.missing_keys:
                logger.warning("mot load_state_dict missing %d keys: %s", len(result.missing_keys), result.missing_keys)
            if result.unexpected_keys:
                logger.warning("mot load_state_dict unexpected %d keys: %s", len(result.unexpected_keys), result.unexpected_keys)
            if not result.missing_keys and not result.unexpected_keys:
                logger.info("mot load_state_dict: all %d keys matched exactly.", len(payload["mot"]))
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            if result.missing_keys:
                logger.warning("dit load_state_dict missing %d keys: %s", len(result.missing_keys), result.missing_keys)
            if result.unexpected_keys:
                logger.warning("dit load_state_dict unexpected %d keys: %s", len(result.unexpected_keys), result.unexpected_keys)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if "progress_head" in payload:
            logger.warning(
                "Checkpoint has legacy 'progress_head' key (direct-regression mode). "
                "This checkpoint is not compatible with flow-matching progress. Ignoring 'progress_head'."
            )
        if "progress_encoder" in payload:
            self.progress_encoder.load_state_dict(payload["progress_encoder"], strict=True)
        else:
            logger.warning("Checkpoint has no 'progress_encoder' weights; keeping random init.")
        if "progress_decoder" in payload:
            self.progress_decoder.load_state_dict(payload["progress_decoder"], strict=True)
        else:
            logger.warning("Checkpoint has no 'progress_decoder' weights; keeping random init.")

        if self.visual_encoder is not None:
            if "visual_encoder" in payload:
                self.visual_encoder.load_state_dict(payload["visual_encoder"], strict=True)
                logger.info("Loaded `visual_encoder` weights from checkpoint.")
            else:
                logger.warning(
                    "Checkpoint has no `visual_encoder` weights; keeping freshly-initialized "
                    "`visual_encoder` params (zero-init projector)."
                )
        elif "visual_encoder" in payload:
            logger.warning(
                "Checkpoint contains `visual_encoder` weights but current model has "
                "`visual_encoder=None`; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
