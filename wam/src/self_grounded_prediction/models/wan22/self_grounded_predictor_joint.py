from typing import Any, Optional

import torch

from self_grounded_prediction.utils.logging_config import get_logger

from .self_grounded_predictor import SelfGroundedPredictor

logger = get_logger(__name__)


class SelfGroundedPredictorJoint(SelfGroundedPredictor):
    """SelfGroundedPredictor variant where action attends to all video latent tokens.

    IMPORTANT — THIS IS THE CLASS USED IN PRODUCTION (via create_self_grounded_predictor in runtime.py).
    All configs with _target_: self_grounded_prediction.runtime.create_self_grounded_predictor instantiate THIS class,
    not the base SelfGroundedPredictor class.

    Key difference from SelfGroundedPredictor (base):
        SelfGroundedPredictor._build_mot_attention_mask:      action → first frame only
        SelfGroundedPredictorJoint._build_mot_attention_mask: action → ALL video tokens (this class)

    Attention mask layout for [task_video | video | action]:
        task_video → task_video:  full
        task_video → video/action: blocked
        video      → task_video:  allowed (condition on task)
        video      → video:       first_frame_causal (first frame cannot see later frames)
        action     → task_video:  blocked  (action conditions on task via cross-attn, not self-attn)
        action     → video:       ALL frames (unlike SelfGroundedPredictor base which only allows first frame)
        action     → action:      full
    """

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for SelfGroundedPredictorJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "SelfGroundedPredictorJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

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
        """Attention mask for [progress | task_video | video | noisy_group | action].

        SelfGroundedPredictorJoint: action attends to ALL video tokens (not just first frame).

        When noisy_group_seq_len > 0, an isolated noisy group is inserted between
        agent video and action. The noisy group self-attends (full) but is completely
        blocked from all other tokens (bidirectional isolation).

        Attention rules:
            progress -> task_video/agent_video: full; -> action/noisy: blocked
            task_video -> progress/task_video: full; -> rest: blocked
            video -> progress/task_video/video: allowed; -> noisy/action: blocked
            noisy_group -> noisy_group: FULL; -> rest: BLOCKED
            rest -> noisy_group: BLOCKED
            action -> progress: FULL; -> task_video: if action_self_attn_to_task_video; -> ALL video: FULL; -> action: full; -> noisy: blocked
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
        # progress -> task_video: FULL
        if p_end > 0 and tv_end > p_end:
            mask[:p_end, p_end:tv_end] = True
        # progress -> agent_video: full
        if p_end > 0:
            mask[:p_end, tv_end:v_end] = True
        # progress -> noisy_group/action: BLOCKED (default False)

        # task_video -> progress: full
        if p_end > 0 and tv_end > p_end:
            mask[p_end:tv_end, :p_end] = True
        # task_video -> task_video: full
        if tv_end > p_end:
            mask[p_end:tv_end, p_end:tv_end] = True
        # task_video -> video/noisy/action: blocked (default False)

        # video -> progress: full
        if p_end > 0:
            mask[tv_end:v_end, :p_end] = True
        # video -> task_video: allowed
        if tv_end > p_end:
            mask[tv_end:v_end, p_end:tv_end] = True
        # video -> video: first_frame_causal
        mask[tv_end:v_end, tv_end:v_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # noisy group -> noisy group: FULL (completely isolated from rest)
        if noisy_group_seq_len > 0:
            mask[v_end:ng_end, v_end:ng_end] = True

        # action -> progress: FULL
        if p_end > 0:
            mask[a_start:, :p_end] = True
        # action -> task_video: allowed when action_self_attn_to_task_video=True
        if self.action_self_attn_to_task_video and tv_end > p_end:
            mask[a_start:, p_end:tv_end] = True
        # action -> full video (SelfGroundedPredictorJoint: attend to ALL video)
        mask[a_start:, tv_end:v_end] = True
        # action -> noisy_group: blocked (default False)
        # action -> action: full
        mask[a_start:, a_start:] = True

        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
        if test_action_with_infer_action:
            logger.warning(
                "`SelfGroundedPredictorJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
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
            test_action_with_infer_action=False,
            task_video=task_video,
            progress_override=progress_override,
            task_video_dropped=task_video_dropped,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
        self.eval()

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
        for _step_i, (step_t_video, step_delta_video, step_t_action, step_delta_action) in enumerate(zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
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
                gt_action=None,
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
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
