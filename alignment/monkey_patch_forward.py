from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModelOutputWithPast
import torch
from typing import Optional, List, Union, Tuple
import transformers.models.qwen2_vl.modeling_qwen2_vl
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling
from config import CONFIG

def replace_qwen_2_with_mixed_modality_forward():
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = qwen2_mixed_modality_forward

def replace_qwen2_5_with_mixed_modality_forward():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_5_mixed_modality_forward

def replace_qwen3_with_mixed_modality_forward():
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.forward = qwen3_vl_mixed_modality_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = qwen3_vl_conditional_generation_forward

def replace_qwen3_vl_moe_with_mixed_modality_forward():
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeModel.forward = qwen3_vl_moe_mixed_modality_forward

def qwen3_vl_moe_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLMoeModelOutputWithPast]:
    
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    
    if pixel_values is None and pixel_values_videos is None:
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(1024, 1536).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 32, 32]]).to(self.visual.device)

        image_embeds, dummy_deepstack = self.get_image_features(dummy_pixel, dummy_grid)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if visual_pos_masks is None:
        B, S, H = inputs_embeds.shape
        visual_pos_masks = torch.zeros((B, S), dtype=torch.bool, device=inputs_embeds.device)
        L = len(self.visual.deepstack_visual_indexes)
        deepstack_visual_embeds = [t.narrow(0, 0, 0) for t in dummy_deepstack]

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                # Use a robust threshold for floating point masks (usually -3e38 or -inf)
                # This avoids division by finfo.min which can cause Inf/NaN artifacts in BF16/FP16
                attention_mask_tensor = (attention_mask_tensor > -1e4).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLMoeModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def _apply_group_attention_mask(attention_mask, input_ids, inputs_embeds, group_starts, batch_idx=0):
    # Function assumes attention_mask is already 4D: (B, 1, S, S)
    # Modifies attention_mask in-place for sample batch_idx

    min_dtype = torch.finfo(inputs_embeds.dtype).min
    start_0 = group_starts[0]
    num_groups = len(group_starts) - 1

    # Prevent prefix from attending to any chunk (prefix should be context-only)
    content_end = group_starts[-1]
    attention_mask[batch_idx, :, 0:start_0, start_0:content_end] = min_dtype

    for k in range(num_groups):
        start_k = group_starts[k]
        end_k = group_starts[k+1]

        # Mask: Prevent Group k from attending to all other groups (full isolate).
        # Block attending to groups before k (start_0:start_k)
        if start_k > start_0:
            attention_mask[batch_idx, :, start_k:end_k, start_0:start_k] = min_dtype
        # Block attending to groups after k (end_k:content_end)
        if end_k < content_end:
            attention_mask[batch_idx, :, start_k:end_k, end_k:content_end] = min_dtype

    return attention_mask

def _apply_group_position_ids(position_ids, group_starts, batch_idx=0):
    # Function assumes position_ids is (3, B, S) and already cloned
    # Modifies position_ids in-place for sample batch_idx
    
    start_0 = group_starts[0]
    # Anchor point: The start of the FIRST group. 
    # All subsequent groups (k > 0) should be shifted so their start aligns with this anchor.
    anchor_pos = position_ids[:, batch_idx, start_0] # Shape: (3,)

    num_groups = len(group_starts) - 1
    
    for k in range(1, num_groups):
        start_k = group_starts[k]
        end_k = group_starts[k+1]
        
        current_pos_start = position_ids[:, batch_idx, start_k] # Shape: (3,)

        # We want: new_pos_start == anchor_pos
        # So: current_pos_start - shift = anchor_pos
        # shift = current_pos_start - anchor_pos
        shift = current_pos_start - anchor_pos # Shape: (3,)

        # position_ids slice shape: (3, len_k)
        # shift aligned for broadcasting: (3, 1)
        shift_broadcast = shift.unsqueeze(-1)
        
        # In-place subtraction
        # We use standard subtraction because position_ids are LongTensor (indices)
        position_ids[:, batch_idx, start_k:end_k] = position_ids[:, batch_idx, start_k:end_k] - shift_broadcast

    return position_ids

def qwen3_vl_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is None and pixel_values_videos is None:
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(1024, 1536).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 32, 32]]).to(self.visual.device)

        image_embeds, dummy_deepstack = self.get_image_features(dummy_pixel, dummy_grid)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if visual_pos_masks is None:
        B, S, H = inputs_embeds.shape
        visual_pos_masks = torch.zeros((B, S), dtype=torch.bool, device=inputs_embeds.device)
        L = len(self.visual.deepstack_visual_indexes)
        deepstack_visual_embeds = [t.narrow(0, 0, 0) for t in dummy_deepstack]

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                # Use a robust threshold for floating point masks (usually -3e38 or -inf)
                # This avoids division by finfo.min which can cause Inf/NaN artifacts in BF16/FP16
                attention_mask_tensor = (attention_mask_tensor > -1e4).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # ----------------------------------------------------------------------
    # Custom Packed Sequence Logic: Token-Based Group Boundary Detection
    # ----------------------------------------------------------------------
    # Use special tokens to mark boundaries:
    # - <|fim_pad|> (CONFIG.SPECIAL_TOKENS.ALIGN_END_TOKEN_ID): marks the end of align video
    # - <|file_sep|> (CONFIG.SPECIAL_TOKENS.CLS_TOKEN_ID): marks the end of each group video
    # This approach is independent of model's temporal compression ratio.

    if attention_mask is not None:
        if attention_mask.ndim == 2:
            bsz, seq_len = input_ids.shape
            min_dtype = torch.finfo(inputs_embeds.dtype).min

            # Full intra-group attention: start with all-zero mask so every token
            # can attend to every other token.  _apply_group_attention_mask will
            # still block cross-chunk backward attention (group k cannot see group
            # k-1 and earlier).  The prefix (align video / instruction) is also
            # full attention under this mode.
            mask = torch.zeros(
                bsz, 1, seq_len, seq_len,
                device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            padding_mask = attention_mask.to(torch.bool)
            mask.masked_fill_(~padding_mask[:, None, None, :], min_dtype)

            attention_mask = mask
        else:
            # If it was already 4D, ensure we clone it so we don't modify shared views or inputs directly
            attention_mask = attention_mask.clone()

    if position_ids is not None:
        position_ids = position_ids.clone()

    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

    if input_ids is not None:
        for b in range(batch_size):
            align_end_token_id = kwargs.get('align_end_token_id')
            if align_end_token_id is not None:
                if hasattr(align_end_token_id, 'item'):
                    align_end_token_id = align_end_token_id.item()
            else:
                align_end_token_id = CONFIG.SPECIAL_TOKENS.ALIGN_END_TOKEN_ID

            cls_id = kwargs.get('cls_token_id')
            if cls_id is not None:
                if hasattr(cls_id, 'item'):
                    cls_id = cls_id.item()
            else:
                cls_id = CONFIG.SPECIAL_TOKENS.CLS_TOKEN_ID

            if cls_id is None or align_end_token_id is None:
                print(f"Warning: No special tokens found for batch {b}. Skipping.")
                continue

            align_end_indices = torch.where(input_ids[b] == align_end_token_id)[0]
            if len(align_end_indices) != 1:
                raise ValueError(
                    f"Batch {b}: Expected exactly 1 align_end token (<|fim_pad|>, ID={align_end_token_id}), "
                    f"found {len(align_end_indices)}"
                )
            align_video_end = align_end_indices[0].item() + 1

            cls_indices = torch.where(input_ids[b] == cls_id)[0]

            if 'num_mains' in kwargs and 'num_refs' in kwargs:
                expected_num_cls = kwargs['num_mains'][b].item() + kwargs['num_refs'][b].item()
                if len(cls_indices) != expected_num_cls:
                    print(f"Warning: Batch {b}: Expected {expected_num_cls} CLS tokens (<|file_sep|>, ID={cls_id}), found {len(cls_indices)}")

            # group_starts[k] is the start position of group k; group_starts[-1] is the content_end
            group_starts = [align_video_end]  # First group starts after align video
            for cls_pos in cls_indices[:-1]:
                group_starts.append(cls_pos.item() + 1)  # Next group starts after CLS
            content_end = cls_indices[-1].item() + 1
            group_starts.append(content_end)

            if attention_mask is not None:
                attention_mask = _apply_group_attention_mask(
                    attention_mask, input_ids, inputs_embeds, group_starts,
                    batch_idx=b
                )
            
            if position_ids is not None:
                position_ids = _apply_group_position_ids(position_ids, group_starts, batch_idx=b)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def qwen2_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)

        image_embeds = self.get_image_features(dummy_pixel, dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)  # (sum_tokens, hidden)
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def qwen2_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.visual.get_device())
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.get_device())

        image_embeds = self.get_image_features(dummy_pixel, dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)  # (sum_tokens, hidden)
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids += delta.to(position_ids.device)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()

def qwen3_vl_conditional_generation_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    return_hidden_only: bool = False,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    return_hidden_only (`bool`, *optional*, defaults to `False`):
        Return the multimodal backbone output before the language-model head.
        Used by alignment, which consumes representations instead of token logits.

    Example:
        TODO: Add example
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    # Alignment consumes the backbone's last hidden state directly and has no
    # language-modeling loss.  Returning here avoids materializing the very
    # large [batch, sequence, vocab] logits tensor while keeping the default
    # conditional-generation path unchanged for callers such as generate().
    if return_hidden_only:
        return outputs

    hidden_states = outputs[0]

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
