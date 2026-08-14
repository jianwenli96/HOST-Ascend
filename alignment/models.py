# coding=utf-8
"""Model Zoo."""

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from monkey_patch_forward import replace_qwen3_with_mixed_modality_forward
from config import CONFIG

class BaseModel(nn.Module):
    """CNN to extract features from frames."""

    def __init__(self, num_steps):
        super(BaseModel, self).__init__()
        network = CONFIG.MODEL.BASE_MODEL.NETWORK

        if 'Qwen3-VL' in network:
            model_name_or_path = CONFIG.MODEL.BASE_MODEL.MODEL_NAME_OR_PATH
            if not model_name_or_path:
                raise ValueError(
                    "CONFIG.MODEL.BASE_MODEL.MODEL_NAME_OR_PATH must be set "
                    "before constructing the model."
                )
            print(f"Loading {network} from {model_name_or_path}...")
            
            replace_qwen3_with_mixed_modality_forward()
            
            # Load model
            self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu", 
                attn_implementation="sdpa"
            )
            self.base_model.gradient_checkpointing_enable()
            
        else:
            raise ValueError('%s not supported.' % network)

        self.num_steps = num_steps

    def forward(self, x):
        if isinstance(x, dict) and 'qwen_input' in x:
            inputs = x['qwen_input']
            # inputs keys: input_ids, attention_mask, pixel_values, image_grid_thw

            device = self.base_model.device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            forward_kwargs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'pixel_values': inputs['pixel_values'],
                'image_grid_thw': inputs['image_grid_thw'],
                'output_hidden_states': True,
                'return_dict': True
            }
            
            if 'pixel_values_videos' in inputs and inputs['pixel_values_videos'] is not None:
                forward_kwargs['pixel_values_videos'] = inputs['pixel_values_videos']
            
            if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
                forward_kwargs['video_grid_thw'] = inputs['video_grid_thw']

            if 'num_mains' in inputs and 'num_refs' in inputs:
                forward_kwargs['num_video_groups'] = inputs['num_mains'] + inputs['num_refs']
                forward_kwargs['num_mains'] = inputs['num_mains']
                forward_kwargs['num_refs'] = inputs['num_refs']

                # Pass CLS token ID for content_end detection in monkey patch
                if 'cls_token_id' in inputs:
                    forward_kwargs['cls_token_id'] = inputs['cls_token_id']

            outputs = self.base_model(**forward_kwargs)
            
            # Extract features: Average pool image tokens from last hidden state
            # Qwen3VLCausalLMOutputWithPast does not have last_hidden_state attribute, use hidden_states[-1]
            hidden_states = outputs.hidden_states[-1] # (Total_Frames, Seq_Len, Hidden_Size)

            input_ids = inputs['input_ids']

            vision_start_token_id = self.base_model.config.vision_start_token_id
            vision_end_token_id = self.base_model.config.vision_end_token_id

            # Vectorized implementation for efficiency
            is_start = (input_ids == vision_start_token_id)
            is_end = (input_ids == vision_end_token_id)

            # cs_start[i, j] tells us how many start tokens have appeared up to index j
            cs_start = is_start.long().cumsum(dim=1)
            cs_end = is_end.long().cumsum(dim=1)
            
            # --- Extraction Logic ---
            if 'num_mains' in inputs and 'num_refs' in inputs:
                # Packed Mode (Efficient)
                num_mains = inputs['num_mains']
                num_refs = inputs['num_refs']
                group_size = CONFIG.DATA.NUM_STEPS

                cls_token_id = inputs.get('cls_token_id')
                if cls_token_id is not None:
                    # cls_token_id is usually a tensor [ID] or scalar
                    val = cls_token_id[0].item() if cls_token_id.dim() > 0 else cls_token_id.item()
                    is_cls = (input_ids == val)
                else:
                    is_cls = None

                all_mains_embs = []
                all_refs_embs = []

                # Qwen3-VL 4x temporal compression: 4 frames -> 1 token
                # Each group is now a video of 4 frames, which results in vision tokens
                # followed by a <|file_sep|> (CLS token).
                
                for b in range(input_ids.shape[0]):
                    n_m = num_mains[b].item()
                    n_r = num_refs[b].item()
                    hs_b = hidden_states[b]
                    
                    sample_embs = []

                    # CLS extraction (preferred): expect exactly n_m + n_r CLS tokens
                    # (one after each group video).
                    if is_cls is not None:
                        cls_indices = torch.where(is_cls[b])[0]
                        # The first few CLS tokens might belong to the align video if it was processed similarly,
                        # but in AlignmentCollator, only chunk_imgs (groups) have <|file_sep|> after them.
                        # The initial video_imgs (align) does NOT have <|file_sep|> after it in content.

                        for g in range(min(len(cls_indices), n_m + n_r)):
                            idx = cls_indices[g]
                            emb = hs_b[idx]

                            sample_embs.append(emb)

                    if len(sample_embs) < (n_m + n_r):
                        print(f"Warning: Incomplete CLS tokens found for batch {b}. Expected {n_m + n_r}, found {len(sample_embs)}.")
                        # Fill remaining with zeros to avoid shape mismatch
                        for _ in range(len(sample_embs), n_m + n_r):
                            sample_embs.append(torch.zeros(hs_b.shape[-1], device=hs_b.device, dtype=hs_b.dtype))

                    all_mains_embs.extend(sample_embs[:n_m])
                    all_refs_embs.extend(sample_embs[n_m:])

                # Concatenate in order [All_Mains, All_Refs] to match Original Flattening
                final_list = all_mains_embs + all_refs_embs
                if final_list:
                    pooled_embeddings = torch.stack(final_list) # (2BT, H)
                else:
                    pooled_embeddings = torch.empty(0, hidden_states.shape[-1], device=hidden_states.device)

                return pooled_embeddings

            else:
                # Identify the target image (the last one in the sequence)
                # Instead of relying on config, we take the last vision input.
                # cs_start is cumulative sum, so the max value is the total number of vision inputs.
                target_img_idx = cs_start.max(dim=1).values.unsqueeze(1) # (B, 1)
                
                # Mask for the target image:
                # We are inside the target image if we have seen `target_img_idx` starts
                # AND we have seen `target_img_idx - 1` ends.
                # We exclude the start token itself (~is_start).
                mask = (cs_start == target_img_idx) & (cs_end == (target_img_idx - 1)) & (~is_start)

                # Expand mask to match hidden_states dimensions: (B, L, 1)
                mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
                
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1) # (B, H)
                count_tokens = mask_expanded.sum(dim=1) # (B, 1)
                
                # Ensure we found the target frame for every sample
                assert (count_tokens > 0).all(), f"Could not find target image (last image) in some samples. Target indices: {target_img_idx.flatten()}"

                # Avoid division by zero
                count_tokens = torch.clamp(count_tokens, min=1e-9)
                
                pooled_embeddings = sum_embeddings / count_tokens # (Total_Frames, H)

                if self.training:
                    num_steps = CONFIG.TRAIN.NUM_FRAMES
                    total_frames = pooled_embeddings.shape[0]
                    batch_size = total_frames // num_steps

                    if batch_size * num_steps != total_frames:
                        raise ValueError(f"Training shape mismatch: Total frames {total_frames} not divisible by num_steps {num_steps}")
                else:
                    # Evaluation: Use dynamic seq_len, assume Batch=1
                    # Check for seq_lens in inputs (qwen_input) first, then top-level x
                    seq_lens = None
                    if 'seq_lens' in inputs:
                        seq_lens = inputs['seq_lens']
                    elif 'seq_lens' in x:
                        seq_lens = x['seq_lens']
                    
                    if seq_lens is not None:
                        # seq_lens is a Tensor of shape (Batch_Size,)
                        batch_size = len(seq_lens)
                        
                        if batch_size != 1:
                            raise ValueError(f"Evaluation mode requires Batch Size=1 for variable length sequences. Got Batch Size: {batch_size}")
                        
                        # For Batch=1, the single sequence length is the num_steps
                        num_steps = int(seq_lens[0].item())
                        
                        if pooled_embeddings.shape[0] != num_steps:
                            # seq_lens may reflect the original frame count rather than the
                            # extracted count; fall through and use the extracted count below.
                            pass

                        if pooled_embeddings.shape[0] != num_steps:
                             num_steps = pooled_embeddings.shape[0]
                    else:
                        # No seq_len metadata: assume batch_size=1, num_steps=all extracted frames
                        num_steps = pooled_embeddings.shape[0]
                        batch_size = 1

                # Return flattened embeddings as required by train.py's merge logic
                return pooled_embeddings

        raise RuntimeError("BaseModel.forward expects Qwen dict input with a 'qwen_input' key.")

class LinearEmbedder(nn.Module):
    """Simple Linear Embedder for Qwen features."""
    def __init__(self, in_channels):
        super(LinearEmbedder, self).__init__()
        embedding_size = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
        self.fc = nn.Linear(in_channels, embedding_size)

    def forward(self, x, num_frames):
        # Ensure input dtype matches model weights
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # x: (Total_Frames, C)
        x = self.fc(x)

        return x

def get_model():
    """Returns model dict."""
    model = {}
    num_steps = CONFIG.TRAIN.NUM_FRAMES

    cnn = BaseModel(num_steps=num_steps)

    if hasattr(cnn, 'base_model') and hasattr(cnn.base_model, 'config'):
        config = cnn.base_model.config
        if hasattr(config, 'hidden_size'):
            in_channels = config.hidden_size
        elif hasattr(config, 'd_model'):
            in_channels = config.d_model
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            in_channels = config.text_config.hidden_size
        else:
            in_channels = 1536
            print(f"Warning: Could not determine hidden_size from config. Using default {in_channels}.")
    else:
        in_channels = 1536 # Fallback for Qwen2-VL-2B
    emb = LinearEmbedder(in_channels)

    model['cnn'] = cnn
    model['emb'] = emb
    return model
