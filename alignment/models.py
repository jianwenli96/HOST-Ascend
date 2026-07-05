# coding=utf-8
"""Model Zoo."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models
import os
from transformers import AutoModelForCausalLM, AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from monkey_patch_forward import replace_qwen3_with_mixed_modality_forward
from config import CONFIG
from utils import check_nan

class BaseModel(nn.Module):
    """CNN to extract features from frames."""

    def __init__(self, num_steps):
        super(BaseModel, self).__init__()
        network = CONFIG.MODEL.BASE_MODEL.NETWORK
        layer = CONFIG.MODEL.BASE_MODEL.LAYER
        
        if network in ['Resnet50', 'Resnet50_pretrained']:
            print('Loading ResNet50 model...')
            # TODO(open-source): internal-cluster path; not load-bearing for the
            # verified training run (NETWORK=Qwen3-VL-2B doesn't reach this branch),
            # and already falls back to a public download below. Low priority.
            # See OPEN_SOURCE_PATH_TODOS.md.
            # Load pretrained ResNet50
            resnet = models.resnet50(pretrained=False)
            resnet_ckpt_path = '/x2robot_v2/ethanchen/open_ckpts/resnet/resnet50-0676ba61.pth'
            if not os.path.exists(resnet_ckpt_path):
                resnet_ckpt_path = '/x2robot_v2/ethanchen/open_ckpts/resnet/resnet50.pth'
            
            if os.path.exists(resnet_ckpt_path):
                print(f"Loading pretrained ResNet50 from {resnet_ckpt_path}")
                resnet.load_state_dict(torch.load(resnet_ckpt_path))
            else:
                print(f"Warning: Could not find ResNet50 weights at {resnet_ckpt_path}. Downloading...")
                resnet = models.resnet50(pretrained=True)

            # We need to extract features from a specific layer.
            # 'conv4_block3_out' in TF ResNet50V2 roughly corresponds to layer3 in PyTorch ResNet.
            # But let's check the architecture.
            # layer3 output is 1024 channels, 14x14 (for 224x224 input).
            # layer4 output is 2048 channels, 7x7.
            
            # In TF code: base_model.get_layer(layer).output
            # If layer is 'conv4_block3_out', it is likely the output of the 3rd block of the 4th stage (which is layer3 in PyTorch).
            
            self.base_model = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3
            )
            # If we need layer4, we would add resnet.layer4
            
        elif network == 'dinov2_vitb14':
            print("Loading DINOv2 Base model...")
            # TODO(open-source): internal-cluster paths below; not load-bearing for
            # the verified training run (NETWORK=Qwen3-VL-2B doesn't reach this
            # branch), and already falls back to a public download further down.
            # Low priority. See OPEN_SOURCE_PATH_TODOS.md.

            # Use local source for DINOv2 to avoid network issues
            repo_dir = '/x2robot_v2/ethanchen/code/dinov2_source'
            if not os.path.exists(repo_dir):
                 raise FileNotFoundError(f"DINOv2 source not found at {repo_dir}. Please clone it first.")
            
            print(f"Loading DINOv2 from local source: {repo_dir}")
            self.base_model = torch.hub.load(repo_dir, 'dinov2_vitb14', source='local', pretrained=False)
            
            # Local paths
            local_dir = '/x2robot_v2/ethanchen/open_ckpts/dinov2_vitb14'
            ckpt_filename = 'dinov2_vitb14_pretrain.pth'
            ckpt_path = os.path.join(local_dir, ckpt_filename)
            
            # Ensure directory exists
            if not os.path.exists(local_dir):
                print(f"Creating directory {local_dir}")
                os.makedirs(local_dir, exist_ok=True)
                
            # Check if weights exist, if not download to target path
            if not os.path.exists(ckpt_path):
                # Check for alternative name just in case
                alt_path = os.path.join(local_dir, 'dinov2_vitb14.pth')
                if os.path.exists(alt_path):
                    ckpt_path = alt_path
                else:
                    print(f"Weights not found at {ckpt_path}. Downloading...")
                    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
                    try:
                        torch.hub.download_url_to_file(url, ckpt_path)
                    except Exception as e:
                        print(f"Error downloading to {ckpt_path}: {e}")
                        print("Falling back to default torch hub cache...")
                        self.base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
                        return

            print(f"Loading DINOv2 weights from {ckpt_path}")
            self.base_model.load_state_dict(torch.load(ckpt_path))

        elif network == 'dinov2_vitl14':
            print("Loading DINOv2 Large model...")
            # TODO(open-source): internal-cluster paths below; not load-bearing for
            # the verified training run (NETWORK=Qwen3-VL-2B doesn't reach this
            # branch), and already falls back to a public download further down.
            # Low priority. See OPEN_SOURCE_PATH_TODOS.md.

            # Use local source for DINOv2 to avoid network issues
            repo_dir = '/x2robot_v2/ethanchen/code/dinov2_source'
            if not os.path.exists(repo_dir):
                 raise FileNotFoundError(f"DINOv2 source not found at {repo_dir}. Please clone it first.")
            
            print(f"Loading DINOv2 from local source: {repo_dir}")
            self.base_model = torch.hub.load(repo_dir, 'dinov2_vitl14', source='local', pretrained=False)
            
            # Local paths
            local_dir = '/x2robot_v2/ethanchen/open_ckpts/dinov2_vitl14'
            ckpt_filename = 'dinov2_vitl14_pretrain.pth'
            ckpt_path = os.path.join(local_dir, ckpt_filename)
            
            # Ensure directory exists
            if not os.path.exists(local_dir):
                print(f"Creating directory {local_dir}")
                os.makedirs(local_dir, exist_ok=True)
                
            # Check if weights exist, if not download to target path
            if not os.path.exists(ckpt_path):
                # Check for alternative name just in case
                alt_path = os.path.join(local_dir, 'dinov2_vitl14.pth')
                if os.path.exists(alt_path):
                    ckpt_path = alt_path
                else:
                    print(f"Weights not found at {ckpt_path}. Downloading...")
                    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
                    try:
                        torch.hub.download_url_to_file(url, ckpt_path)
                    except Exception as e:
                        print(f"Error downloading to {ckpt_path}: {e}")
                        print("Falling back to default torch hub cache...")
                        self.base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
                        return

            print(f"Loading DINOv2 weights from {ckpt_path}")
            self.base_model.load_state_dict(torch.load(ckpt_path))

        elif 'Qwen3-VL' in network:
            print(f"Loading {network}...")
            # TODO(open-source): internal-cluster path, load-bearing for the
            # verified training run — do not remove without re-verifying end-to-end.
            # Planned fix: env var override defaulting to this path, falling back to
            # the public HF repo https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
            # if neither is present. See OPEN_SOURCE_PATH_TODOS.md.
            model_path = '/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            replace_qwen3_with_mixed_modality_forward()
            
            # Load model
            self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu", 
                attn_implementation="sdpa"
            )
            self.base_model.gradient_checkpointing_enable()
            
        else:
            raise ValueError('%s not supported.' % network)

        self.num_steps = num_steps

    def forward(self, x):
        # Handle Qwen dictionary input
        if isinstance(x, dict) and 'qwen_input' in x:
            inputs = x['qwen_input']
            # inputs keys: input_ids, attention_mask, pixel_values, image_grid_thw
            
            # Run model
            # Ensure inputs are on the correct device
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

            # Step 4: Inject Group Info for Attention Mask
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
            
            if CONFIG.DEBUG:
                check_nan(hidden_states, 'qwen_hidden_states', 'BaseModel.forward')

            input_ids = inputs['input_ids']
            
            vision_start_token_id = self.base_model.config.vision_start_token_id
            vision_end_token_id = self.base_model.config.vision_end_token_id
            
            # Vectorized implementation for efficiency
            # Identify positions of start and end tokens
            is_start = (input_ids == vision_start_token_id)
            is_end = (input_ids == vision_end_token_id)
            
            # Cumulative sums to identify regions
            # cs_start[i, j] tells us how many start tokens have appeared up to index j
            cs_start = is_start.long().cumsum(dim=1)
            cs_end = is_end.long().cumsum(dim=1)
            
            # --- Extraction Logic ---
            if 'num_mains' in inputs and 'num_refs' in inputs:
                # Packed Mode (Efficient)
                num_mains = inputs['num_mains']
                num_refs = inputs['num_refs']
                group_size = CONFIG.DATA.NUM_STEPS
                
                # Identify CLS tokens if available
                cls_token_id = inputs.get('cls_token_id')
                if cls_token_id is not None:
                    # cls_token_id is usually a tensor [ID] or scalar
                    val = cls_token_id[0].item() if cls_token_id.dim() > 0 else cls_token_id.item()
                    is_cls = (input_ids == val)
                else:
                    is_cls = None

                all_mains_embs = []
                all_refs_embs = []
                all_mains_tokens = []
                all_refs_tokens = []
                
                # Qwen3-VL 4x temporal compression: 4 frames -> 1 token
                # Each group is now a video of 4 frames, which results in vision tokens
                # followed by a <|file_sep|> (CLS token).
                
                for b in range(input_ids.shape[0]):
                    n_m = num_mains[b].item()
                    n_r = num_refs[b].item()
                    hs_b = hidden_states[b]
                    
                    sample_embs = []
                    sample_tokens = []

                    # 1. Try CLS Extraction (Preferred)
                    # We expect exactly n_m + n_r CLS tokens (one after each group video)
                    if is_cls is not None:
                        cls_indices = torch.where(is_cls[b])[0]
                        # The first few CLS tokens might belong to the align video if it was processed similarly,
                        # but in AlignmentCollator, only chunk_imgs (groups) have <|file_sep|> after them.
                        # The initial video_imgs (align) does NOT have <|file_sep|> after it in content.
                        
                        for g in range(min(len(cls_indices), n_m + n_r)):
                            idx = cls_indices[g]
                            emb = hs_b[idx]
                            
                            if CONFIG.DEBUG:
                                check_nan(emb, f'cls_emb_b{b}_g{g}', 'BaseModel.forward CLS extraction')

                            sample_embs.append(emb)
                            # For AttentionGate, use single CLS token
                            sample_tokens.append(hs_b[idx:idx+1]) # (1, D)
                    
                    # 2. Warning if CLS not found or incomplete
                    if len(sample_embs) < (n_m + n_r):
                        print(f"Warning: Incomplete CLS tokens found for batch {b}. Expected {n_m + n_r}, found {len(sample_embs)}.")
                        # Fill remaining with zeros to avoid shape mismatch
                        for _ in range(len(sample_embs), n_m + n_r):
                            sample_embs.append(torch.zeros(hs_b.shape[-1], device=hs_b.device, dtype=hs_b.dtype))
                            sample_tokens.append(torch.zeros((1, hs_b.shape[-1]), device=hs_b.device, dtype=hs_b.dtype))

                    # Split
                    all_mains_embs.extend(sample_embs[:n_m])
                    all_refs_embs.extend(sample_embs[n_m:])
                    all_mains_tokens.extend(sample_tokens[:n_m])
                    all_refs_tokens.extend(sample_tokens[n_m:])
                    
                # Concatenate in order [All_Mains, All_Refs] to match Original Flattening
                final_list = all_mains_embs + all_refs_embs
                if final_list:
                    pooled_embeddings = torch.stack(final_list) # (2BT, H)
                else:
                    pooled_embeddings = torch.empty(0, hidden_states.shape[-1], device=hidden_states.device)
                
                if CONFIG.ALIGNMENT.USE_ATTN_GATE:
                    # Return both embeddings and raw tokens for gating
                    # We need to reshape tokens safely. Since we use IMAGE_SIZE=224, 
                    # they should have same P. 
                    # We check and pad if necessary.
                    max_p = max([t.size(0) for t in all_mains_tokens + all_refs_tokens])
                    def pad_tokens(t_list, p_len):
                        padded = []
                        for t in t_list:
                            if t.size(0) < p_len:
                                # Pad with zeros
                                padding = torch.zeros((p_len - t.size(0)), t.size(1), device=t.device, dtype=t.dtype)
                                padded.append(torch.cat([t, padding], dim=0))
                            else:
                                padded.append(t[:p_len])
                        return torch.stack(padded)
                    
                    mains_tokens_tensor = pad_tokens(all_mains_tokens, max_p)
                    refs_tokens_tensor = pad_tokens(all_refs_tokens, max_p)
                    
                    batch_size = input_ids.shape[0]
                    # Reshape to (B, T, P, D)
                    # Note: all_mains_tokens was extend, so it's a list. 
                    # The T might vary per sample in dataset, but for now we assume uniform for tensor construction
                    # or we return a list. TCC logic handles list better? No, usually prefers tensors.
                    
                    return {
                        'embeddings': pooled_embeddings,
                        'mains_tokens': mains_tokens_tensor,
                        'refs_tokens': refs_tokens_tensor
                    }
                    
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
                
                # Compute pooled embeddings
                # Expand mask to match hidden_states dimensions: (B, L, 1)
                mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
                
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1) # (B, H)
                count_tokens = mask_expanded.sum(dim=1) # (B, 1)
                
                # Ensure we found the target frame for every sample
                assert (count_tokens > 0).all(), f"Could not find target image (last image) in some samples. Target indices: {target_img_idx.flatten()}"

                # Avoid division by zero
                count_tokens = torch.clamp(count_tokens, min=1e-9)
                
                pooled_embeddings = sum_embeddings / count_tokens # (Total_Frames, H)
                
                # Reshape to (B, T, H, 1, 1)
                if self.training:
                    # Training: Use fixed config
                    num_steps = CONFIG.TRAIN.NUM_FRAMES
                    total_frames = pooled_embeddings.shape[0]
                    batch_size = total_frames // num_steps
                    
                    # Sanity check for training
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
                        
                        # Verify consistency with actual output
                        if pooled_embeddings.shape[0] != num_steps:
                            # It's possible seq_lens was original frames, but we extracted fewer?
                            # For now just trust the extracted count if mismatch, or use extracted count.
                            # But TCC eval usually needs reshaping.
                            pass
                            
                        # Overwrite num_steps to match actual output if needed, or use seq_lens
                        if pooled_embeddings.shape[0] != num_steps:
                             # print(f"Warning: Output embeddings count {pooled_embeddings.shape[0]} != seq_len {num_steps}. Using output count.")
                             num_steps = pooled_embeddings.shape[0]
                    else:
                        # Fallback if no metadata
                        # Assume Num_Frames from Config or just Treat as T=Total, B=1
                        num_steps = pooled_embeddings.shape[0]
                        batch_size = 1

                # return pooled_embeddings.view(batch_size, num_steps, -1, 1, 1)
                # Return flattened embeddings as requested by train.py logic
                return pooled_embeddings

        # Standard ResNet/DINO path
        # x: (batch_size, num_steps, C, H, W)
        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size * num_steps, c, h, w)
        
        network = CONFIG.MODEL.BASE_MODEL.NETWORK
        if network in ['dinov2_vitb14', 'dinov2_vitl14']:
            # DINOv2 forward
            # forward_features returns dict with 'x_norm_patchtokens'
            out = self.base_model.forward_features(x)
            patch_tokens = out['x_norm_patchtokens'] # (B*T, N, D)
            
            # Reshape to (B*T, D, H_grid, W_grid)
            # Assuming square input and patch size 14
            # N = (H/14) * (W/14)
            bt, n, d = patch_tokens.shape
            h_grid = int(n**0.5)
            w_grid = h_grid
            
            x = patch_tokens.permute(0, 2, 1).reshape(bt, d, h_grid, w_grid)
        else:
            x = self.base_model(x)
        
        _, c, h, w = x.shape
        x = x.view(batch_size, num_steps, c, h, w)
        return x

class ConvEmbedder(nn.Module):
    """Embedder network."""

    def __init__(self):
        super(ConvEmbedder, self).__init__()
        
        conv_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS
        fc_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS
        use_bn = CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN
        embedding_size = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
        cap_scalar = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR
        
        # Input channels? It comes from BaseModel.
        # ResNet50 layer3 output has 1024 channels.
        network = CONFIG.MODEL.BASE_MODEL.NETWORK
        if network == 'dinov2_vitb14':
            in_channels = 768
        elif network == 'dinov2_vitl14':
            in_channels = 1024
        else:
            in_channels = 1024 
        
        self.conv_layers = nn.ModuleList()
        
        for channels, kernel_size, activate in conv_params:
            out_channels = int(cap_scalar * channels)
            # TF uses 'same' padding. For kernel_size 3, padding is 1.
            padding = kernel_size // 2
            
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_bn)
            layers_list = [conv]
            if use_bn:
                layers_list.append(nn.BatchNorm3d(out_channels))
            if activate:
                layers_list.append(nn.ReLU(inplace=True))
            
            self.conv_layers.append(nn.Sequential(*layers_list))
            in_channels = out_channels
            
        self.flatten_method = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD
        
        self.fc_layers = nn.ModuleList()
        for channels, activate in fc_params:
            out_channels = int(cap_scalar * channels)
            layers_list = []
            layers_list.append(nn.Linear(in_channels, out_channels))
            if activate:
                layers_list.append(nn.ReLU(inplace=True))
            self.fc_layers.append(nn.Sequential(*layers_list))
            in_channels = out_channels
            
        self.embedding_layer = nn.Linear(in_channels, embedding_size)
        
        self.base_dropout_rate = CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE
        self.fc_dropout_rate = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE
        self.l2_normalize = CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE
        self.base_dropout_spatial = CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL

    def forward(self, x, num_frames):
        # Ensure input dtype matches model weights
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # x: (batch_size, total_num_steps, C, H, W)
        batch_size, total_num_steps, c, h, w = x.shape
        num_context = total_num_steps // num_frames
        
        # Reshape for 3D Conv: (batch_size * num_frames, C, num_context, H, W)
        x = x.view(batch_size * num_frames, num_context, c, h, w)
        x = x.permute(0, 2, 1, 3, 4) # (B*N, C, T, H, W)
        
        # Dropout
        if self.base_dropout_rate > 0:
             # SpatialDropout3D in TF drops entire feature maps.
             # nn.Dropout3d in PyTorch does the same (zeroes out entire channels).
             if self.base_dropout_spatial:
                 x = F.dropout3d(x, p=self.base_dropout_rate, training=self.training)
             else:
                 x = F.dropout(x, p=self.base_dropout_rate, training=self.training)

        for layer in self.conv_layers:
            x = layer(x)
            
        # Spatial pooling
        if self.flatten_method == 'max_pool':
            # Global Max Pooling over (T, H, W)
            x = F.adaptive_max_pool3d(x, (1, 1, 1))
        elif self.flatten_method == 'avg_pool':
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        elif self.flatten_method == 'flatten':
            pass # Flatten handled below
        else:
             raise ValueError('Supported flatten methods: max_pool, avg_pool and flatten.')
             
        x = x.view(x.size(0), -1)
        
        for layer in self.fc_layers:
            if self.fc_dropout_rate > 0:
                x = F.dropout(x, p=self.fc_dropout_rate, training=self.training)
            x = layer(x)
            
        x = self.embedding_layer(x)
        
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)
            
        return x

class ConvGRUEmbedder(nn.Module):
    """Embedder network which uses ConvGRU."""
    # Simplified implementation using standard GRU after Conv2D
    
    def __init__(self):
        super(ConvGRUEmbedder, self).__init__()
        if CONFIG.DATA.NUM_STEPS != 1:
             raise ValueError('Cannot use GRU with context frames.')
             
        conv_params = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.CONV_LAYERS
        use_bn = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.USE_BN
        gru_params = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.GRU_LAYERS
        
        in_channels = 1024 # From ResNet layer3
        
        self.conv_layers = nn.ModuleList()
        for channels, kernel_size, activate in conv_params:
            padding = kernel_size // 2
            layers_list = []
            layers_list.append(nn.Conv2d(in_channels, channels, kernel_size, padding=padding, bias=not use_bn))
            if use_bn:
                layers_list.append(nn.BatchNorm2d(channels))
            if activate:
                layers_list.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.Sequential(*layers_list))
            in_channels = channels
            
        self.gru_layers = nn.ModuleList()
        for units in gru_params:
            self.gru_layers.append(nn.GRU(input_size=in_channels, hidden_size=units, batch_first=True))
            in_channels = units
            
        self.dropout_rate = CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.DROPOUT_RATE

    def forward(self, x, num_frames):
        # x: (batch_size, num_steps, C, H, W)
        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size * num_steps, c, h, w)
        
        for layer in self.conv_layers:
            if self.dropout_rate > 0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = layer(x)
            
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(batch_size, num_steps, -1)
        
        for layer in self.gru_layers:
            x, _ = layer(x)
            
        # x is (batch_size, num_steps, hidden_size)
        # We need to flatten to (batch_size * num_steps, hidden_size) to match ConvEmbedder output style if needed
        # But wait, ConvEmbedder returns (batch_size * num_frames, embedding_size)
        
        x = x.contiguous().view(batch_size * num_steps, -1)
        return x

class LinearEmbedder(nn.Module):
    """Simple Linear Embedder for Qwen features."""
    def __init__(self, in_channels):
        super(LinearEmbedder, self).__init__()
        embedding_size = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
        self.fc = nn.Linear(in_channels, embedding_size)
        self.l2_normalize = CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE

    def forward(self, x, num_frames):
        # Handle dictionary input (e.g. from Attention Gate)
        extra_info = {}
        if isinstance(x, dict):
            extra_info = x
            x = x['embeddings']
            
        # Ensure input dtype matches model weights
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Handle both Qwen (2D: Total_Frames x C) and ResNet (5D: B x T x C x H x W) inputs
        if x.dim() == 5:
            # x: (B, T, C, 1, 1)
            b, t, c, h, w = x.shape
            x = x.view(b * t, c)
        
        # If already 2D, straight to FC (x is [Batch*Steps, C])
        
        x = self.fc(x)
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)
            
        if extra_info:
            # Update embeddings in dictionary and return
            extra_info['embeddings'] = x
            # Rename key for compatibility if needed, but 'embeddings' is standard
            return extra_info
            
        return x

class AttentionGate(nn.Module):
    def __init__(self, in_channels, d_model=256, nhead=8, num_layers=4, dim_feedforward=512):
        super(AttentionGate, self).__init__()
        # Project high-dim Qwen tokens (e.g. 1536) to a smaller d_model (256) for efficiency
        self.compress = nn.Linear(in_channels, d_model)
        
        # Use standard Transformer Decoder for cross-attention
        # tgt = main video patches, memory = reference video patches
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection to distance/weight scalar
        self.gate_head = nn.Linear(d_model, 1)

    def forward(self, tokens_m, tokens_r):
        """
        Gives a gate for every pair of frames between main and ref.
        tokens_m: (B, T_m, P, in_channels)
        tokens_r: (B, T_r, P, in_channels)
        Returns: (B, T_m, T_r, 1)
        """
        # Compress tokens to d_model first
        tokens_m = self.compress(tokens_m)
        tokens_r = self.compress(tokens_r)

        B, Tm, P, D = tokens_m.shape
        Tr = tokens_r.shape[1]
        
        # Prepare inputs: Cross-attention computed for all Tm * Tr pairs.
        # We process all transition pairs in a single large batch dimension.
        tgt = tokens_m.unsqueeze(2).expand(-1, -1, Tr, -1, -1).reshape(B * Tm * Tr, P, D)
        memory = tokens_r.unsqueeze(1).expand(-1, Tm, -1, -1, -1).reshape(B * Tm * Tr, P, D)
        
        # Multi-layer Transformer (Cross-Attention + Norms + Add + FFN)
        output = self.decoder(tgt, memory) # (B * Tm * Tr, P, D)
        
        # Pool patches (mean) to get a feature vector per frame-pair
        pair_feats = output.mean(dim=1) # (B * Tm * Tr, D)
        
        # Linear + Sigmoid to get scalar gate in range [0, 1]
        gates = torch.sigmoid(self.gate_head(pair_feats)) # (B * Tm * Tr, 1)
        
        return gates.view(B, Tm, Tr, 1)

def get_model():
    """Returns model dict."""
    model = {}
    num_steps = CONFIG.TRAIN.NUM_FRAMES

    cnn = BaseModel(num_steps=num_steps)

    if 'Qwen' in CONFIG.MODEL.BASE_MODEL.NETWORK:
        # Use LinearEmbedder for Qwen
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
        
        # Add Transformer Gate if enabled
        if CONFIG.ALIGNMENT.USE_ATTN_GATE:
            gate_d_model = CONFIG.ALIGNMENT.GATE_HIDDEN_DIM # E.g. 256
            gate_heads = CONFIG.ALIGNMENT.GATE_HEADS
            gate_layers = CONFIG.ALIGNMENT.get('GATE_LAYERS', 1)
            model['gate'] = AttentionGate(
                in_channels, 
                d_model=gate_d_model,
                nhead=gate_heads, 
                num_layers=gate_layers, 
                dim_feedforward=gate_d_model * 2
            )
            
    elif CONFIG.MODEL.EMBEDDER_TYPE == 'conv':
        emb = ConvEmbedder()
    elif CONFIG.MODEL.EMBEDDER_TYPE == 'convgru':
        emb = ConvGRUEmbedder()
    else:
        raise ValueError('%s not supported.' % CONFIG.MODEL.EMBEDDER_TYPE)

    model['cnn'] = cnn
    model['emb'] = emb
    return model
