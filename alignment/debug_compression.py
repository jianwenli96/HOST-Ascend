import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import numpy as np
import os

def verify_compression():
    model_path = '/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading processor and model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    # We only need the config and vision tower for this check, but loading the whole model is safer for consistency
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    model.eval()

    # Create a dummy video: 8 frames of 224x224
    num_frames = 8
    dummy_video = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(num_frames)]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": dummy_video, "fps": 1.0},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process with different frame counts to see the pattern
    # Qwen3-VL-2B might have a very small default resolution if not specified
    for n in [4, 8, 12, 16]:
        test_video = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(n)]
        
        # Try to use the helper from datasets.py logic
        from data_utils import get_video_info
        video_input_pair, video_kwargs = get_video_info(
            test_video,
            min_pixels=224 * 224,
            max_pixels=256 * 256,
            width=None,
            height=None,
            fps=2.0,
            image_patch_size=16,
            return_video_metadata=True
        )
        video_tensor, video_metadata = video_input_pair
        
        inputs = processor(
            text=[text],
            videos=[video_tensor],
            padding=True,
            return_tensors="pt",
            video_metadata=[video_metadata],
            **video_kwargs
        )
        
        # Manually check the config or processor attributes
        print(f"Processor video_fps: {getattr(processor, 'video_fps', 'Not set')}")
        print(f"Processor image_processor patch_size: {getattr(processor.image_processor, 'patch_size', 'Not set')}")
        print(f"Processor image_processor temporal_patch_size: {getattr(processor.image_processor, 'temporal_patch_size', 'Not set')}")
        
        input_ids = inputs['input_ids']
        # Vision token ID for Qwen2/3-VL is usually 151652
        vision_token_id = 151652 
        num_vision_tokens = (input_ids == vision_token_id).sum().item()
        
        print(f"\nVideo Frames: {n}")
        print(f"Total Vision Tokens in input_ids: {num_vision_tokens}")
        
        # Let's see if we can deduce the spatial tokens from the grid_thw
        if 'video_grid_thw' in inputs:
            grid = inputs['video_grid_thw'][0] # [T, H, W]
            print(f"Video Grid THW: {grid.tolist()}")
            t_tokens, h_tokens, w_tokens = grid.tolist()
            spatial_tokens_per_frame = h_tokens * w_tokens
            print(f"Spatial Tokens per temporal step: {spatial_tokens_per_frame}")
            print(f"Temporal Tokens (T): {t_tokens}")
            compression = n / t_tokens
            print(f"Calculated Temporal Compression Ratio: {compression:.1f}")

if __name__ == "__main__":
    with torch.no_grad():
        verify_compression()
