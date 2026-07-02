import os
import argparse
import torch
import json
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# Import your model definition to ensure structure matches
# We need to know the exact structure of the saved state_dict to map it back to HF model
# The saved state_dict is likely from 'Algorithm' class, which contains 'model' -> 'base_model' -> 'Qwen'
# Let's inspect the keys in the script.

def convert(args):
    print(f"Loading state dict from {args.checkpoint_path}...")
    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Inspect keys to determine prefix
    # Typical structure in TCC code:
    # model_engine -> Algorithm -> model (BaseModel) -> base_model (Qwen)
    # So keys might look like: "module.model.base_model.model.layers.0..." or "model.base_model.model..."
    
    print("Detecting key prefix...")
    first_key = next(iter(state_dict.keys()))
    print(f"Sample key: {first_key}")
    
    prefix = ""
    if "module.model.base_model." in first_key:
        prefix = "module.model.base_model."
    elif "model.base_model." in first_key:
        prefix = "model.base_model."
    elif "cnn.base_model." in first_key:
        prefix = "cnn.base_model."
    elif "base_model." in first_key:
        prefix = "base_model."
        
    print(f"Removing prefix: '{prefix}'")
    
    # Create new state dict for HF model
    hf_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            hf_state_dict[new_key] = v
            
    # Load empty HF model
    print(f"Loading empty config from {args.model_name_or_path}...")
    try:
        # Try Qwen3VL first
        if 'Qwen3' in args.model_name_or_path and Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        else:
            # Fallback to AutoModel
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
    except Exception as e:
        print(f"Error loading model config: {e}")
        return

    # Load weights
    print("Loading weights into HF model...")
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    
    print(f"Missing keys: {len(missing)}")
    if len(missing) > 0:
        print(f"First 5 missing: {missing[:5]}")
        
    print(f"Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print(f"First 5 unexpected: {unexpected[:5]}")

    # Save
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving HF model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    # Save processor
    try:
        print("Saving processor...")
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Could not save processor: {e}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the converted pytorch_model.bin")
    parser.add_argument("--model_name_or_path", type=str, default="/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B", help="Path to original HF model for config")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HF model")
    args = parser.parse_args()
    
    convert(args)
