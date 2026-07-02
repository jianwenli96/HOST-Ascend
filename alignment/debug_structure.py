
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

# Create dummy images
os.makedirs("dummy_data", exist_ok=True)
dummy_frames = []
for i in range(20):
    path = f"dummy_data/frame_{i}.jpg"
    if not os.path.exists(path):
        Image.fromarray(np.zeros((224, 224, 3), dtype='uint8')).save(path)
    dummy_frames.append(path)

# Mock CONFIG
class Config:
    class DATA:
        NUM_STEPS = 2
CONFIG = Config()

def test_tokenization():
    # Load processor (Assuming Qwen2-VL is available via HF or similar API compatible)
    # Since I cannot load local checkpoint, I will try to load a public one or mock the token ids behavior if possible.
    # But Qwen tokenization is specific.
    # I'll try 'Qwen/Qwen2-VL-2B-Instruct' from HF Hub if internet access is allowed/cached.
    # Or just check the code in monkey_patch more deeply.
    
    try:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    except Exception as e:
        print(f"Cannot load processor: {e}")
        return

    # Simulate packing
    align_frames = dummy_frames[:10] # 10 frames for Align
    main_frames = dummy_frames[10:14] # 4 frames -> 2 chunks (Context=2)
    
    # Chunking
    chunks = [main_frames[i:i+CONFIG.DATA.NUM_STEPS] for i in range(0, len(main_frames), CONFIG.DATA.NUM_STEPS)]
    # chunks = [[10, 11], [12, 13]]
    
    content = []
    # 1. Align Video
    content.append({"type": "video", "video": align_frames})
    
    # 2. Chunks
    for chunk in chunks:
        for f in chunk:
            content.append({"type": "image", "image": f})
            
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Generated Text Template:")
    print(text)
    
    # Tokenize
    # Check if vision tokens are present in text
    vision_start_count = text.count("<|vision_start|>")
    vision_end_count = text.count("<|vision_end|>")
    
    print(f"Vision Start Count: {vision_start_count}")
    print(f"Vision End Count: {vision_end_count}")
    
    # Expected structure:
    # 1 Video (Align) + 4 Images (2 chunks * 2 images) = 5 vision blocks
    
    # Input ID check
    # Note: we need pixel_values to actually run processor fully usually, but for text we can use tokenizer
    inputs = processor.tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids
    
    vision_token_id = 151652 # <|vision_start|>
    
    indices = torch.where(input_ids == vision_token_id)[1]
    print(f"Vision Start Indices: {indices}")
    print(f"Total starts: {len(indices)}")
    
    if len(indices) == 5:
        print("Structure: 1 for Align, 1 per Image.")
    else:
        print(f"Unexpected structure. Got {len(indices)} starts.")

if __name__ == "__main__":
    test_tokenization()
