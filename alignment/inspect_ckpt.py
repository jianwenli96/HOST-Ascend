import torch

ckpt_path = '/x2robot_v2/ethanchen/code/tcc_py_Qwen3/test/pytorch_model.bin'
try:
    state_dict = torch.load(ckpt_path, map_location='cpu')
    print("Keys in checkpoint:")
    for k in list(state_dict.keys())[:20]:
        print(k)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
