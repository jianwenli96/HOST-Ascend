
import random
import torch
import numpy as np

# Simulate variables from datasets.py
full_len = 100
align_paths = list(range(full_len)) # 0..99
ref_chosen_steps = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])

print(f"Original Align Len: {len(align_paths)}")
print(f"Ref Steps: {ref_chosen_steps}")

# Simulation of Cut Logic in TCCCollator
full_align_len = len(align_paths)
cut_pct = 0.2
cut_len = int(full_align_len * cut_pct) # 20 frames

start_idx = 40 # Fixed for testing
end_idx = start_idx + cut_len # 60

print(f"Cutting indices [{start_idx}, {end_idx})")

# Apply Cut
align_paths_cut = align_paths[:start_idx] + align_paths[end_idx:]
print(f"Cut Align Len: {len(align_paths_cut)}")
print(f"Missing in Align: {[x for x in align_paths if x not in align_paths_cut]}")

# Create Mask
r_steps = ref_chosen_steps
mask = (r_steps >= start_idx) & (r_steps < end_idx)
is_cut_mask = mask.float()

print(f"Mask: {is_cut_mask}")

# Verify
cut_steps = r_steps[mask.bool()]
print(f"Masked Steps: {cut_steps}")

# Check consistency
for s in cut_steps:
    if s.item() in align_paths_cut:
        print(f"ERROR: Step {s} is masked but PRESENT in align_paths!")
    else:
        print(f"Step {s} is masked and correctly ABSENT from align_paths.")

# Check inverse
frame_kept_steps = r_steps[~mask.bool()]
for s in frame_kept_steps:
    # If s is in the range of the cut, it should be masked.
    # But since mask checks range, this is naturally true.
    pass
