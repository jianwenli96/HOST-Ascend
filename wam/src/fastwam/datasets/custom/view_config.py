# -*- coding: utf-8 -*-
"""
Instruction filename overrides per dataset (JSON mode).

Multi-camera view combinations are defined per-task in cam_mapping JSON files
(cam_mapping_dir) and sampled in train/datasets.py (num_view_probs).
"""

# video_info["dataset"] (str) -> instruction file basename under episode/segment dir.
# If set, that file is tried first; missing file falls back to instruction.txt.
DATASET_INSTRUCTION_FILENAME = {
    "10042": "all_instruction.txt",
    "10058": "all_instruction.txt",
    "10053": "all_instruction.txt",
}
