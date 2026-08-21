#!/usr/bin/env bash

set -euo pipefail

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

HUMAN_AND_ROBOT_ALIGN_DATA="/efs-gy1/Embodied-AI/datasets/ICL-TTT/HumanAndRobot/align_data"
VIDEO_PATHS_JSON="${HUMAN_AND_ROBOT_ALIGN_DATA}/HumanAndRobot_video_paths.json"
WAN22_MODEL_DIR="/efs-gy1/Embodied-AI/ckpts/Wan-AI/Wan2.2-TI2V-5B"

python scripts/precompute_text_embeds.py \
    task=human_and_robot_policy \
    model.model_id="${WAN22_MODEL_DIR}" \
    model.tokenizer_model_id="${WAN22_MODEL_DIR}" \
    model.redirect_common_files=false \
    +text_embedding_output_mode=instruction \
    +video_paths_json="${VIDEO_PATHS_JSON}" \
    +overwrite=true