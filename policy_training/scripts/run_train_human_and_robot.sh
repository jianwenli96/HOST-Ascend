#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${POLICY_ROOT}"

# Physical NPU ids supplied for this machine.  torch_npu remaps them to local
# logical device ids 0..3 inside the training processes.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# The task config points at a complete local Wan2.2 DiT/VAE snapshot.  Refuse
# accidental network downloads during training.
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HUMAN_AND_ROBOT_ALIGN_DATA="${HUMAN_AND_ROBOT_ALIGN_DATA:-/mi/data2T/lijianwen/Datasets/ICL-TTT/HumanAndRobot/align_data}"
export WAN22_MODEL_DIR="${WAN22_MODEL_DIR:-/mi/data2T/Embodied-AI/ckpts/Wan-AI/Wan2.2-TI2V-5B}"

VIDEO_PATHS_JSON="${HUMAN_AND_ROBOT_ALIGN_DATA}/HumanAndRobot_video_paths.json"
if [ "${PRECOMPUTE_TEXT_EMBEDS:-true}" = "true" ]; then
  python scripts/precompute_text_embeds.py \
    task=human_and_robot_policy \
    +text_embedding_output_mode=instruction \
    +video_paths_json="${VIDEO_PATHS_JSON}" \
    model.model_id="${WAN22_MODEL_DIR}" \
    model.tokenizer_model_id="${WAN22_MODEL_DIR}" \
    model.redirect_common_files=false \
    +overwrite=false
fi

exec bash scripts/run_train.sh \
  task=human_and_robot_policy \
  "$@"
