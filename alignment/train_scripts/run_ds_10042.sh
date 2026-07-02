#!/bin/bash
set -e

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
  unset CUDA_HOME CUDA_PATH CUDAToolkit_ROOT
  export CUDA_HOME="${CONDA_PREFIX}"
  export CUDA_PATH="${CONDA_PREFIX}"
  export CUDAToolkit_ROOT="${CONDA_PREFIX}"
fi

# --- Auto-link logs to checkpoint storage ---
PROJECT_NAME=$(basename "$PWD")
TARGET_BASE="/mnt/data/checkpoint/ethanchen/code"
TARGET_PROJECT_DIR="$TARGET_BASE/$PROJECT_NAME"
TARGET_LOGS="$TARGET_PROJECT_DIR/logs"
LOCAL_LOGS="./logs"


# Ensure target project directory exists
# if [ ! -d "$TARGET_PROJECT_DIR" ]; then
#     mkdir -p "$TARGET_PROJECT_DIR"
# fi

# # Handle logs directory
# if [ -d "$LOCAL_LOGS" ] && [ ! -L "$LOCAL_LOGS" ]; then
#     echo "Detected local 'logs' directory. Moving to checkpoint storage ($TARGET_LOGS)..."
#     if [ ! -d "$TARGET_LOGS" ]; then
#         mv "$LOCAL_LOGS" "$TARGET_LOGS"
#     else
#         echo "Target $TARGET_LOGS already exists. Merging contents..."
#         # Move contents, ignore errors if files exist
#         mv "$LOCAL_LOGS"/* "$TARGET_LOGS"/ 2>/dev/null || true
#         rm -rf "$LOCAL_LOGS"
#     fi
#     ln -s "$TARGET_LOGS" "$LOCAL_LOGS"
# elif [ ! -e "$LOCAL_LOGS" ]; then
#     echo "Creating 'logs' symlink pointing to $TARGET_LOGS..."
#     mkdir -p "$TARGET_LOGS"
#     ln -s "$TARGET_LOGS" "$LOCAL_LOGS"
# fi
# --------------------------------------------

# Set your own key in the environment before launching: export WANDB_API_KEY=...
export WANDB_API_KEY="${WANDB_API_KEY:-}"
BASE_PROJECT_NAME="tcc_qwen_alignment_10042"

# Synchronization logic for timestamp
# Assuming RANK is provided by environment (e.g. SLURM_NODEID) or defaults to 0
RANK=${RANK:-${SLURM_NODEID:-0}}
mkdir -p "./train_sync"
SYNC_FILE="./train_sync/tcc_timestamp_${BASE_PROJECT_NAME}"

if [ "$RANK" -eq 0 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "${TIMESTAMP}" > "${SYNC_FILE}"
else
    echo "Node ${RANK} waiting for timestamp in ${SYNC_FILE}..."
    # Wait loop
    while [ ! -f "${SYNC_FILE}" ]; do
        sleep 1
    done
    TIMESTAMP=$(cat "${SYNC_FILE}")
fi

export WANDB_PROJECT="${BASE_PROJECT_NAME}_${TIMESTAMP}"
echo "WandB Project: $WANDB_PROJECT"

# Detect GPU count
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l)
elif command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=0
fi

echo "Detected $NUM_GPUS GPUs."

# Torchrun configuration
# DLC/Kubernetes usually provide these. We set defaults for local run.
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}

# Define arguments
NETWORK="Qwen3-VL-2B"
VIDEO_PATHS="/open_data/cgy/processed_data/video_paths_basket/clean/10042_video_paths.json"
# VIDEO_PATHS="/open_data/cgy/processed_data/video_paths_basket/clean/rt1_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/fmb_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/SSv2_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/bridgev2_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/calvin_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/libero_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/maniskill_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/droid_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/clean/robocoin_video_paths.json"
# VIDEO_PATHS="/open_data/cgy/processed_data/video_paths_basket/libero_video_paths.json,/open_data/cgy/processed_data/video_paths_basket/berkeley_autolab_ur5_video_paths.json"
DS_CONFIG="scripts/ds_config_zero3.json"
# RESUME_DIR="/mnt/data/checkpoint/ethanchen/code/tcc_py_Qwen3_video/logs/tcc_qwen_alignment_20260107_155137" 
# RESUME_DIR="/x2robot_v2/ethanchen/code/tcc_py_Qwen3_video_fast_3_3_aug_high_reverse_causal_2/logs/tcc_qwen_alignment_20260115_232104" 
# RESUME_DIR="/x2robot_v2/ethanchen/code/tcc_py_Qwen3_video_fast_3_3_aug_high_reverse_causal_dustbin_eval_var_attn_pool_2_e_3_vid_2/logs/tcc_qwen_alignment_20260129_204418" 
# RESUME_DIR="/x2robot_v2/ethanchen/code/tcc_py_Qwen3_video_fast_3_3_aug_high_reverse_causal_dustbin_eval_var_attn_pool_2_e_3_vid_2_large_2_mp4_dtw_2_1_easy_no_attn_EMA_R_test/logs/tcc_qwen_alignment_20260221_212251"
# RESUME_DIR="/x2robot_v2/ethanchen/code/tcc_py_Qwen3_video_fast_3_3_aug_high_reverse_causal_dustbin_eval_var_attn_pool_2_e_3_vid_2_large_2_mp4_dtw_2_1_all_no_attn_EMA_R_test_agi_cut_filter/logs/tcc_qwen_alignment_20260305_224554"
# RESUME_DIR="/x2robot_v2/ethanchen/code/tcc_py_Qwen3_video_fast_3_3_aug_high_reverse_causal_dustbin_eval_var_attn_pool_2_e_3_vid_2_large_2_mp4_dtw_2_1_all_no_attn_R_test_agi_cut_filter_eval_bk_align_diff/logs/tcc_qwen_alignment_RoboChallenge_20260316_215350"
# RESUME_DIR="/x2robot_v2/ethanchen/code/Video_alignment_SmoothDTW_3view_no_his/logs/tcc_all_qwen_alignment_20260405_105444"

# Calculate Gradient Accumulation Steps
# Target: 4 / WORLD_SIZE (Nodes)
# Ensure it is at least 1
GRAD_ACCUM_STEPS=$((4 / WORLD_SIZE))
if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi
echo "Setting Gradient Accumulation Steps to: $GRAD_ACCUM_STEPS (World Size: $WORLD_SIZE)"

LOGDIR="logs/$WANDB_PROJECT"
SAVE_INTERVAL=500
MAX_ITERS=3000
# NUM_ALIGN_FRAMES=24
ARGS="--alsologtostderr --logdir $LOGDIR --network $NETWORK --video_paths $VIDEO_PATHS --gradient_accumulation_steps $GRAD_ACCUM_STEPS --ds_config $DS_CONFIG --save_interval $SAVE_INTERVAL --max_iters $MAX_ITERS"

if [ ! -z "$RESUME_DIR" ]; then
    ARGS="$ARGS --resume_dir $RESUME_DIR"
fi

if [ ! -z "$PRETRAIN_WEIGHTS" ]; then
    ARGS="$ARGS --pretrain_weights $PRETRAIN_WEIGHTS"
fi

mkdir -p "$LOGDIR"
chown -R 2103:2103 "$LOGDIR"

# Launch with torchrun
CMD="torchrun --nproc_per_node=$NUM_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $ARGS"
echo "Running: $CMD"
$CMD

