#!/bin/bash
set -e

# Reduce NPU allocator fragmentation for variable-size packed video batches.
# Respect an explicit caller-provided allocator configuration.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

NETWORK="${NETWORK:-Qwen3-VL-Embedding-8B}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
VIDEO_PATHS="${VIDEO_PATHS:-}"

if [ -z "${MODEL_NAME_OR_PATH}" ]; then
    echo "ERROR: set MODEL_NAME_OR_PATH to the local model directory or Hugging Face model ID." >&2
    echo "Example: MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-Embedding-8B VIDEO_PATHS=/path/to/video_paths.json bash train_scripts/run_ds.sh" >&2
    exit 1
fi

if [ -z "${VIDEO_PATHS}" ]; then
    echo "ERROR: set VIDEO_PATHS to your episode-list JSON before launching." >&2
    echo "Example: VIDEO_PATHS=/path/to/video_paths.json bash train_scripts/run_ds.sh" >&2
    exit 1
fi

# Set your own key in the environment before launching: export WANDB_API_KEY=...
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# If no wandb key is available (e.g. a fresh open-source checkout), fall back to
# offline logging rather than crashing at wandb.init()'s interactive login prompt.
if [ -z "${WANDB_API_KEY}" ]; then
    echo "[wandb] WARNING: WANDB_API_KEY is empty -> exporting WANDB_MODE=offline (local logging only). Set WANDB_API_KEY to enable cloud sync." >&2
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
BASE_PROJECT_NAME="host_alignment"

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
echo "NPU allocator config: $PYTORCH_NPU_ALLOC_CONF"

# Detect GPU count
if [ -n "${ASCEND_RT_VISIBLE_DEVICES+x}" ]; then
    NUM_GPUS=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l)
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
DS_CONFIG="scripts/ds_config_zero3.json"
# To resume from a previous run, set: RESUME_DIR="/path/to/previous/run/logs/tcc_qwen_alignment_<timestamp>"

LOGDIR="logs/$WANDB_PROJECT"
SAVE_INTERVAL=500
MAX_ITERS=10000
ARGS=(
    --alsologtostderr
    --logdir "$LOGDIR"
    --network "$NETWORK"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --video_paths "$VIDEO_PATHS"
    --ds_config "$DS_CONFIG"
    --save_interval "$SAVE_INTERVAL"
    --max_iters "$MAX_ITERS"
)

if [ -n "${RESUME_DIR:-}" ]; then
    ARGS+=(--resume_dir "$RESUME_DIR")
fi

if [ -n "${PRETRAIN_WEIGHTS:-}" ]; then
    ARGS+=(--pretrain_weights "$PRETRAIN_WEIGHTS")
fi

mkdir -p "$LOGDIR"

# Launch with torchrun
CMD=(
    torchrun
    "--nproc_per_node=$NUM_GPUS"
    "--nnodes=$WORLD_SIZE"
    "--node_rank=$RANK"
    "--master_addr=$MASTER_ADDR"
    "--master_port=$MASTER_PORT"
    train.py
    "${ARGS[@]}"
)
echo "Running: ${CMD[*]}"
"${CMD[@]}"
