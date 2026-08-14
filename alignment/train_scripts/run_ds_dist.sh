#!/bin/bash
set -eo pipefail

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
    echo "Example: MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-Embedding-8B VIDEO_PATHS=/path/to/video_paths.json bash train_scripts/run_ds_dist.sh" >&2
    exit 1
fi

if [ -z "${VIDEO_PATHS}" ]; then
    echo "ERROR: set VIDEO_PATHS to your episode-list JSON before launching." >&2
    echo "Example: VIDEO_PATHS=/path/to/video_paths.json bash train_scripts/run_ds_dist.sh" >&2
    exit 1
fi

# ============================================================================
# Cluster Configuration (from scheduler environment)
# ============================================================================
NNODES="${MA_NUM_HOSTS:-1}"                     # Total number of nodes
NODE_RANK="${VC_TASK_INDEX:-0}"                 # Current node rank
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"            # Master node IP (first in list)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"         # Fallback to localhost for single-node
MASTER_PORT="${MASTER_PORT:-29500}"             # Master port for distributed communication

# Number of GPUs per node (from scheduler or detect locally)
if [ -n "${MA_NUM_GPUS:-}" ]; then
    NUM_GPUS="${MA_NUM_GPUS}"
else
    # Detect GPU count locally
    if [ -n "${ASCEND_RT_VISIBLE_DEVICES+x}" ]; then
        NUM_GPUS=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l)
    elif command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        NUM_GPUS=0
    fi
fi

echo "Cluster Configuration:"
echo "  - Total Nodes (NNODES): $NNODES"
echo "  - Current Node Rank: $NODE_RANK"
echo "  - Master Address: $MASTER_ADDR"
echo "  - Master Port: $MASTER_PORT"
echo "  - GPUs per Node: $NUM_GPUS"

# ============================================================================
# HCCL / Ascend NPU Configuration
# ============================================================================
export HCCL_DEBUG=INFO
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

# ============================================================================
# WandB Configuration
# ============================================================================
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY}" ]; then
    echo "[wandb] WARNING: WANDB_API_KEY is empty -> exporting WANDB_MODE=offline (local logging only). Set WANDB_API_KEY to enable cloud sync." >&2
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
BASE_PROJECT_NAME="host_alignment"

# Synchronization logic for timestamp across nodes
mkdir -p "./train_sync"
SYNC_FILE="./train_sync/tcc_timestamp_${BASE_PROJECT_NAME}"

if [ "$NODE_RANK" -eq 0 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "${TIMESTAMP}" > "${SYNC_FILE}"
else
    echo "Node ${NODE_RANK} waiting for timestamp in ${SYNC_FILE}..."
    while [ ! -f "${SYNC_FILE}" ]; do
        sleep 1
    done
    TIMESTAMP=$(cat "${SYNC_FILE}")
fi

export WANDB_PROJECT="${BASE_PROJECT_NAME}_${TIMESTAMP}"
LOGDIR="logs/$WANDB_PROJECT"
TRAIN_LOG_DIR="$LOGDIR/train_logs"
mkdir -p "$TRAIN_LOG_DIR"

# Mirror subsequent launcher and training output to a persistent log while
# keeping it visible in the terminal. Avoid concurrent writes to one file in
# multi-node runs; the primary node keeps the conventional train.log name.
if [ "$NODE_RANK" -eq 0 ]; then
    LOG_FILE="$TRAIN_LOG_DIR/train.log"
else
    LOG_FILE="$TRAIN_LOG_DIR/train_node_${NODE_RANK}.log"
fi
exec > >(tee -a "$LOG_FILE") 2>&1

echo "WandB Project: $WANDB_PROJECT"
echo "Training log: $LOG_FILE"
echo "NPU allocator config: $PYTORCH_NPU_ALLOC_CONF"

# ============================================================================
# Training Configuration
# ============================================================================
DS_CONFIG="scripts/ds_config_zero3.json"
# To resume from a previous run, set: RESUME_DIR="/path/to/previous/run/logs/tcc_qwen_alignment_<timestamp>"

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

# ============================================================================
# Launch Distributed Training
# ============================================================================
CMD=(
    /efs-gy1/apps/miniconda3/envs/host_alignment/bin/python -m torch.distributed.run
    "--nproc_per_node=$NUM_GPUS"
    "--nnodes=$NNODES"
    "--node_rank=$NODE_RANK"
    "--master_addr=$MASTER_ADDR"
    "--master_port=$MASTER_PORT"
    train.py
    "${ARGS[@]}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
