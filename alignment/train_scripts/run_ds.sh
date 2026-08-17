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

# Torchrun configuration
# DLC/Kubernetes usually provide these. We set defaults for local run.
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${NNODES:-1}          # Total number of nodes
NODE_RANK=${NODE_RANK:-0}    # Current node rank

# Synchronization logic for timestamp across nodes using TCPStore
# This bypasses NFS caching issues that can cause different nodes to use different timestamps
if [ "$NNODES" -le 1 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
else
    SYNC_PORT="${TIMESTAMP_SYNC_PORT:-$((MASTER_PORT + 11))}"
    SYNC_TIMEOUT="${TIMESTAMP_SYNC_TIMEOUT:-180}"

    TIMESTAMP=$(python3 - <<PY
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = "${MASTER_ADDR}"
port = ${SYNC_PORT}
timeout_s = ${SYNC_TIMEOUT}
node_rank = ${NODE_RANK}
num_nodes = ${NNODES}

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_nodes,
    is_master=(node_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = "timestamp::${BASE_PROJECT_NAME}"
if node_rank == 0:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    store.set(key, ts)
ts = store.get(key).decode("utf-8")
print(ts)
PY
    )
    echo "Timestamp synchronized via TCPStore: ${TIMESTAMP} (port=${SYNC_PORT})"
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

# Detect GPU count
if [ -n "${ASCEND_RT_VISIBLE_DEVICES+x}" ]; then
    NUM_GPUS=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l)
elif command -v npu-smi &> /dev/null; then
    NUM_GPUS=$(npu-smi info -l | grep "Total Count" | awk '{print $NF}')
else
    NUM_GPUS=0
fi

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No NPUs detected. Set ASCEND_RT_VISIBLE_DEVICES or run on a node with npu-smi." >&2
    exit 1
fi

echo "Cluster Configuration:"
echo "  - Total Nodes (NNODES): $NNODES"
echo "  - Current Node Rank: $NODE_RANK"
echo "  - Master Address: $MASTER_ADDR"
echo "  - Master Port: $MASTER_PORT"
echo "  - NPUs per Node: $NUM_GPUS"

# Define arguments
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

# Launch with torchrun
conda activate host_alignment
CMD=(
    torchrun
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
