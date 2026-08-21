#!/usr/bin/env bash
# Distributed training script for internal cluster (DLC / torchrun)
# Usage: bash scripts/run_train_dist.sh [hydra_overrides...]

set -euo pipefail

# Ensure we are in the project root directory
CURR_FOLDER="$(cd "$(dirname "$0")/.." && pwd)"

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
    elif command -v npu-smi &> /dev/null; then
        NUM_GPUS=$(npu-smi info -l | grep "Total Count" | awk '{print $NF}')
    else
        NUM_GPUS=0
    fi
fi

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No NPUs detected. Set MA_NUM_GPUS, ASCEND_RT_VISIBLE_DEVICES, or run on a node with npu-smi." >&2
    exit 1
fi

echo "Cluster Configuration:"
echo "  - Total Nodes (NNODES): $NNODES"
echo "  - Current Node Rank: $NODE_RANK"
echo "  - Master Address: $MASTER_ADDR"
echo "  - Master Port: $MASTER_PORT"
echo "  - NPUs per Node: $NUM_GPUS"

EXTRA_ARGS=("$@")

# ── Task basename ────────────────────────────────────────────────────────────
TASK_BASENAME="human_and_robot_policy"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done
# Append suffix to distinguish runs in logs and wandb
TASK_SUFFIX="base"
FILTERED_ARGS=()
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    task_suffix=*)
      TASK_SUFFIX="${arg#task_suffix=}"
      ;;
    *)
      FILTERED_ARGS+=("$arg")
      ;;
  esac
done
EXTRA_ARGS=("${FILTERED_ARGS[@]}")
TASK_FULLNAME="${TASK_BASENAME}_${TASK_SUFFIX}"

# ── Timestamp synchronization via TCPStore (bypasses NFS caching issues) ─────
BASE_PROJECT_NAME="FAST-WAM"
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

# ── RUN_ID (reuse synced timestamp) ─────────────────────────────────────────
if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="${TIMESTAMP}"
fi

# ── wandb ────────────────────────────────────────────────────────────────────
WANDB_PROJECT="${BASE_PROJECT_NAME}"
# Set your own key in the environment before launching: export WANDB_API_KEY=...
WANDB_API_KEY="${WANDB_API_KEY:-}"
# If no wandb key is available (e.g. a fresh open-source checkout), fall back to
# offline logging rather than crashing at wandb.init()'s interactive login prompt.
if [ -z "${WANDB_API_KEY}" ]; then
    echo "[wandb] WARNING: WANDB_API_KEY is empty -> exporting WANDB_MODE=offline (local logging only). Set WANDB_API_KEY to enable cloud sync." >&2
    WANDB_MODE="${WANDB_MODE:-offline}"
fi
# Only the global master node (NODE_RANK=0) logs to wandb; all other nodes are disabled
if [ "${NODE_RANK:-0}" -ne 0 ]; then
    WANDB_MODE=disabled
fi
WANDB_SAVE_CODE=true

TOTAL_PROCESSES=$((NUM_GPUS * NNODES))

echo "[launch] ngpus=${NUM_GPUS} nnodes=${NNODES} node_rank=${NODE_RANK} total_processes=${TOTAL_PROCESSES} run_id=${RUN_ID}"

# ── Log directory setup ──────────────────────────────────────────────────────
LOGDIR="${CURR_FOLDER}/logs/${TASK_FULLNAME}/${RUN_ID}"
TRAIN_LOG_DIR="$LOGDIR/train_logs"
mkdir -p ${TRAIN_LOG_DIR}

# Mirror subsequent launcher and training output to a persistent log while
# keeping it visible in the terminal. Avoid concurrent writes to one file in
# multi-node runs; the primary node keeps the conventional train.log name.
if [ "$NODE_RANK" -eq 0 ]; then
    LOG_FILE="$TRAIN_LOG_DIR/train.log"
else
    LOG_FILE="$TRAIN_LOG_DIR/train_node_${NODE_RANK}.log"
fi

echo "WandB Project: $WANDB_PROJECT"
echo "Training log: $LOG_FILE"

# ── Launch ───────────────────────────────────────────────────────────────────
sudo -i bash -i -c "
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" && \
  export HCCL_CONNECT_TIMEOUT=6000 && \
  export WANDB_PROJECT=${WANDB_PROJECT} && \
  export WANDB_API_KEY=${WANDB_API_KEY} && \
  export WANDB_MODE=${WANDB_MODE} && \
  export WANDB_SAVE_CODE=${WANDB_SAVE_CODE} && \
  conda activate host_policy && \
  cd ${CURR_FOLDER} && \
  accelerate launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_processes ${TOTAL_PROCESSES} \
    --num_machines ${NNODES} \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train.py \
    output_dir=${LOGDIR} \
    wandb.enabled=true \
    wandb.name=${TASK_FULLNAME}_${RUN_ID} \
    wandb.project=${WANDB_PROJECT} \
    task=${TASK_BASENAME} \
    ${EXTRA_ARGS[@]}
" 2>&1 | tee -a ${LOG_FILE}
