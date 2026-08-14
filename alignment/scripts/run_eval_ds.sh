#!/bin/bash
set -e

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

# --- Same WandB setup as run_ds.sh ---
# Set your own key in the environment before launching: export WANDB_API_KEY=...
export WANDB_API_KEY="${WANDB_API_KEY:-}"
BASE_PROJECT_NAME="tcc_qwen_alignment"

# Synchronization logic for timestamp
RANK=${RANK:-${SLURM_NODEID:-0}}
SYNC_FILE="./.tcc_timestamp.sync"

if [ "$RANK" -eq 0 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "${TIMESTAMP}" > "${SYNC_FILE}"
else
    echo "Node ${RANK} waiting for timestamp in ${SYNC_FILE}..."
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

echo "Detected $NUM_GPUS GPUs for evaluation."

# Torchrun configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29501"}  # Different port to avoid conflicts
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}

# Define default arguments
NETWORK="Qwen3-VL-Embedding-8B"
MODEL_NAME_OR_PATH=""
# Evaluation dataset path — pass your own via --video_paths
VIDEO_PATHS=""

# Checkpoint directory to evaluate — required, pass via --resume_dir
RESUME_DIR=""

# Evaluation parameters
BATCH_SIZE=1  # Must be divisible by 4 for multiplier=4
EVAL_CHUNK_PROBS="0,0,1"  # Force 4x multiplier (96 frames total)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume_dir)
            RESUME_DIR="$2"
            shift 2
            ;;
        --video_paths)
            VIDEO_PATHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_chunk_probs)
            EVAL_CHUNK_PROBS="$2"
            shift 2
            ;;
        --network)
            NETWORK="$2"
            shift 2
            ;;
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_NAME_OR_PATH" ]; then
    echo "Error: --model_name_or_path must be specified for evaluation."
    echo "Usage: $0 --model_name_or_path /path/to/model --resume_dir /path/to/checkpoint --video_paths /path/to/video_paths.json"
    exit 1
fi

# Check if RESUME_DIR is provided
if [ -z "$RESUME_DIR" ]; then
    echo "Error: --resume_dir must be specified for evaluation."
    echo "Usage: $0 --resume_dir /path/to/checkpoint [--video_paths /path/to/video_paths.json] [--batch_size 4] [--eval_chunk_probs 0,0,1]"
    exit 1
fi

if [ ! -d "$RESUME_DIR" ]; then
    echo "Error: Checkpoint directory $RESUME_DIR does not exist."
    exit 1
fi

# Check if VIDEO_PATHS is provided
if [ -z "$VIDEO_PATHS" ]; then
    echo "Error: --video_paths must be specified for evaluation."
    echo "Usage: $0 --resume_dir /path/to/checkpoint --video_paths /path/to/video_paths.json [--batch_size 4] [--eval_chunk_probs 0,0,1]"
    exit 1
fi

# Set LOGDIR using same format as run_ds.sh
LOGDIR="logs/$WANDB_PROJECT"
mkdir -p "$LOGDIR"
chown -R 2103:2103 "$LOGDIR" 2>/dev/null || true

echo "==================================="
echo "Evaluation Configuration:"
echo "  Loading checkpoint from: $RESUME_DIR"
echo "  Saving results to: $LOGDIR"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Network: $NETWORK"
echo "  Model Name or Path: $MODEL_NAME_OR_PATH"
echo "  Video Paths: $VIDEO_PATHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Eval Chunk Probs: $EVAL_CHUNK_PROBS (1x, 2x, 4x)"
echo "  GPUs: $NUM_GPUS"
echo "==================================="

# Build arguments (using absl flags format for evaluate_v2.py)
ARGS=(
    "--logdir=$LOGDIR"
    "--resume_dir=$RESUME_DIR"
    "--network=$NETWORK"
    "--model_name_or_path=$MODEL_NAME_OR_PATH"
    "--video_paths=$VIDEO_PATHS"
    "--batch_size=$BATCH_SIZE"
    "--eval_chunk_probs=$EVAL_CHUNK_PROBS"
    --ds_config=scripts/ds_config_zero3.json
)

# Launch with torchrun - use evaluate_v2.py which is based on train.py
CMD=(
    torchrun
    "--nproc_per_node=$NUM_GPUS"
    "--nnodes=$WORLD_SIZE"
    "--node_rank=$RANK"
    "--master_addr=$MASTER_ADDR"
    "--master_port=$MASTER_PORT"
    evaluate_v2.py
    "${ARGS[@]}"
)
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
