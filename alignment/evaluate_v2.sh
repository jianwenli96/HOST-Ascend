#!/bin/bash
set -eo pipefail

# Reduce NPU allocator fragmentation for variable-size packed video batches.
# Respect an explicit caller-provided allocator configuration.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# Ensure we are in the project root directory
cd "$(dirname "$0")"

# ============================================================================
# HCCL / Ascend NPU Configuration
# ============================================================================
export HCCL_DEBUG=INFO
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

# Running parameters
HOST_QWEN_MODEL="/path/to/ckpts/Qwen3-VL-Embedding-8B"
HOST_ALIGN_DATA="/path/to/dataset/align-data/video-paths.json"
HOST_ALIGN_RUN="/path/to/trained/ckpts/alignment-model"
HOST_ALIGN_EVAL="${HOST_ALIGN_RUN}/eval_results"

torchrun \
  --nproc_per_node=16 \
  --master_port=29501 \
  evaluate_v2.py \
  --model_name_or_path="$HOST_QWEN_MODEL" \
  --resume_dir="$HOST_ALIGN_RUN" \
  --video_paths="$HOST_ALIGN_DATA" \
  --logdir="$HOST_ALIGN_EVAL" \
  --network=Qwen3-VL-Embedding-8B \
  --batch_size=1 \
  --eval_chunk_probs=0,0,1 \
  --ds_config=scripts/ds_config_zero3.json
