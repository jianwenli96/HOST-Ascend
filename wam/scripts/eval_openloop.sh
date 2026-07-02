#!/usr/bin/env bash
# FastWAM Open-Loop Evaluation
# Usage: bash scripts/eval_openloop.sh

set -euo pipefail

# ── Configuration (edit here) ───────────────────────────────────────────
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./logs/real_joint_2cam_224_1e-4/2026-04-25_17-10-05}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"                  # empty = latest
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/open_data/cgy/processed_data/video_paths_basket/clean/flower_placemnet/10042_video_paths.json}"
DATASET_NAME="${DATASET_NAME:-10042}"
VIEWS="${VIEWS:-leftImg rightImg}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
PREDICTION_STRIDE="${PREDICTION_STRIDE:-15}"
USE_TASK_VIDEO="${USE_TASK_VIDEO:-true}"
USE_TASK_DESCRIPTION="${USE_TASK_DESCRIPTION:-true}"
VISUALIZE="${VISUALIZE:-false}"
REMOVE_STATIC_FRAMES="${REMOVE_STATIC_FRAMES:-auto}"    # auto = from checkpoint config
TASK_VIDEO_STATIC_THRESHOLD_RATIO="${TASK_VIDEO_STATIC_THRESHOLD_RATIO:-0.5}"
SEED="${SEED:-42}"
MAX_EPISODES="${MAX_EPISODES:-}"                     # empty = all
LOG_DIR="${LOG_DIR:-./logs_eval/$(basename $(dirname ${CHECKPOINT_DIR}))/$(basename ${CHECKPOINT_DIR})}"

# ── Build command ───────────────────────────────────────────────────────
CMD=(
    python eval/real_openloop/evaluate_openloop.py
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --eval_data_path "${EVAL_DATA_PATH}"
    --dataset_name "${DATASET_NAME}"
    --views ${VIEWS}
    --diffusion_steps "${DIFFUSION_STEPS}"
    --prediction_stride "${PREDICTION_STRIDE}"
    --seed "${SEED}"
    --log_dir "${LOG_DIR}"
)

[ -n "${CHECKPOINT_STEP}" ]             && CMD+=(--checkpoint_step "${CHECKPOINT_STEP}")
[ -n "${MAX_EPISODES}" ]                && CMD+=(--max_episodes "${MAX_EPISODES}")
[ "${USE_TASK_VIDEO}" = "false" ]       && CMD+=(--no_task_video)
[ "${USE_TASK_DESCRIPTION}" = "false" ] && CMD+=(--no_task_description)
[ "${VISUALIZE}" = "true" ]            && CMD+=(--visualize)
[ "${REMOVE_STATIC_FRAMES}" = "true" ]  && CMD+=(--remove_static_frames)
[ "${REMOVE_STATIC_FRAMES}" = "false" ] && CMD+=(--no_remove_static_frames)
CMD+=(--task_video_static_threshold_ratio "${TASK_VIDEO_STATIC_THRESHOLD_RATIO}")

# ── Launch ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "FastWAM Open-Loop Evaluation"
echo "  checkpoint:  ${CHECKPOINT_DIR}"
echo "  step:        ${CHECKPOINT_STEP:-latest}"
echo "  data:        ${EVAL_DATA_PATH}"
echo "  dataset:     ${DATASET_NAME}"
echo "  views:       ${VIEWS}"
echo "  diffusion:   ${DIFFUSION_STEPS} steps"
echo "  stride:      ${PREDICTION_STRIDE}"
echo "  task_video:  ${USE_TASK_VIDEO}"
echo "  task_desc:   ${USE_TASK_DESCRIPTION}"
echo "  visualize:   ${VISUALIZE}"
echo "  rm_static:   ${REMOVE_STATIC_FRAMES}"
echo "  tv_static_r: ${TASK_VIDEO_STATIC_THRESHOLD_RATIO}"
echo "  max_ep:      ${MAX_EPISODES:-all}"
echo "  log_dir:     ${LOG_DIR}"
echo "============================================================"

exec "${CMD[@]}"
