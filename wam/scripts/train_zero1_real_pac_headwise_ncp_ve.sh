#!/usr/bin/env bash
# Train FastWAM with visual encoder + noisy clean progress token (pac_headwise + ncp + DINOv2)
# Usage: bash scripts/train_zero1_real_pac_headwise_ncp_ve.sh [hydra_overrides...]

set -euo pipefail

# ── Environment variables (from DLC / torchrun) ─────────────────────────────
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

EXTRA_ARGS=("$@")

# ── Task basename ────────────────────────────────────────────────────────────
TASK_BASENAME="real_joint_2cam_224_1e-4_pac_headwise_ncp_ve"
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
TASK_SUFFIX="pac_hw_ncp_ve"
TASK_FULLNAME="${TASK_BASENAME}_${TASK_SUFFIX}"

# ── TIMESTAMP sync (coordinator filename for cross-DLC rendezvous) ──────────
mkdir -p logs/rdzv
_SYNC_FILE="logs/.exp_timestamp_${MASTER_PORT}"

_IS_GLOBAL_MASTER=0
if [ "${RANK:-0}" -eq 0 ]; then
    if [ -z "${CROSS_JOB_ID:-}" ] || [ "${CROSS_JOB_ID}" -eq 0 ]; then
        _IS_GLOBAL_MASTER=1
    fi
fi

if [ "${_IS_GLOBAL_MASTER}" -eq 1 ]; then
    rm -f "${_SYNC_FILE}"
    _TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    echo "${_TIMESTAMP}" > "${_SYNC_FILE}"
    echo "[timestamp] generated: ${_TIMESTAMP}"
    ( sleep 300 && rm -f "${_SYNC_FILE}" ) &
else
    echo "[timestamp] waiting for ${_SYNC_FILE} ..."
    while true; do
        if [ -f "${_SYNC_FILE}" ]; then
            _FILE_AGE=$(( $(date +%s) - $(stat -c %Y "${_SYNC_FILE}") ))
            if [ "${_FILE_AGE}" -lt 300 ]; then
                _TIMESTAMP=$(cat "${_SYNC_FILE}")
                echo "[timestamp] got: ${_TIMESTAMP} (age ${_FILE_AGE}s)"
                break
            fi
        fi
        sleep 1
    done
fi

# ── Cross-DLC override (before RUN_ID sync so TCPStore sees global values) ──
# Usage: set in each DLC job's task cmd:
#   Job 0: CROSS_JOB_ID=0 CROSS_JOB_TOTAL_NODES=<total_nodes>
#   Job 1: CROSS_JOB_ID=1 CROSS_JOB_TOTAL_NODES=<total_nodes>
# logs/rdzv/ must be on a shared filesystem visible to all DLC nodes.
# If CROSS_JOB_ID is unset, this block is skipped entirely.
if [ -n "${CROSS_JOB_ID:-}" ]; then
    _DLC_RANK=${RANK:-0}
    _DLC_NODES=${WORLD_SIZE:-1}
    _CROSS_JOB_ID=${CROSS_JOB_ID}
    if [ -z "${CROSS_JOB_TOTAL_NODES:-}" ]; then
        echo "[cross-job] ERROR: CROSS_JOB_TOTAL_NODES must be set when CROSS_JOB_ID is set" >&2
        exit 1
    fi
    _TOTAL_NODES=${CROSS_JOB_TOTAL_NODES}
    _GLOBAL_RANK=$(( _CROSS_JOB_ID * _DLC_NODES + _DLC_RANK ))
    _COORD_FILE="logs/rdzv/.cross_job_coord_${MASTER_PORT}_${_TIMESTAMP}.txt"

    if [ "${_CROSS_JOB_ID}" -eq 0 ] && [ "${_DLC_RANK}" -eq 0 ]; then
        _MY_IP=$(ip -4 addr show eth0 | awk '/inet /{print $2}' | cut -d/ -f1)
        if [ -z "${_MY_IP}" ]; then
            echo "[cross-job] WARNING: eth0 not found, falling back to hostname -I" >&2
            _MY_IP=$(hostname -I | awk '{print $1}')
        fi
        _RDZV_FILE="logs/rdzv/rdzv_${_TIMESTAMP}.txt"
        echo "${_MY_IP}" > "${_RDZV_FILE}"
        echo "${_RDZV_FILE}" > "${_COORD_FILE}.tmp" && mv "${_COORD_FILE}.tmp" "${_COORD_FILE}"
        echo "[cross-job] master IP written: ${_MY_IP} -> ${_RDZV_FILE}"
        _CROSS_MASTER_IP="${_MY_IP}"
    else
        echo "[cross-job] job=${_CROSS_JOB_ID} rank=${_DLC_RANK}: waiting for ${_COORD_FILE} ..."
        while [ ! -f "${_COORD_FILE}" ]; do sleep 1; done
        _RDZV_FILE=$(cat "${_COORD_FILE}")
        while [ ! -f "${_RDZV_FILE}" ]; do sleep 1; done
        _CROSS_MASTER_IP=$(cat "${_RDZV_FILE}")
        echo "[cross-job] job=${_CROSS_JOB_ID} rank=${_DLC_RANK}: master=${_CROSS_MASTER_IP}"
    fi

    _CROSS_PORT=${CROSS_JOB_RDZV_PORT:-${MASTER_PORT}}
    export WORLD_SIZE=${_TOTAL_NODES}
    export RANK=${_GLOBAL_RANK}
    export MASTER_ADDR=${_CROSS_MASTER_IP}
    export MASTER_PORT=${_CROSS_PORT}
    echo "[cross-job] effective: WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
fi

TOTAL_PROCESSES=$((NGPUS * WORLD_SIZE))

# ── RUN_ID (reuse synced timestamp — no separate TCPStore needed) ────────────
if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="${_TIMESTAMP}"
fi

echo "[launch] ngpus=${NGPUS} world_size=${WORLD_SIZE} rank=${RANK} total_processes=${TOTAL_PROCESSES} run_id=${RUN_ID}"

# ── wandb ────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="FAST-WAM"
# Set your own key in the environment before launching: export WANDB_API_KEY=...
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# If no wandb key is available (e.g. a fresh open-source checkout), fall back to
# offline logging rather than crashing at wandb.init()'s interactive login prompt.
if [ -z "${WANDB_API_KEY}" ]; then
    echo "[wandb] WARNING: WANDB_API_KEY is empty -> exporting WANDB_MODE=offline (local logging only). Set WANDB_API_KEY to enable cloud sync." >&2
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
# Only the global master node (RANK=0) logs to wandb; all other nodes are disabled
# to prevent duplicate runs when each DLC node launches its own accelerate process.
if [ "${RANK:-0}" -ne 0 ]; then
    export WANDB_MODE=disabled
fi
export WANDB_SAVE_CODE=true
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    export WANDB_GIT_COMMIT=$(git rev-parse HEAD)
    export WANDB_GIT_REMOTE_URL=$(git config --get remote.origin.url)
fi

# ── Launch ───────────────────────────────────────────────────────────────────
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${WORLD_SIZE}" \
  --machine_rank "${RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train.py \
  "output_dir=./logs/${TASK_FULLNAME}/${RUN_ID}" \
  "wandb.enabled=true" \
  "wandb.name=${TASK_FULLNAME}_${RUN_ID}" \
  "wandb.project=FAST-WAM" \
  "model=fastwam_joint_cross_attn_ve" \
  "task=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
