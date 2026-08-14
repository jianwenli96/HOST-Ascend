#!/bin/bash
set -e # Exit on error

# Usage: ./scripts/convert_ds_to_hf.sh <log_directory> [model_name_or_path]
# Optional: CONVERT_HF=1 also runs step 3 (HuggingFace checkpoint conversion).
# When CONVERT_HF=1, model_name_or_path is required.
# Example: CONVERT_HF=1 ./scripts/convert_ds_to_hf.sh logs/tcc_qwen_alignment_20260104_162651 /path/to/model

INPUT_ARG=${1:-"tcc_qwen_alignment_"}
MODEL_PATH=${2:-}

if [ -z "$INPUT_ARG" ]; then
    echo "Usage: $0 <log_directory_or_pattern> [model_name_or_path]"
    echo "Example: $0 logs/tcc_qwen_alignment_20260104_162651"
    echo "Example: $0 tcc_qwen_alignment_ (finds latest)"
    exit 1
fi

if [ "${CONVERT_HF:-0}" = "1" ] && [ -z "$MODEL_PATH" ]; then
    echo "Error: model_name_or_path is required when CONVERT_HF=1."
    echo "Usage: CONVERT_HF=1 $0 <log_directory_or_pattern> <model_name_or_path>"
    exit 1
fi

# Resolve Log Directory
if [ -d "$INPUT_ARG" ]; then
    LOG_DIR="$INPUT_ARG"
else
    # Handle {date} placeholder
    CURRENT_DATE=$(date +%Y%m%d)
    SEARCH_PATTERN=${INPUT_ARG//\{date\}/$CURRENT_DATE}
    
    echo "Searching for latest directory matching: $SEARCH_PATTERN ..."
    
    # Try to find matching directories, sorted by name (reverse) to get latest timestamp
    # 1. Try pattern in current directory
    MATCHES=$(ls -d ${SEARCH_PATTERN}* 2>/dev/null | sort -r)
    LOG_DIR=$(echo "$MATCHES" | head -n 1)
    
    # 2. If not found, try inside logs/
    if [ -z "$LOG_DIR" ]; then
        MATCHES=$(ls -d logs/${SEARCH_PATTERN}* 2>/dev/null | sort -r)
        LOG_DIR=$(echo "$MATCHES" | head -n 1)
    fi
    
    if [ -z "$LOG_DIR" ]; then
        echo "Error: No directories found matching '$SEARCH_PATTERN' or 'logs/$SEARCH_PATTERN'"
        exit 1
    fi
    
    if [ ! -d "$LOG_DIR" ]; then
         echo "Error: Found '$LOG_DIR' but it is not a directory."
         exit 1
    fi
    
    echo "Auto-selected latest log directory: $LOG_DIR"
fi

# Convert to absolute path
LOG_DIR=$(realpath $LOG_DIR)
# Assuming script is in scripts/ folder, root is one level up
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

echo "========================================================"
echo "Processing Log Directory: $LOG_DIR"
if [ -n "$MODEL_PATH" ]; then
    echo "Base Model Path: $MODEL_PATH"
fi
echo "========================================================"

# 1. Prepare Output Directories
FP32_DIR="$LOG_DIR/fp32_converted"
HF_DIR="$LOG_DIR/hf_converted"

echo "[1/3] Creating output directories..."
mkdir -p $FP32_DIR
echo "  - FP32 Output: $FP32_DIR"
if [ "${CONVERT_HF:-0}" = "1" ]; then
    mkdir -p $HF_DIR
    echo "  - HF Output:   $HF_DIR"
fi

# 2. Run DeepSpeed zero_to_fp32
OUTPUT_BIN="$FP32_DIR/pytorch_model.bin"

echo "[2/3] Running DeepSpeed zero_to_fp32 conversion..."

# Helper: Check for 'latest' file or checkpoint structure
if [ ! -f "$LOG_DIR/latest" ]; then
    # Check if the provided dir itself has .pt files (is the checkpoint dir)
    if ls $LOG_DIR/*.pt 1> /dev/null 2>&1; then
        echo "  - Detected checkpoint files directly in $LOG_DIR. Proceeding..."
    else
        echo "  - Warning: 'latest' file not found in $LOG_DIR."
        # Try to find subdirectories that look like checkpoints (digits or global_step)
        # Find directories that are purely digits or start with global_step
        CKPT_DIRS=$(find $LOG_DIR -maxdepth 1 -type d | grep -E "/[0-9]+$|/global_step[0-9]+$")
        
        # Count lines
        NUM_CKPTS=$(echo "$CKPT_DIRS" | grep -c "^")
        
        if [ "$NUM_CKPTS" -eq "1" ]; then
            CKPT_NAME=$(basename $CKPT_DIRS)
            echo "  - Found single checkpoint folder: $CKPT_NAME. Creating 'latest' file."
            echo $CKPT_NAME > "$LOG_DIR/latest"
        elif [ "$NUM_CKPTS" -gt "1" ]; then
            echo "  - Found multiple checkpoints. Please ensure 'latest' file points to the desired one, or pass the specific checkpoint folder."
            echo "  - Candidates:"
            echo "$CKPT_DIRS"
        fi
    fi
fi

python -m deepspeed.utils.zero_to_fp32 $LOG_DIR $OUTPUT_BIN

echo "  - FP32 binary saved to: $OUTPUT_BIN"

# 3. Run HF Conversion (optional)
CONVERT_SCRIPT="$PROJECT_ROOT/scripts/convert_ckpt_to_hf.py"

if [ "${CONVERT_HF:-0}" = "1" ]; then
    echo "[3/3] Converting to HuggingFace format..."
    python "$CONVERT_SCRIPT" \
        --checkpoint_path "$OUTPUT_BIN" \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$HF_DIR"
    echo "========================================================"
    echo "SUCCESS!"
    echo "HuggingFace model is ready at: $HF_DIR"
    echo "========================================================"
else
    echo "[3/3] Skipped HuggingFace conversion (set CONVERT_HF=1 to enable)."
    echo "========================================================"
    echo "SUCCESS!"
    echo "FP32 checkpoint: $OUTPUT_BIN"
    echo "========================================================"
fi
