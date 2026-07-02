# Training vs Inference Workflow Comparison

This document provides a detailed comparison between the training workflow (`run_ds.sh`) and the inference workflow (`run.sh` / `extract_embeddings.py`).

## 1. Overview

| Feature | Training (`run_ds.sh`) | Inference (`run.sh`) |
| :--- | :--- | :--- |
| **Script** | `torchrun` -> `train.py` | `python extract_embeddings.py` |
| **Framework** | **DeepSpeed** (Distributed Training) | **Native PyTorch** (Single Process) |
| **Parallelism** | Multi-GPU / Multi-Node (Data Parallel) | Single GPU |
| **Checkpoint Output** | DeepSpeed Partitioned Checkpoints (e.g., `global_step1000/`) | N/A (Read-only) |
| **Checkpoint Input** | DeepSpeed Checkpoints | Standard PyTorch State Dict (`checkpoint.pth.tar` or `fp32_converted/pytorch_model.bin`) |
| **Batch Size** | Defined via `ds_config` & micro-batch args | Defined via `--frames_per_batch` argument |

## 2. Detailed Technical Comparison

### A. Model Initialization & Execution
*   **Training (`train.py`):**
    *   Uses `deepspeed.initialize()` to wrap the model.
    *   The model engine handles FP16/BF16 casting, gradient accumulation, and distributed communication.
    *   Execution is managed by `torchrun` which handles process spawning.
*   **Inference (`extract_embeddings.py`):**
    *   Manually instantiates the model using `get_model()` / `Algorithm`.
    *   Wraps the model in a lightweight `GenericAlgo` class.
    *   Moves the model to GPU usage `model.cuda()`.
    *   **Crucial Difference:** It does **not** load the DeepSpeed engine. It treats the model as a standard `nn.Module`.

### B. Checkpoint Compatibility (Critical)
There is a significant format mismatch between the output of training and the input of inference:

1.  **Training Output:**
    *   `train.py` saves checkpoints using `model_engine.save_checkpoint()`.
    *   This creates a directory structure (e.g., `logs/PROJECT/global_stepXYZ/`) containing partitioned weights (`mp_rank_00_model_states.pt`, etc.) specific to the ZeRO stage used.
    *   It also saves a HuggingFace format copy using `save_pretrained` for Qwen models.

2.  **Inference Input:**
    *   `extract_embeddings.py` calls `restore_ckpt` in `utils.py`.
    *   `restore_ckpt` looks for specific files in this order:
        1.  `checkpoint.pth.tar` (Standard PyTorch checkpoint dict)
        2.  `fp32_converted/pytorch_model.bin` (Consolidated FP32 weights)
    *   **The Mismatch:** `extract_embeddings.py` **cannot** directly read the raw DeepSpeed checkpoints produced by `train.py`.

**Action Required:** You must run a conversion script (e.g., `zero_to_fp32.py` or the provided `scripts/convert_ds_to_hf.sh`) to consolidate the DeepSpeed checkpoint into the `fp32_converted/pytorch_model.bin` format expected by the inference script.

### C. Data Loading
*   **Training:**
    *   Calls `create_dataset(..., distributed=True)`.
    *   Uses specific samplers to ensure each GPU gets a unique slice of the data.
    *   Processes data in continuous iterations (`max_iters`).
*   **Inference:**
    *   Calls `create_one_epoch_dataset(..., mode='eval')`.
    *   Iterates through the dataset once ensuring all frames/videos are processed.
    *   Focuses on extracting embeddings and saving them to disk (`.npy` files).

## 3. Configuration & Arguments

| Parameter | `run_ds.sh` | `run.sh` |
| :--- | :--- | :--- |
| **Network** | `Qwen3-VL-2B` | `Qwen3-VL-2B` |
| **Video Paths** | Comma-separated list of multiple datasets | Single dataset path: `/open_data/.../video_paths.json` |
| **DeepSpeed Config** | `scripts/ds_config_zero3.json` | None (Ignored) |
| **Logging** | Auto-links logs to `/mnt/data/checkpoint` | Searches for latest log directory in `./logs` |

## 4. Deep Dive: Alignment Verification

You asked for a detailed check of whether key steps in the process are aligned. Here is the component-by-component analysis:

### 1. Model Structure & Forward Pass (✅ Aligned)
*   **Training**: Uses `Algorithm` class which wraps a `BaseModel` (containing Qwen).
*   **Inference**: Uses `extract_embeddings.py` -> `GenericAlgo` -> `BaseModel`.
*   **Critical Link**: Both workflows rely on `utils.py:get_cnn_feats()` for the actual forward pass.
    *   **Training**: `Algorithm.train_one_iter` -> `self.forward` -> `get_cnn_feats`.
    *   **Inference**: `get_embeddings_dataset` (in `utils.py`) manually calls `get_cnn_feats(model.cnn, ...)` to extract features.
    *   **Conclusion**: Since both funnel through `get_cnn_feats`, the core feature extraction logic is consistent, provided the model weights are loaded correctly.

### 2. Dustbin Mechanism & Attention Isolation (❌ Misaligned)
*   **Training (`LiberoDataset`)**: 
    - Randomly prepends a "Dustbin" frame (black image) to the reference sequence.
    - Sets `has_dustbin=True` metadata.
    - **Attention Effect**: The monkey-patched Qwen forward pass detects `has_dustbin=True` and applies `isolate_first_group=True`, which blocks the Dustbin frame from attending to the task abstraction prefix.
*   **Inference (`LiberoVideoDataset`)**:
    - Currently **does not** prepend a Dustbin frame to the candidate/reference sequence.
    - `has_dustbin` defaults to `False`.
    - **Effect**: The model uses standard attention for the first frame of the sequence, which differs from training.
*   **Action Required**: `LiberoVideoDataset` must be updated to include a Dustbin frame and trigger the same attention isolation logic to maintain feature consistency.

### 3. Data Preprocessing & Qwen Input (⚠️ Complex but Aligned)
This is the most complex part but appears designed to work:
*   **Training (`LiberoDataset`)**: returns a small set of "steps" (randomly sampled windows) from a video. `TCCCollator` processes these independent windows.
*   **Inference (`LiberoVideoDataset`)**: returns an *entire video* as a single sample (dense extraction).
    *   **Logic Check**: `LiberoVideoDataset` (inference) reimplements the exact same logic for context window creation (`_get_context_steps` equivalents) and path collection as `LiberoDataset` (training).
    *   **Instruction Handling**: Both classes correctly read `instruction.txt` and pass it to the Qwen processor.
    *   **When `TCCCollator` runs on this single "video sample", it effectively converts the whole video into a huge batch of windows (length = Total_Frames).
    *   `utils.py` then *manually checks* for `qwen_input` and slices it (lines 360+) to match the mini-batches it sends to the GPU.
    *   **Alignment Check**: The slicing logic in `utils.py` (`start_idx:end_idx`) perfectly mirrors the window-based flattening in `TCCCollator`.
    *   **Note**: This relies on `extract_embeddings.py` using the exact same `TCCCollator` as training, which is confirmed via `datasets.py` structure.

### 3. Checkpoint Loading (❌ Misaligned)
*   **Training**: Produces DeepSpeed sharded checkpoints (`mp_rank_00...`).
*   **Inference**: `utils.py:restore_ckpt` expects a standard PyTorch `state_dict` (`checkpoint.pth.tar` or `fp32_converted/pytorch_model.bin`).
*   **Analysis**: `extract_embeddings.py` has **no mechanism** to read DeepSpeed shards. If you run it strictly after training without conversion, it will likely fail to load weights or (worse) initialize a random model if it doesn't find the file, leading to garbage embeddings.

### 4. Environment & Distributed Context (⚠️ Gap)
*   **Training**: Uses `deepspeed` launcher which injects environment variables (`LOCAL_RANK`, `WORLD_SIZE`).
*   **Inference**: Runs as a standard python script.
    *   This is generally fine, but `models.py` imports `torch.distributed`. If `Qwen3VLForConditionalGeneration` or other components expect distributed initialization (which they shouldn't in inference mode), it might break.
    *   Generally, HuggingFace models handle non-distributed inference gracefully.

## 5. Recommendations
To ensure smooth transition from training to inference:

1.  **Add Conversion Step:** Insert a checkpoint conversion step in your workflow between `run_ds.sh` and `run.sh`.
    ```bash
    # Example
    python deepspeed_ckpt_to_fp32.py --input_folder logs/EXP/global_stepXYZ --output_file logs/EXP/fp32_converted/pytorch_model.bin
    ```
2.  **Verify Model Loading:** Ensure `extract_embeddings.py` correctly handles the Qwen architecture when loading the consolidated weights, as `Algorithm` wrapper expects specific keys (`cnn`, `emb`).
