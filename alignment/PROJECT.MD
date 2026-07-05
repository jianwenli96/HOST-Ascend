# Project Overview: Optimized TCC Training with Qwen3-VL

This document serves as a comprehensive reference for the optimization logic implemented in this project. The primary goal is to accelerate the training of Time-Contrastive Networks (TCC) using the Qwen3-VL-2B model by leveraging **Packed Sequences with Shared Prefixes**.

## 1. Core Optimization Concept

In the standard TCC training pipeline, for a batch size $B$ and sequence length $T$, the model performs $B \times T$ forward passes. Each pass typically involves encoding a long "Align Video" (the video being aligned against) plus a single frame (or small chunk) from the source video.

**Problem**: The "Align Video" is identical for all $T$ steps of a single sample, leading to massive redundant computation.

**Solution**: We pack the Main sequence and Reference sequence separately per sample, using the *other* video's abstraction as a shared prefix.
- **Old Optimized**: $B$ Forward Passes of `[Align_Video_Ref + Main_Groups + Ref_Groups]`
- **New Symmetric Optimized**: $2B$ Forward Passes: 
    1. `[Align_Video_Ref + Main_Groups]`
    2. `[Align_Video_Main + Ref_Groups]`

This maximizes cross-video attention efficiency. By splitting into two sequences, we reduce individual sequence length $S$, which improves memory efficiency due to the $O(S^2)$ attention complexity, while still encoding all required alignment pairs.

## 2. Architecture & Data Flow

### 2.1. Packed Sequence Structure
For a single sample in the batch, two input token sequences are constructed. Each group (chunk) is terminated by a specialized **Summary Token** (`<|file_sep|>`, ID: 151664) which acts as a learnable CLS token for that chunk.

**Sequence 1 (Main):**
```text
[ Shared Prefix ] + [ Main Group 0 ] + [ CLS ] + ... + [ Main Group K ] + [ CLS ]
```

**Sequence 2 (Ref):**
```text
[ Shared Prefix ] + [ Ref Group 0 ] + [ CLS ] + ... + [ Ref Group K ] + [ CLS ]
```

- **Shared Prefix**: Contains the tokens for the full alignment video (usually 10 sampled frames).
- **Groups**: Contain the context frames for timestep $k$ (usually 1 frame + 1 context).
- **CLS Token**: Appended after each group. It is the only token extracted for the final embedding and for AttentionGate computations.

### 2.2. Attention Masking (Block-Diagonal)
To mathematically simulate independent forward passes, we modify the Attention Mask.
- **Shared Prefix**: Visible to ALL groups.
- **Group $k$**: Visible to itself.
- **Isolation**: Group $k$ **CANNOT** see Group $j$ (where $j \neq k$).

This ensures that the representation of Frame $T$ is not influenced by Frame $T+1$ (future leakage) or Frame $T-1$ (unless explicitly desired in chunking).

### 2.3. Position ID Reset
To ensure the model processes each group as if it immediately follows the prefix:
- The Position IDs for **Group 0** continue linearly after the Prefix.
- The Position IDs for **Group $k$** are **reset** to match Group 0.
- Example: `Prefix(0-100) -> Grp0(101-110) -> Grp1(101-110) ...`

## 3. Implementation Details

### 3.1. Data Pipeline (`datasets.py`)
The `AlignmentCollator` class handles the complex batch construction:

1.  **Chunking**: Slices the `frame_paths` into chunks of size `CONFIG.DATA.NUM_STEPS`.
2.  **Packing**: Constructs the Qwen input. After each chunk's images, it appends the `self.cls_token_name` (`<|file_sep|>`).
3.  **Metadata Flattening**:
    - The collator flattens `seq_lens` into a list `[L_{m0}, ..., L_{mn}, L_{r0}, ..., L_{rn}]`.
    - It passes `cls_token_id` to the model to facilitate vectorized token extraction.

### 3.2. Model Forward (`models.py`)
The `BaseModel` handles extraction using the CLS tokens:

1.  **Input Identification**: Detects `num_mains` and `num_refs` to switch to Packed Mode.
2.  **CLS-Based Extraction**:
    - Locates all indices where `input_ids == cls_token_id`.
    - Extracts `hidden_states` at these exact indices. This replaces the old mean-pooling logic, allowing the model to learn a task-specific summary of the chunk.
3.  **Attention Gate Integration**: 
    - The same extracted CLS embeddings are used as tokens for the **AttentionGate**. 
    - This reduces the gate's input from $P \approx 196$ patches down to 1 token per chunk, significantly improving throughput.
4.  **Reordering**: Matches legacy formats expected by the loss functions.
4.  **Dimension Compatibility**:
    - `LinearEmbedder` (for Qwen) supports both 2D `(Total_Steps, H)` and 5D `(B, T, C, H, W)` inputs, ensuring compatibility with both the new packed logic and legacy ResNet paths.

### 3.3. Low-Level Patching (`monkey_patch_forward.py`)
This file intercepts the `forward` call of `Qwen3VLModel`.

- **`_apply_group_attention_mask`**: Constructs the 4D mask to implement block-diagonal logic. It now uses the **last CLS token** index to define the `content_end`, ensuring prompt/template tokens are ignored.
- **`_apply_group_position_ids`**: Implements the position ID reset at each `vision_start`.
- **Trigger**: The logic is triggered automatically when packed metadata is detected.

### 3.4. AttentionGate (Temporal Cross-Attention)
The `AttentionGate` is a transformer-based cross-attention module that allows the "Main" video chunks to attend to the "Reference" video's alignment frames.

1.  **Old Behavior**: The gate received all image patches from a chunk ($P \times D$, where $P \sim 196$ for Qwen3-VL). This was computationally expensive ($O(T_{main} \times T_{ref} \times P^2)$).
2.  **Optimized Behavior**: It now only receives the single **CLS token embedding** per chunk.
    - **Complexity**: Reduced significantly, enabling faster convergence and lower memory usage.
    - **Contextualization**: The CLS token represents the entire chunk, allowing the cross-attention to focus on high-level temporal alignment rather than per-patch correlation.

## 4. Configuration

Key parameters in `config.py` that affect this logic:

- **`CONFIG.DATA.NUM_STEPS`**: The size of each "Group" or "Chunk".
    - Usually set to 2 (Current Frame + 1 Context Frame).
    - Determines the granularity of the packing.
- **`CONFIG.TRAIN.NUM_FRAMES`**: The number of logical frames sampled per video.
    - The Collator will attempt to pack `NUM_FRAMES` chunks into one sequence.
- **`CONFIG.MODEL.BASE_MODEL.NETWORK`**: Must contain 'Qwen' to trigger the specific embedding logic.

## 6. Long Video Support & Multi-Multiplier Strategy

To support training on long video sequences (up to 96 frames) without exceeding memory limits, we implemented a **Chunk-Multiplier Strategy**.

### 6.1. Multi-Multiplier Collation (`datasets.py`)
The `AlignmentCollator` now supports variable multipliers $M \in \{1, 2, 4\}$:

- **Resampling**: The dataset samples up to 96 frames initially. The collator then performs **uniform resampling** (via `np.linspace`) to reach the target length $M \times 24$.
- **Chunking**: The resampled sequence is split into $M$ chunks, each containing exactly 24 steps ($chunk\_len$).
- **Batching**: These chunks are treated as independent samples within the DeepSpeed batch but carry `group_id` and `chunk_id` metadata.

### 6.2. Video Reconstruction and Loss (`tcc/alignment.py`)
Since TCC loss requires comparing full temporal sequences, we perform **Embedding Reconstruction** within the loss function:
- **Grouping**: Embeddings belonging to the same `group_id` are identified.
- **Sorting**: Chunks are sorted by `chunk_id` to restore temporal order.
- **Concatenation**: Chunks are concatenated along the time dimension to form a single continuous sequence of length 24, 48, or 96.
- **Metadata Merging**: `frame_paths` are similarly merged to allow for continuous video logging.

### 6.3. Observability & Visualization
- **GIF Reconstruction**: `utils.py` recognizes `merged_metadata` in the loss output. When a high-loss sample is logged, it uses the merged paths to generate a GIF representing the **full long video** (e.g., all 96 frames) instead of a single 24-frame chunk.

## 7. Temporal Reversal Augmentation

To improve model robustness and data variety, we implemented a physically consistent temporal reversal augmentation. This allows the model to learn from both forward and backward representations of the same task.

### 7.1. Independent Reversal
- **Independent Logic**: Each sample's `main` sequence and `ref` sequence independently roll for reversal based on `CONFIG.AUGMENTATION.REVERSE_PROB` (default 0.5).
- **Combination Diversity**: This creates four possible scenarios per sample pair: (Forward, Forward), (Reverse, Forward), (Forward, Reverse), and (Reverse, Reverse).

### 7.2. Physically Consistent Context (`datasets.py`)
A simple sequence reversal of frames would make "past" context frames appear like they are in the "future" relative to the current frame. To maintain physical consistency, we modified the sampling logic:
- **`_get_context_steps`**: Now supports a `reverse` flag.
    - **Forward Mode**: Context window is sampled as `[step - (N-1)*stride, ..., step]`.
    - **Reverse Mode**: Context window is sampled as `[step + (N-1)*stride, ..., step]`.
- **Logic**: In both modes, the "label" frame (`chosen_steps`) is the **last** frame of the window. In a reversed video, an index that was originally "later" (`t+1`) acts as the "physical past" for a frame at index `t`.

### 7.3. Prefix Stability
- **Non-Reversed Abstractions**: The Shared Prefix (Video Abstraction) frames used for alignment grounding are **never reversed**.
- **Reasoning**: This ensures that even when the model is processing a "backward" action, it is aligning it against a stable, forward-moving canonical reference of the task.

### 7.4. Label Consistency
- **Simultaneous Reversal**: Both the image paths and their corresponding temporal timestamps (`chosen_steps`) are reversed together.
- **TCC Grounding**: The TCC loss regression remains valid because $t_{reversed}(i)$ correctly maps to the content of $Frame_{reversed}(i)$.

## 8. Dustbin Alignment Mechanism & Cut Logic

To handle out-of-distribution frames and "cut" scenarios where no alignment exists, we implemented a robust **Dustbin Mechanism**.

### 8.1. Data Pipeline Modifications (`datasets.py`)
- **Block Cuts**: The `AlignmentCollator` randomly selects a contiguous block of indices (10% to 50% of total length) to remove from the reference video, simulating temporal gaps.
- **Dustbin Chunk**: A specific "Dustbin" chunk (black images, index 0) is prepended to the reference sequence.
- **Is-Cut Mask**: A mask is maintained tracking which reference time-steps correspond to the cut video segments.

### 8.2. Dual-Softmax Loss Logic (`tcc/deterministic_alignment.py`)
The alignment loss (`Main -> Ref`) uses a specialized dual-softmax approach:
1.  **First Softmax (Full)**: Computes attention over `[Dustbin, Ref_Frame_1, ..., Ref_Frame_N]`. 
    - The probability `P(Dustbin)` is extracted here.
    - Loss: Binary Cross Entropy between `P(Dustbin)` and the ground truth "is_cut" likelihood (derived from real frames).
2.  **Second Softmax (Real-Only)**: Excludes the dustbin and normalizes over `[Ref_Frame_1, ..., Ref_Frame_N]`.
    - This distribution defines the "Cut Target": `sum_cut_prob = sum(P_real[i] * is_cut[i])`.
    - Interpretation: "How much does the model think this query belongs to the cut frames, relative to other real frames?"
3.  **Global Constraints**:
    - **Per-Sample Weighting**: Cycle consistency loss is scaled by `(1 - P(Dustbin))`, down-weighting alignment for frames the model deems "garbage".
    - **Global Penalty**: A batch-level penalty (`GLOBAL_DUSTBIN_LOSS_WEIGHT`) ensures the total mass of `P(Dustbin)` roughly matches the known cut ratio of the batch, within a tolerance (`GLOBAL_DUSTBIN_LOSS_TOLERANCE`).

### 8.3. Visual Debugging
To verify the mechanism works:
- **Mask Propagation**: `utils.py` extracts `is_cut_mask` from the loss dictionary and saves it into `high_loss_samples.jsonl`.
- **Visualization**: A GIF generator can render a **Red Border** around frames in the reference sequence that are marked as "cut", appending `[CUT]` to the frame title, to visually confirm the model correctly assigns "Cut" frames to the Dustbin (or heavily penalizes mis-alignment).