# Project Context: TCC Qwen3-VL Alignment (EMA)

## Goal
Unsupervised video temporal alignment via cycle-consistency (TCC) loss, using Qwen3-VL-2B as the visual backbone. The model learns to align pairs of robot manipulation videos by finding soft nearest-neighbour correspondences in embedding space.

## Architecture
```
data (paired main/ref videos)
  → TCCCollator (augmentation + Qwen processor)
  → BaseModel (Qwen3-VL-2B backbone)       ← EMA'd by teacher_cnn
  → LinearEmbedder (1536 → 128)             ← EMA'd by teacher_emb
  → compute_deterministic_alignment_loss_paired()
       ├── student sim_matrix  → cycle-consistency loss + DTW regression loss
       └── teacher sim_matrix  → DTW index extraction (stable targets)
```

## Key Components
| File | Role |
|---|---|
| `train.py` | Main training loop with DeepSpeed ZeRO-3 |
| `models.py` | `BaseModel` (Qwen3-VL) + `LinearEmbedder` |
| `algos/algorithm.py` | `Algorithm` base: forward pass, teacher EMA |
| `algos/alignment.py` | `Alignment`: computes TCC loss |
| `tcc/deterministic_alignment.py` | All loss math: cycle-consistency, DTW guidance |
| `datasets.py` | `TCCCollator`: augmentation, Qwen processor packing |
| `config.py` | All hyperparameters |

## Student-Teacher EMA Design
- **Teacher** = EMA copy of student (`teacher_cnn` + `teacher_emb`), frozen (`requires_grad=False`)
- **Teacher forward**: runs on clean (no-aug) data via `qwen_input_teacher` from `TCCCollator`
- **EMA update**: called after each `model_engine.step()`; operates on local ZeRO-3 parameter shards directly (no gather needed — EMA is element-wise)
- **DTW targets**: `extract_alignment_indices_from_sim_matrix_dtw` uses teacher sim_matrix when `USE_TEACHER_FOR_DTW=True`; cycle-consistency loss always uses student embs

## Config Flags (config.py)
| Flag | Default | Meaning |
|---|---|---|
| `CONFIG.ALIGNMENT.EMA_DECAY` | 0.9995 | EMA momentum |
| `CONFIG.ALIGNMENT.USE_TEACHER_FOR_DTW` | True | Teacher sim_matrix → DTW indices |
| `CONFIG.ALIGNMENT.EMA_BACKBONE` | True | Also EMA the Qwen backbone (2x forward compute) |
| `CONFIG.ALIGNMENT.USE_DTW` | True | Enable DTW guidance loss |
| `CONFIG.ALIGNMENT.DTW_GUIDANCE_LAMBDA` | 0.5 | Weight of DTW guidance loss |
| `CONFIG.ALIGNMENT.SAVE_RAW_SIM12` | False | Rank 0 only: save scaled pre-softmax `sim_12` (MR/RM) to disk for debugging |
| `CONFIG.ALIGNMENT.SAVE_RAW_SIM12_DIR` | `""` | Output directory when `SAVE_RAW_SIM12=True` |
| `CONFIG.ALIGNMENT.SAVE_RAW_SIM12_EVERY` | 1000 | Save when `global_step % EVERY == 0` (training only) |

## Tech Stack
- PyTorch + DeepSpeed (ZeRO-3 stage-3)
- Qwen3-VL-2B (`/mnt/data/checkpoint/ethanchen/Qwen3/Qwen3-VL-Embedding-8B`)
- Datasets: BerkeleyUR5, CMU Play Fusion, BridgeV2, DROID, ManiSkill, CALVIN, FMB, RT-1, LIBERO

---

## Changelog

### 2026-04-11 — Joint mapping: longest prefix + dim guard

**What changed:** `datasets.py` and `datasets_sq.py`: `_resolve_joint_mapping` multi-key lookup now picks the **longest** matching path prefix (so `.../Franka-sim` uses the `Franka-sim` entry, not `.../Franka`). `_load_joint_data` checks `raw_joints.shape[1] == nm.shape[0]` before normalize and logs a clear warning if not.

**Why:** `sub_data_path.startswith(".../Franka")` is true for `.../Franka-sim`, so the first-hit Franka (18-D) mapping was applied to sim JSON (16-D), causing a numpy broadcast error during joint normalization.

### 2026-04-05 — Cache-aware image loading (reduce dataloader spike)

**What changed:**

- `datasets.py`: Added `_prepare_images` method (extracts image loading from `_build_qwen_input`). `_build_qwen_input` gains optional `prepared` param so images are loaded once and reused for both paired and main_only sinks. `TCCCollator.__init__` adds `self._ref_cache_index = None`. `__call__` checks `_ref_cache_index` (Manager dict) before loading: if all refs cached → load main only + build main_only only; else → load main+ref once, reuse `prepared` for both paired and main_only sinks. Added `import zlib`.

- `algos/algorithm.py`: `RefEmbeddingCache` gains `index` param (`Manager().dict()`). `put()` writes key to `index`; eviction deletes from `index`. `qwen_meta` fallback handles `qwen_input_paired=None` (cache-hit path).

- `evaluate_v2.py`: Creates `Manager().dict()` as shared ref-cache index; injects into `_ref_cache.index` and `collate_fn._ref_cache_index` before the eval loop.

**Why:** Dataloader was calling `_build_qwen_input` twice (paired + main_only), loading main images twice from disk on every batch regardless of cache state. Cache hit: now only loads main (skips ref entirely). Cache miss: loads main+ref once, reuses for both paired and main_only. This eliminates the periodic 40s spikes caused by dataloader becoming the bottleneck.

**What changed:**

- `algos/algorithm.py`: Added `RefEmbeddingCache` (LRU, maxsize=256) — stores sample-level ref embeddings `(M*NUM_STEPS, D)` keyed by `tuple(ref_frame_paths)`. `Algorithm.__init__` initializes `self._ref_cache`. `forward()` checks cache at eval time: all refs hit → use main_only path (skip ref Qwen forward, concat cached embs); miss → paired path (current logic, populate cache after merge).

- `datasets.py`: `TCCCollator.__call__` now simultaneously builds `qwen_input_paired` (main+ref) and `qwen_input_main_only` (main only, ref Pass 2 skipped). Both are stored in `collated_batch`.

- `config.py`: Added `CONFIG.EVAL.REF_CACHE_MAXSIZE = 256` and `CONFIG.EVAL.REF_CACHE_MAIN_ONLY = True`.

- `evaluate_v2.py`: Added cache hit rate logging at eval epoch end.

**Why:** eval batch_size=1 with M=4 means the same ref video is processed by Qwen M times. The cache avoids this by storing each sample's merged ref embedding and reusing it on subsequent appearances. TCCCollator cannot access the cache (model side), so both paired and main_only batches are pre-built; the model decides at runtime based on cache hit/miss.

**Verification:** warmup cache → all hits → Mode B equivalent to paired; first encounter miss → auto-fallback to paired, identical results.

### 2026-02-20 — Student-Teacher EMA for Stable DTW Targets
**What changed:**
- `datasets.py`: `TCCCollator` now generates a `qwen_input_teacher` (clean, no-aug) alongside the student's augmented `qwen_input` when `EMA_BACKBONE=True`. Implemented by adding `pixel_sink` + `update_metadata` params to `process_packed_sequence` to avoid duplicating metadata accumulators.
- `config.py`: Added `EMA_DECAY=0.9995`, `USE_TEACHER_FOR_DTW=True`, `EMA_BACKBONE=True`.
- `algos/algorithm.py`: `Algorithm.__init__` now deepcopies `cnn` → `teacher_cnn` and `emb` → `teacher_emb` (both frozen). `forward()` runs a `torch.no_grad()` teacher pass and threads `teacher_embs` through the chunk-merging block. Added `update_teacher_ema()` method.
- `tcc/deterministic_alignment.py`: `compute_deterministic_alignment_loss_paired()` accepts `teacher_embs_main/ref`; computes teacher sim matrices and uses them as input to `extract_alignment_indices_from_sim_matrix_dtw` instead of the student's raw sim matrix.
- `tcc/alignment.py` + `algos/alignment.py`: Threaded `teacher_embs` parameter through the call chain.
- `train.py`: Calls `model_engine.module.update_teacher_ema()` after each optimizer step.

**Why:** The student's sim_matrix fluctuates during training, creating noisy DTW alignment targets. Using a slow-moving EMA teacher produces more stable DTW indices, improving the quality of the regression supervision signal.

### 2026-02-20 — Fix: EMA RuntimeError with ZeRO-3 (size mismatch at dim 4)

**Error:** `RuntimeError: The size of tensor a (0) must match the size of tensor b (16) at non-singleton dimension 4` in `update_teacher_ema()`.

**Root Cause:** ZeRO-3 concatenates ALL model parameters (student + teacher) into one flat buffer and slices it across ranks. `student_param_i` and `teacher_param_i` occupy different positions in the buffer, so their local `.data` shards cover different element ranges and may have size 0 on some ranks. Direct shard-level EMA fails with a shape mismatch.

**Attempted Fix:** Used `deepspeed.zero.GatheredParameters([t_p, s_p], modifier_rank=0)` per parameter pair. Led to a second error (see below).

---

### 2026-02-20 — Fix: EMA AssertionError with ZeRO-3 + gradient checkpointing (`ds_active_sub_modules`)

**Error:** `AssertionError: {'active_sub_modules': {5, 6}, ...}` in `free_param()` inside `GatheredParameters.__exit__`.

**Root Cause:** `GatheredParameters.__exit__` always calls `partition() → _partition_param() → free_param()`, which asserts `param.ds_active_sub_modules` is empty. Qwen's visual encoder has `gradient_checkpointing_enable()` active (`models.py:160`). ZeRO-3 pre/post-forward hooks increment `ds_active_sub_modules` during the gradient-checkpointing recompute phase; due to a DeepSpeed interaction they are not fully cleared after `model_engine.step()`. Both `modifier_rank=0` and `modifier_rank=None` trigger this path — there is no safe way to call `GatheredParameters` on these params after the backward pass.

**Fix:** Switched from ZeRO-3 to **ZeRO-2** (`scripts/ds_config_zero2.json`).
- ZeRO-2 shards optimizer states and gradients only; **parameters are replicated** on every rank.
- Teacher and student params are both full tensors → EMA is trivially applied with `torch._foreach_mul_` + `torch._foreach_add_`, identical to the DINOv3 pattern.
- Removed all `GatheredParameters` / `import deepspeed` from `update_teacher_ema()`.

**Files changed:**
- `scripts/run_ds.sh`: `DS_CONFIG` changed from `ds_config_zero3.json` → `ds_config_zero2.json`.
- `algos/algorithm.py`: `update_teacher_ema()` simplified to `torch._foreach_mul_` + `torch._foreach_add_`; comment updated to reflect ZeRO-2 assumption.

---

### 2026-02-21 — Teacher Forward-Order Frames + TCCCollator Refactor

**What changed:**

**`datasets.py`** (rebuilt from `datasets_ori.py` with clean structure):
- `LiberoDataset._get_item_impl`: Saves `main_steps_teacher` / `ref_steps_teacher` (copies of pre-reversal steps). Computes `teacher_frame_paths` / `teacher_ref_frame_paths` via `_context_steps_to_paths` + `_get_context_steps(..., reverse=False)`. Returns `do_reverse_main`, `do_reverse_ref`, and teacher paths in the data dict.
- `LiberoDataset._context_steps_to_paths`: New helper — converts flat context-step indices to a list of file paths (or numpy frames), mapping `-1` → `"DUSTBIN"`.
- `LiberoDataset.__init__`: Added `self.create_teacher_view` flag (mirrors collator).
- `TCCCollator` refactored into 5 clean methods + slim `__call__`:
  - `_apply_m_chunking(batch)`: selects M, chunk indices, cut params; propagates `teacher_frame_paths` with the **same** `chunk_indices` as student (key invariant for temporal alignment).
  - `_prepare_per_sample(batch)`: applies cut to align paths, computes `_cached_align_paths`, `_is_cut_mask`, `_has_dustbin` once — shared by student and teacher.
  - `_pack_sequence(sink, ..., update_metadata)`: class method replacing the old closure; writes pixel data always, metadata only when `update_metadata=True`.
  - `_build_qwen_input(batch, ..., frame_paths_key, do_augment, update_metadata)`: unified pipeline — load from the given key, optionally augment, run Pass 1 + Pass 2, write to a `sink` dict. Student calls with `frame_paths_key='frame_paths'` + `do_augment=True`; teacher calls with `'teacher_frame_paths'` + `do_augment=False`.
  - `_assemble_qwen_dict(sink, shared_metadata)`: pads/cats pixel tensors; when `shared_metadata` is provided, reuses all metadata tensors from student (teacher has no separate seq_lens/num_mains/etc.).
  - `__call__`: 4-line orchestration — M-chunk → prepare → student pass → teacher pass.

**`tcc/deterministic_alignment.py`:**
- Added `_dtw_fwd_to_student(dtw_indices, do_reverse_src, do_reverse_tgt, T_tgt)`: remaps forward-space DTW indices to student (possibly reversed) coordinates by flipping the source axis and/or remapping target indices.
- DTW call site: when teacher embs are used (`USE_TEACHER_FOR_DTW=True`), passes `direction=+1` (all-ones) to `extract_alignment_indices_from_sim_matrix_dtw` (teacher sim matrix is always forward×forward), then calls `_dtw_fwd_to_student` to convert MR and RM indices back to student coordinate space before using them as DTW guidance regression targets.

**Why:** The teacher EMA was previously processing frames in the same (possibly reversed) order as the student, undermining the stability benefit of EMA. Now the teacher always sees natural forward-order frames, producing consistent per-frame representations. The DTW guidance targets are remapped back to the student's coordinate system, so the regression supervision signal is both stable (teacher EMA) and correctly oriented (student frame ordering).

---

### 2026-03-13 — Fix: Gradient Accumulation Override Bug + EMA Micro-Step Bug

**What changed:**

**`train.py`:**
- **Override condition fix**: Changed `if FLAGS.gradient_accumulation_steps > 1:` → `if FLAGS.gradient_accumulation_steps != ds_config.get('gradient_accumulation_steps', 1):`. Previously, when `run_ds_all.sh` computed `GRAD_ACCUM_STEPS=1` (e.g. WORLD_SIZE=4), the override was skipped and `ds_config_zero2.json`'s hardcoded `"gradient_accumulation_steps": 4` stayed in effect — accumulation ran at 4× instead of the intended 1×.
- **EMA boundary guard**: Wrapped `update_teacher_ema()` in `if model_engine.is_gradient_accumulation_boundary():`. Previously, EMA was applied on every micro-step even though student weights only change at the accumulation boundary. This caused the effective EMA decay per real optimizer step to be `0.9995^grad_accum ≈ 0.998` rather than `0.9995`, making the teacher track the student ~4× faster than intended.

**Why:** Both bugs silently degraded training correctness when gradient accumulation was active. The override bug meant effective batch size was always 4× micro-batch × WORLD_SIZE regardless of node count. The EMA bug eroded the slow-moving teacher's stability advantage — the core reason for having a teacher in the first place.

### 2026-03-29 — LiberoDataset `num_steps` tiers from `NUM_FRAMES`

**What changed:** `LiberoDataset._get_item_impl` no longer hardcodes 96/48/24 for `num_steps`. It uses `4×`, `2×`, and `1×` `CONFIG.TRAIN.NUM_FRAMES` (`steps_4x` / `steps_2x` / `steps_1x`), matching `TCCCollator` M-chunk multipliers. `MAX_BATCH_FRAMES` default when unset is `steps_4x` instead of 96. Same logic applied in `datasets_ori.py` for parity.

**Why:** Hardcoded values assumed `NUM_FRAMES=24`; with `NUM_FRAMES=16` and `MAX_BATCH_FRAMES=64`, sampling should stay aligned with training chunk length and batch frame caps.

### 2026-04-01 — Training data augmentation enabled (`CONFIG.AUGMENTATION`)

**What changed:** `config.py` defaults: `RANDOM_FLIP=True`, `BRIGHTNESS=True`, `CONTRAST=True`, `REVERSE_PROB=0.5`; `RANDOM_CROP` remains `False`; hue/saturation remain off.

**Why:** Student-side `TCCCollator` augmentation was previously all-off; enabling standard photometric + flip + temporal reverse improves robustness. Teacher path stays no-aug (`do_augment=False`). Restart training after changing these flags.

### 2026-04-02 — Optional save of scaled pre-softmax `sim_12` (MR / RM)

**What changed:** `config.py` adds `SAVE_RAW_SIM12`, `SAVE_RAW_SIM12_DIR`, `SAVE_RAW_SIM12_EVERY`, `SAVE_RAW_SIM12_MAX_BATCH`. `tcc/deterministic_alignment.py` adds `_maybe_save_raw_sim12` (writes `sim12_mr_step{step}_{timestamp}.pt` and `sim12_rm_step{step}_{timestamp}.pt`; payloads include `sim12` tensors and metadata; with dustbin, MR payload may include `sim12_real_frames_only`). Only **global rank 0** performs I/O. `compute_deterministic_alignment_loss_paired` takes `global_step` and `training`; `compute_alignment_loss` and `Alignment.compute_loss` thread them through.

**Why:** Persist `get_scaled_similarity` outputs separately from softmax/smooth-DTW β for analysis; filenames distinguish MR vs RM and step vs wall-clock time.

When a step saves `.pt` files, `loss_dict` carries `raw_sim12_mr_path` / `raw_sim12_rm_path` (abs paths); `log_and_save_high_loss_samples` copies them into each jsonl line plus `raw_sim12_batch_index` to slice `sim12[idx]` in the batch tensor.

### 2026-04-02 — LiberoDataset: normalize `cam_list` to exactly 3 views

**What changed:** When JSON/cam_mapping provides `cam_list`, `LiberoDataset._get_item_impl` forces **exactly three** camera names: `len < 3` → cycle-pad (e.g. `[A]` → `[A,A,A]`, `[A,B]` → `[A,B,A]`); `len > 3` → keep first three (`[:3]`); `len == 0` → `IgnoreSample`. `logging.warning` on pad/truncate. Then `num_views == 3` and `num_ctx = NUM_STEPS // 3`. JSON path uses `assert cam_list_ld is not None` (`self.pickle_data is None`). Pickle omits `cam_list` (`num_views` stays 1).

**Why:** Fixed 3-view batch geometry while keeping primary camera order for path substitution.

### 2026-04-06 — `convert_ds_to_hf.sh`: optional HF conversion (step 3)

**What changed:** `scripts/convert_ds_to_hf.sh` runs DeepSpeed `zero_to_fp32` (steps 1–2) by default. Step 3 (`convert_ckpt_to_hf.py`) runs only when `CONVERT_HF=1`. `hf_converted/` is created only in that case.

**Why:** Skip slow HF export when only `fp32_converted/pytorch_model.bin` is needed (e.g. `scripts/run.sh`).
