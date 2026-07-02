# FastWAM

Video-conditioned World-Action Model for robot manipulation. Diffusion-based architecture that jointly generates future video frames and predicts actions, conditioned on task demonstration video and/or text instructions.

## Architecture

```
FastWAM/
├── configs/
│   ├── train.yaml                    # Base training config (lr, eval steps, etc.)
│   ├── data/
│   │   ├── custom.yaml               # tv+txt (task_video_drop=0.5, text_drop=1.0)
│   │   ├── custom_text.yaml          # txt_only (task_video_drop=1.0, text_drop=0.0)
│   │   └── custom_cross.yaml         # cross_attn mode (task_video_drop=0.4, text_drop=1.0)
│   ├── model/
│   │   ├── fastwam_joint.yaml        # Joint video+action model (prepend conditioning)
│   │   └── fastwam_joint_cross_attn.yaml  # Cross-attention conditioning
│   └── task/                          # Per-experiment configs (override data+model)
├── src/fastwam/
│   ├── models/wan22/
│   │   ├── fastwam.py                # FastWAM BASE class — NOT used directly in prod (see fastwam_joint.py)
│   │   ├── fastwam_joint.py          # FastWAMJoint (prod class): action attends ALL video tokens
│   │   ├── mot.py                    # Mixture of Transformers (video + action experts)
│   │   ├── action_dit.py             # Action DiT expert
│   │   └── wan_video_dit.py          # Video DiT expert (flash_attention, cross-attention)
│   ├── datasets/custom/
│   │   └── mydatasets.py             # Emu3SFTDataset — JSON-based episode loading
│   ├── trainer.py                    # Training loop + eval
│   └── runtime.py                    # Hydra entry points (run_training, create_fastwam_joint)
├── eval/real_openloop/               # Open-loop evaluation (real robot)
│   ├── fastwam_eval.py               # Model wrapper (infer_real, task_video window management)
│   ├── eval_dataset.py               # Evaluation dataset (GT actions, frame paths, joints)
│   ├── evaluate_openloop.py          # Main eval script (chunk-based metrics)
│   ├── action_utils.py               # Denormalization, 6D→Euler conversion
│   └── tools/
│       └── joint_action_mapping_norms.py  # Normalization vector loading
└── scripts/
    ├── train_zero1_real.sh            # Training launch (ZeRO-1, cross-DLC support)
    ├── train_zero1_real_text.sh       # Text-only variant
    ├── train_zero1_real_cross_attn.sh # Cross-attention variant
    └── eval_openloop.sh              # Open-loop eval launch
```

## Key Concepts

### Task Video Conditioning Modes

Three training modes determined per-sample by dataset-side drop probabilities:
- **tv+txt**: task_video present + text context (full conditioning)
- **txt_only**: task_video dropped + text context only
- **tv_only**: task_video present + text context zeroed out

Drop decisions are made in `mydatasets.py:_get_single_item` (per-sample, not batch-level):
- `task_video_drop_prob`: probability to drop task_video → fixed action frame intervals
- `task_text_drop_prob`: probability to zero text context (only when task_video present)

When task_video is dropped: zero tensor padding (same shape as real task_video), `task_video_dropped=True` flag, attention mask blocks video→task_video for that sample.

### Task Video Conditioning Architecture

Two modes controlled by `task_video_conditioning_mode` in model config:
- **prepend**: task_video tokens prepended to self-attention sequence `[task_video | agent_video | action]`
- **cross_attn**: task_video tokens added to cross-attention context alongside text embeddings

### Action Tensor Format

`[L, action_dim + 2]` where:
- `[:, :action_dim]` — action values (20D with 6D rotation, or 14D Euler)
- `[:, -2]` — **progress**: task video keyframe position `[anchor_frame/max_frames, next_keyframe/max_frames]` linearly interpolated, normalized to [-1, 1]
- `[:, -1]` — **mask**: 1.0=valid, 0.0=padding (currently always 1.0 since uniform sampling)

Progress is used at eval time to advance the task video window: `predicted_progress → keyframe_idx → absolute_frame → next window center`.

### Frame Sampling

- `dataset_fps`: per-dataset FPS (e.g. 10042→32). Action frames resampled to this rate.
- `action_frames`: base interval between agent keyframes (overridden by dataset_fps per sample)
- `sampling_interval_min/max_mult`: random interval range when task_video present (0.5~2.0×)
- When task_video dropped: fixed interval = `action_frames`

### Normalization

- Actions: `normed = clip(2*(raw - norm_min) / norm_delta - 1, -1, 1)`
- Denorm: `raw = 0.5 * (normed + 1) * (norm_high - norm_low) + norm_low`
- 6D rotation fields use identity norm (min=-1, delta=2)
- Joint/proprio: same formula with separate `j_nm, j_nd`
- All norm params from `joint_action_mapping_dir/*_joint_action_mapping.json`

## Training

### Launch

```bash
bash scripts/train_zero1_real.sh task=real_joint_2cam_224_1e-4
```

### Training Eval (in-loop)

Triggered every `eval_every` steps. Single-pass inference using val dataset sample (respects drop probs). Mode auto-detected from `task_video_dropped` and `context_mask` state. Saves:
- `step_{step}_rank_{rank}_{mode}.mp4` — stitched video [task | pred | vae_recon | gt]
- Metrics logged to wandb: `{mode}/psnr_rg`, `{mode}/action_l1`, etc.
- Loss split: `with_task_video/loss` vs `without_task_video/loss` (only for pure batches, mixed batches excluded)

### Checkpoints

```
logs/{task_name}/{timestamp}/
├── config.yaml            # Full merged Hydra config (all params saved)
├── checkpoints/
│   ├── weights/step_004000.pt    # Model weights (MoT + proprio_encoder)
│   └── state/step_004000/        # Optimizer + scheduler + random state
├── eval/                          # Training eval outputs
└── wandb/
```

### Cross-DLC Multi-Node

Scripts support spanning multiple DLC jobs via env vars:
```bash
CROSS_JOB_ID=0 CROSS_JOB_TOTAL_NODES=8 bash scripts/train_zero1_real.sh
```

## Open-Loop Evaluation

### FastWAMEval Wrapper (`eval/real_openloop/fastwam_eval.py`)

Stateful wrapper for inference. All config read from checkpoint's `config.yaml`.

```python
wrapper = FastWAMEval(
    checkpoint_dir="./logs/real_joint_2cam_224_1e-4/2026-04-25_17-10-05",
    dataset_name="10042",
    views=["leftImg", "rightImg"],
    use_task_video=True,         # False → txt_only mode
    use_task_description=True,   # False (with task_video) → tv_only mode
)
wrapper.set_task_video(episode_dir)  # load once
result = wrapper.infer_real(
    observations={"leftImg": img_np, "rightImg": img_np},  # {view: [H,W,3] uint8}
    joints=joint_np,             # [D_joint] raw
    instruction="pick up the cup",
)
# result["action_raw"]: [max_action_len, D_euler] denormalized
# result["predicted_progress"]: float, task video keyframe position
```

### Task Video Window Management

1. `set_task_video(episode_dir)`: loads all frames (PIL) for all views
2. `_get_task_video_window(max_frames)`: samples keyframes using `frame_to_window_center`
3. Window advances using model-predicted progress: `progress → keyframe_idx → absolute_frame`
4. `_build_task_video_tensor`: PIL frames → `[C, T_task, H_total, W]` tensor in [-1,1]

### Eval Launch

```bash
bash scripts/eval_openloop.sh
# Or with overrides:
CHECKPOINT_DIR=./logs/... DIFFUSION_STEPS=20 VISUALIZE=true bash scripts/eval_openloop.sh
```

### Metrics

- Per-chunk: raw L1/L2, normed L1/L2
- Per-episode: MAE, RMSE, pos_err_m, rot_err_deg, grip_acc (14D dual-arm)
- Aggregate: mean across episodes, saved to JSON

### Visualization (`--visualize`)

Saves per-step stitched MP4 `[task_video(red border on aligned) | pred | gt]` + meta.json with progress/keyframe info. Red border marks the keyframes the model believes are aligned with the current agent position.

## Config Parameters (all in data YAML, saved to config.yaml)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset_fps` | Per-dataset FPS dict | `{'10042': 32}` |
| `task_max_frames` | Max keyframes per num_views | `{2: [6,6]}` |
| `sampling_interval_min/max_mult` | Random interval range | `0.5, 2.0` |
| `task_video_drop_prob` | Per-sample task_video drop | `0.5` |
| `task_text_drop_prob` | Per-sample text drop | `0.5` |
| `task_paths_filename` | Peer episode file name | `"task_paths.json"` |
| `action_video_freq_ratio` | Video downsample ratio | `8` |
| `max_action_len` | Action sequence length | `32` |
| `context_len` | T5 text embedding length | `128` |

## ⚠️ Critical: FastWAM vs FastWAMJoint

**All production configs use `FastWAMJoint` (not `FastWAM` base), even though the main logic lives in `fastwam.py`.**

- `_target_: fastwam.runtime.create_fastwam_joint` → instantiates `FastWAMJoint` (`fastwam_joint.py`)
- `FastWAM` (base, `fastwam.py`) is **never directly instantiated** in production

The two classes differ in **one critical way** — action token self-attention:

| Class | action → video (self-attn) |
|---|---|
| `FastWAM` (base) | first frame only |
| `FastWAMJoint` (production) | **all video tokens** |

Full attention mask for `[task_video | video | action]` in production (`FastWAMJoint`):

```
task_video → task_video:  full
task_video → video/action: blocked
video      → task_video:  allowed
video      → video:       first_frame_causal (first frame cannot see later frames)
action     → task_video:  blocked  (task video injected via dedicated cross-attn layer)
action     → video:       ALL frames
action     → action:      full
```

When reading `fastwam.py:_build_mot_attention_mask`, note that `FastWAMJoint` overrides it — the base version (first-frame only) is **not** the production behavior.

## Progress Token Training Options

The progress token has three independent training modes, all controlled by model YAML fields.

### `noise_to_progress_token` (model YAML, default: `true`)

Controls whether the dedicated progress token (prepended to the video expert sequence) uses flow-matching noise during training.

| Value | Behavior | `loss_progress` |
|-------|----------|-----------------|
| `true` (default) | `progress_gt` is noised via `train_progress_scheduler`; model learns to denoise it | computed normally |
| `false` | clean `progress_gt_normed` fed directly as the conditioning token | skipped (`target_progress=None`) |

**What is NOT affected:** action noise (`noisy_action`) and video noise are completely independent and unchanged.

**Code path** (`fastwam.py:FastWAM.forward`):
```
progress_gt → progress_gt_normed (×2−1)
  if noise_to_progress_token:  add_noise → latents_progress, compute target_progress
  else:                        latents_progress = progress_gt_normed, target_progress = None
→ _prepend_progress_token(latents_progress) → prepended to video sequence
→ if target_progress is not None: loss_progress computed via progress_decoder
```

**Config chain:** `fastwam_joint_cross_attn.yaml` → `runtime.py:create_fastwam_joint` → `fastwam.py:FastWAM.__init__` → `self.noise_to_progress_token`

**Note:** The progress channel embedded inside the *action tensor* (`action[:, :, -2]`) is separate — it is always noised together with the full action tensor via `train_action_scheduler` regardless of this flag.

### `noise_clean_progress_token` (model YAML, default: `false`)

Adds bounded Gaussian noise to the clean progress token during training for robustness. Requires `use_noisy_progress_group=true` and `noise_to_progress_token=false`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_clean_progress_token` | `false` | Enable/disable clean progress noise |
| `noise_clean_progress_prob` | `0.5` | Probability of adding noise per forward pass |
| `noise_clean_progress_scale` | `0.1` | Gaussian std for the noise |

**Motivation:** At inference Stage 2, the clean progress token receives Stage 1's denoised prediction (which has errors). Adding noise during training makes video+action generation robust to imperfect progress conditioning.

**Behavior:** `noisy = clamp(progress + N(0, scale²), -1, 1)`. No loss on clean progress (purely conditioning). t_mod stays at timestep=0 (no train-inference gap).

**Code path** (`fastwam.py:FastWAM.training_loss`):
```
progress_gt_normed (clean GT in [-1, 1])
  if noise_clean_progress_token and rand < prob:
    latents_progress = _add_clean_progress_noise(progress_gt_normed)
  else:
    latents_progress = progress_gt_normed
→ _prepend_progress_token(latents_progress) with timestep=0 t_mod (unchanged)
```

### `use_noisy_progress_group` (model YAML, default: `false`)

**Branch:** `feat/noisy-progress-group`

Enables an isolated noisy group for progress denoising with two-stage inference. Requires `noise_to_progress_token=false` and `task_video_conditioning_mode=prepend_cross_attn`.

**Motivation:** The main group's `clean_progress_token` conditions video+action using GT progress (proven effective). The noisy group learns to predict progress independently from visual observations, then at inference provides the predicted progress to the clean_progress_token.

#### Architecture: Two Token Groups

**Main Group** (unchanged from `noise_to_progress_token=false`):
- `clean_progress_token` (1 token): clean GT progress, no noise, no loss
- `task_video`, `agent_video`, `action`: standard processing

**Noisy Group** (informationally isolated):
- `noisy_progress_token` (1 token): noisy progress, flow-matching loss via `progress_decoder`
- `tv_for_noisy` (S_tv tokens): detached clone of task_video tokens
- `ff_for_noisy` (S_1f tokens): detached clone of agent first frame tokens

#### Video Expert Sequence Layout (Training)

```
[clean_p | task_video | agent_video | noisy_p | tv_noisy | ff_noisy]
 ├── Main Group ─────────────────────┘ ├── Noisy Group ──────────────────┘
```

#### Self-Attention Mask

Complete bidirectional isolation between groups. Noisy group tokens only self-attend within the group; no token in the main group or action expert can attend to the noisy group or vice versa.

```
                    clean_p  task_v  agent_v  noisy_p  tv_noisy  ff_noisy  action
clean_progress       FULL    FULL    FULL     BLOCK    BLOCK     BLOCK     BLOCK
task_video           FULL    FULL    BLOCK    BLOCK    BLOCK     BLOCK     BLOCK
agent_video          FULL    FULL    causal   BLOCK    BLOCK     BLOCK     BLOCK
noisy_progress       BLOCK   BLOCK   BLOCK    FULL     FULL      FULL      BLOCK
tv_for_noisy         BLOCK   BLOCK   BLOCK    FULL     FULL      FULL      BLOCK
ff_for_noisy         BLOCK   BLOCK   BLOCK    FULL     FULL      FULL      BLOCK
action               FULL    BLOCK   ALL      BLOCK    BLOCK     BLOCK     FULL
```

#### Cross-Attention Rules

- **Text cross-attn**: noisy group CAN attend to text (text doesn't leak progress info)
- **Task-video cross-attn** (`prepend_cross_attn` mode): noisy group BLOCKED. Enforced by zeroing `tv_out_dyn` for noisy group rows in `_apply_expert_post_block` (`noisy_group_zero_range`), ensuring training matches inference (both = 0).

#### RoPE Positions

Same as originals (isolation via mask makes duplicates safe):
- `noisy_progress`: temporal = `f_task + f_agent` (same as `clean_progress`)
- `tv_for_noisy`: temporal = 0..f_task-1 (same as `task_video`)
- `ff_for_noisy`: temporal = 0 within agent video (same as agent first frame)

#### Time Modulation (t_mod)

- `noisy_progress_token`: from `_compute_t_mod_from_timestep(timestep_progress)` — reflects actual progress noise level
- `clean_progress_token`: from first agent frame t_mod (timestep=0 in separated_timestep mode)
- `tv_for_noisy` / `ff_for_noisy`: from their source pre_dit (timestep=0 for task_video, first frame t_mod for ff)

#### Token Construction Details

All conditioning tokens (`tv_for_noisy`, `ff_for_noisy`) are `.detach().clone()` from their sources to prevent gradient flow from `loss_progress` back to `pre_dit`. Only `progress_encoder` and the shared video expert DiT blocks receive gradients from progress loss.

`_build_noisy_group()` takes explicit `f_agent_meta` parameter (not derived from `video_pre`) to ensure correct RoPE position at both training and inference.

#### Per-Sample Task Video Drop

When `task_video_dropped=True` for a sample: 4D attention mask blocks both task_video AND noisy group for that sample. `loss_progress` for dropped samples is zeroed.

#### Inference: Two-Stage Pipeline

**Stage 1 — Progress Denoising** (`_infer_progress_stage1`):
- Tokens: `[noisy_p | tv_noisy | ff_noisy]` only (no main group, no action)
- Runs through `MoT.forward_single_expert("video", ...)` — uses same decomposed MoT code paths (`_build_expert_attention_io` → `_mixed_attention` → `_apply_expert_post_block`) for train-inference consistency
- Text cross-attn: allowed (uses projected context from `task_video_pre["context"]`)
- Task-video cross-attn: skipped (no `pac_slice` in context payload)
- `num_inference_steps_progress` steps (configurable, default = main steps)
- Output: denoised progress `[B, 2]` in [-1, 1]

**Stage 2 — Video + Action Denoising** (standard pipeline):
- `clean_progress_token` value = Stage 1 output (fixed, NOT updated during loop)
- Sequence: `[clean_p | task_video | agent_video | action]` (no noisy group)
- Identical to `noise_to_progress_token=false` inference

#### Config

```yaml
noise_to_progress_token: false
use_noisy_progress_group: true
num_inference_steps_progress: 10
loss:
  lambda_progress: 10.0          # must be non-zero for progress training
progress_scheduler:
  num_train_timesteps: 1000
  train_shift: 3.0
  infer_shift: 3.0
```

#### Key Files

| File | Key additions |
|------|--------------|
| `fastwam.py` | `_build_noisy_group()`, `_compute_t_mod_from_timestep()`, `_infer_progress_stage1()`, modified `training_loss()`, `infer_joint()`, `_build_mot_attention_mask()` |
| `fastwam_joint.py` | `_build_mot_attention_mask()` override with `noisy_group_seq_len` |
| `mot.py` | `forward_single_expert()`, `noisy_group_zero_range` in `_apply_expert_post_block` |
| `runtime.py` | Pass `use_noisy_progress_group`, `num_inference_steps_progress` |

## Conventions

- All hardcoded constants must be in config YAML, not in code. Missing config → ValueError (no silent fallback).
- Fallback paths that change computation must have `logging.warning` (no silent data substitution).
- Per-sample drop decisions at dataset side, not model-side batch-level.
- Val config inherits train drop probs via `${data.train.*}` for train-test consistency.
