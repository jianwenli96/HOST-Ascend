# Self-Grounded Prediction (wam/)

This module implements **self-grounded prediction**, the mechanism behind **HOST (Human-to-robot
One-shot Skill Transfer)** that resolves execution from a coupled visual demonstration: it
localizes the robot's current progress within a visual demonstration, predicts the robot's own
future observations conditioned on that localized segment, and derives motor commands from the
predicted future. This is implemented as a single autoregressive diffusion model with dual
video/action experts (a Mixture-of-Transformers architecture) built on top of the
[Fast-WAM](https://arxiv.org/abs/2603.16666) codebase — see [Acknowledgements](#acknowledgements).

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

```text
wam/
├── configs/
│   ├── data/                 # Dataset configs (data_path, cam_mapping_dir, joint_action_mapping_dir, ...)
│   ├── model/                # Model architecture configs (self_grounded_predictor_joint*.yaml)
│   └── task/                 # Task-level configs (composes a data + model config, sets training hparams)
├── scripts/
│   ├── train.py
│   ├── train_zero1_real_pac_headwise_ncp_ve.sh  # Deepspeed ZeRO-1 training entrypoint (verified path)
│   ├── eval_openloop.sh                          # Real-robot open-loop evaluation entrypoint
│   ├── preprocess_action_dit_backbone.py         # Preprocess ActionDiT backbone before training
│   └── precompute_text_embeds.py                 # Precompute T5 text embedding cache before training
├── eval/real_openloop/       # Real-robot open-loop evaluation harness
├── src/self_grounded_prediction/   # Core code (package name; see Environment Setup)
│   ├── models/wan22/         # SelfGroundedPredictor(Joint), video/action DiT, MoT, VAE, text encoder
│   ├── datasets/custom/      # CustomDataset loader — see Data Preparation
│   └── runtime.py            # Hydra factory (create_self_grounded_predictor)
├── runs/                     # Training outputs (ckpt, logs)
├── checkpoints/              # Pretrained or external checkpoints
└── data/                     # Data directory
```

## Environment Setup

```bash
conda create -n self_grounded_prediction python=3.10 -y
conda activate self_grounded_prediction
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The package is importable as `self_grounded_prediction` after this step. Note that the actual
training/eval scripts in this repo don't rely on the pip install — `scripts/_ensure_project_src.py`
puts `src/` on `sys.path` directly, so `pip install -e .` is only needed if you want to `import
self_grounded_prediction` from outside this directory.

## Model Preparation

This step is required before both training and inference.

Step 1: set the Wan model directory first (optional, default `./checkpoints`):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Step 2: pre-generate the ActionDiT backbone (interpolated from Wan2.2 DiT):

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/self_grounded_predictor_joint.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Data Preparation

Training reads three things per dataset: a **video-paths list**, a **camera mapping**, and a
**joint/action normalization mapping**. The shipped configs (`configs/data/custom*.yaml`) point at
this team's internal cluster paths — replace `data_path` / `cam_mapping_dir` /
`joint_action_mapping_dir` with your own, following the format below.

### 1. Video paths list — `{dataset_id}_video_paths.json`

A JSON array of episode directory paths. The dataset id is taken from the filename (e.g.
`10042_video_paths.json` → dataset id `"10042"`), and is used as the lookup key into the camera
mapping, joint/action mapping, and `dataset_fps`/`dataset_image_size` config dicts.

```json
["/data/episodes/task_a/episode_001", "/data/episodes/task_a/episode_002"]
```

`exclude_episode_json` (optional, in the data config) takes the same format and excludes any
listed episode from training.

### 2. Episode directory contents

```text
episode_001/
├── episode_001.json     # action/joint trajectory — REQUIRED if actions/joints are enabled
├── instruction.txt      # plain-text task instruction — REQUIRED
├── instruction.pt       # precomputed T5 text embedding {"context": [L,4096], "mask": [L]}
├── info_dtw.json        # {"aligned_progress": {"<frame_idx>": <0..1>, ...}} — optional
├── task_paths.json      # {"same": ["/data/episodes/task_a/episode_002", ...]} — peer episodes of the same task
├── images/              # frame sequence for one camera view: 0.jpg, 1.jpg, ...
└── gripper_images/      # frame sequence for a second camera view (if 2-view)
```

Video files (`{view}.mp4`) are also supported instead of an image-sequence subdirectory. Frame
count must match the number of entries in `episode_001.json`'s `"data"` array — no silent
truncation.

`episode_001.json` (action/joint trajectory):

```json
{
  "data": [
    {"follow_left_position": [x, y, z], "follow_left_rotation": [r, p, y], "follow_left_gripper": g,
     "follow_right_position": [x, y, z], "follow_right_rotation": [r, p, y], "follow_right_gripper": g},
    "... one entry per frame ..."
  ]
}
```

Rotations are `[roll, pitch, yaw]` Euler triples; when `use_6d_rotation: true` these are converted
to a 6D rotation-matrix representation during loading.

### 3. Camera mapping — `{cam_mapping_dir}/{dataset_id}_cam_mapping.json`

Maps each task directory (one level above the episode directory) to its ordered list of camera/
view names on disk:

```json
{"/data/episodes/task_a": ["images", "gripper_images"]}
```

`num_view_probs` (e.g. `'{"2": 1.0}'`) selects how many of the listed views to sample per batch.

### 4. Joint/action normalization — `{joint_action_mapping_dir}/{dataset_id}_joint_action_mapping.json`

Declares which fields to read from `episode_001.json` and their per-dimension min/delta used to
normalize actions to `[-1, 1]` (`norm = 2*(raw - min)/delta - 1`):

```json
{
  "action_keys": ["follow_left_position", "follow_left_rotation", "follow_left_gripper",
                   "follow_right_position", "follow_right_rotation", "follow_right_gripper"],
  "joint_keys": ["left_rotation", "right_rotation", "left_position", "right_position"],
  "norm_min_delta": {
    "follow_left_position": {"min": [-0.08, -0.08, -0.08], "delta": [0.16, 0.16, 0.16]},
    "follow_left_rotation": {"min": [-0.30, -0.30, -0.30], "delta": [0.60, 0.60, 0.60]},
    "follow_left_gripper":  {"min": [-0.5], "delta": [6.5]}
  }
}
```

### 5. Remaining per-dataset config fields

In `configs/data/custom.yaml`, also set per dataset id: `dataset_fps` (used for minimum-length
filtering and action-frame sampling) and `dataset_image_size` (`[width, height]`). If your dataset
has a different action dimensionality than the shipped example, update `processor.action_output_dim`
/ `processor.proprio_output_dim` and the corresponding `action_dim` in your model config to match.

## Training

### 1) Precompute T5 embedding cache before training

```bash
python scripts/precompute_text_embeds.py task=real_joint_2cam_224_1e-4_pac_headwise_ncp_ve
# multi-GPU:
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=real_joint_2cam_224_1e-4_pac_headwise_ncp_ve
```

### 2) Training

When running a new task for the first time, set `pretrained_norm_stats` in the corresponding
`configs/data/*.yaml` to `null`. After one training run, a `dataset_stats.json` file is generated
in the run directory (`runs/{task_name}/{run_id}/dataset_stats.json`); point `pretrained_norm_stats`
at that file for subsequent runs.

```bash
bash scripts/train_zero1_real_pac_headwise_ncp_ve.sh
```

This is the verified production entrypoint — it composes
`configs/task/real_joint_2cam_224_1e-4_pac_headwise_ncp_ve.yaml`, which selects the
`self_grounded_predictor_joint_cross_attn_ve` model config and the `custom_cross_all` data config.
Override any Hydra field on the command line, e.g. `batch_size=4`.

## Evaluation

`eval/real_openloop/` runs the trained model open-loop against your own episode data (the same
format as [Data Preparation](#data-preparation)), predicting an action chunk from the current
observation and comparing it against ground truth — this is **not** a live-robot control loop, it
replays a recorded dataset:

```bash
bash scripts/eval_openloop.sh
```

Checkpoint and dataset-stats paths are set inside the script; the visual encoder
(`configs/model/*.yaml`'s `visual_encoder.backbone_local_repo`/`backbone_weights_path`/
`siglip_local_weights_path`) currently defaults to internal weight paths with an automatic
fallback to public downloads (`facebookresearch/dinov2` via `torch.hub`, TIMM for SigLIP) when
those fields are unset — see `OPEN_SOURCE_PATH_TODOS.md` at the repo root for the current status of
making every internal path configurable.

## Acknowledgements

This module's codebase is built on top of
[Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
(Yuan et al.). We thank the Fast-WAM authors for releasing their codebase.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{host2026,
  title={HOST: Human-to-robot One-shot Skill Transfer},
  % TODO: fill in authors, venue, year, DOI/arXiv once available
}
```
