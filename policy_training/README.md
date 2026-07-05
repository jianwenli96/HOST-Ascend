# Self-Grounded Prediction (policy_training/)

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
policy_training/
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
  --model-config configs/model/self_grounded_predictor_joint_cross_attn_ve.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Data Preparation

`policy_training/` and `alignment/` consume the same on-disk data convention — see
[`data_preprocessing/README.md`](../data_preprocessing/README.md) at the repo root for the full,
single-source-of-truth schema (video-paths list, episode directory layout, camera mapping,
joint/action normalization). The shipped configs (`configs/data/custom*.yaml`) point at this
team's internal cluster paths — replace `data_path` / `cam_mapping_dir` /
`joint_action_mapping_dir` with your own data in that format.

`policy_training/` specifically **requires** the action/joint trajectory (`episode_001.json`) and
text instruction (`instruction.txt`/`instruction.pt`) fields described there — see
[§2.6 of `data_preprocessing/README.md`](../data_preprocessing/README.md#26-per-module-differences)
for the exact per-module requirements. Also set, per dataset id in `configs/data/custom_cross_all.yaml`:
`dataset_fps` (minimum-length filtering and action-frame sampling) and `dataset_image_size`
(`[width, height]`). If your dataset has a different action dimensionality than the shipped
example, update `processor.action_output_dim` / `processor.proprio_output_dim` and the
corresponding `action_dim` in your model config to match.

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
