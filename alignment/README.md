# Target Coupling (alignment/)

This module implements **target coupling**, the mechanism behind **HOST (Human-to-robot
One-shot Skill Transfer)** that aligns a visual demonstration to a robot trajectory on a shared
task-progress manifold, so the robot's prediction target is coupled to the corresponding point in
the demonstration rather than to raw clock time. It trains a frame-embedding model (Qwen3-VL
backbone) with Temporal Cycle-Consistency (TCC) and Smooth DTW losses so that semantically
corresponding moments across two videos of the same task map to nearby points in embedding space;
`wam/`'s dataset construction uses this alignment to build its coupled prediction targets.

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## File Structure

```text
alignment/
├── train.py                  # Training entrypoint
├── evaluate_v2.py             # Evaluation entrypoint
├── models.py                  # BaseModel, embedders, AttentionGate
├── datasets.py                # AlignmentDataset / AlignmentVideoDataset, AlignmentCollator
├── config.py                  # CONFIG.* — data/training/model hyperparameters
├── tcc/                       # Smooth DTW + TCC loss implementations
├── algos/                     # Alignment / Algorithm module-level wrappers
├── monkey_patch_forward.py    # Qwen3-VL packed-sequence attention/position-id patches
├── train_scripts/
│   └── run_ds_10042.sh        # Deepspeed training entrypoint (verified path)
└── scripts/                   # Checkpoint conversion utilities
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate emu_vla_rl
```

`environment.yml` is a full conda environment export; if you'd rather build one incrementally,
the key dependencies are PyTorch, DeepSpeed, `transformers` (for Qwen3-VL), and the Qwen3-VL
image/video processor stack.

## Data Preparation

Training reads a **video-paths list**, each entry's episode directory, and (optionally) a
**camera mapping** — the shipped script points at this team's internal cluster paths
(`--video_paths /open_data/cgy/processed_data/video_paths_basket/clean/10042_video_paths.json`).
Replace this with your own data in the format below. This module needs only video frames plus
task grouping — no robot actions/joint data are required unless you enable joint conditioning
(`CONFIG.JOINTS.USE_JOINTS`).

### 1. Video paths list — `{dataset_id}_video_paths.json`

A JSON array of episode paths, passed via `--video_paths` (comma-separated if using multiple
files). The dataset id is taken from the filename and used to look up camera mappings /
per-dataset config.

```json
["/data/episodes/task_a/episode_001", "/data/episodes/task_a/episode_002"]
```

Each entry can also address a segment of a longer recording as `path:segment_id:start-end`
(inclusive frame range) — see `alignment/SEGMENTED_VIDEO_FORMAT.md` for the segmented-recording
case.

### 2. Episode directory contents

```text
episode_001/
├── images/              # frame sequence: 0.jpg, 1.jpg, ...     (or images.mp4)
├── instruction.txt      # plain-text task instruction — optional
└── task_paths.json      # {"same": ["/data/episodes/task_a/episode_002", ...]}
```

`task_paths.json` lists peer episodes of the *same task* — during training, one is sampled as the
"Reference" video that the "Main" video's chunks are aligned against (the `"same"` pool is
preferred; a lower-quality `"100-95"` pool is used as fallback if present). Without it, training
falls back to self-alignment (Main and Reference are the same episode).

### 3. Camera mapping (optional, multi-view only) — `{cam_mapping_dir}/{dataset_id}_cam_mapping.json`

Only needed if an episode has more than one camera view:

```json
{"/data/episodes/task_a": ["images", "gripper_images"]}
```

Set `CONFIG.DATA.MULTI_VIEW_DATASETS` / `CONFIG.DATA.NUM_VIEWS` in `config.py` accordingly. With a
single view, this file can be omitted (the first available camera / `images/` subdir is used).

### 4. Relevant `config.py` fields

```python
CONFIG.DATA.NUM_STEPS = 3            # frames per chunk
CONFIG.DATA.FRAME_STRIDE = 10        # stride between context frames
CONFIG.TRAIN.NUM_FRAMES = 24         # anchor frames sampled per video
CONFIG.TRAIN.NUM_ALIGN_FRAMES = 24   # reference frames used for the alignment window
CONFIG.DATA.CAM_MAPPING_DIR = ''     # set to your cam_mapping directory if multi-view
CONFIG.DATA.SEGMENTED_PATH_DATASETS = ('AgiBotWorld', '10042')  # dataset ids using the segmented path:id:start-end format
CONFIG.JOINTS.USE_JOINTS = False     # set True + JOINT_ACTION_MAPPING_DIR to condition on joint state
```

## Training

```bash
bash train_scripts/run_ds_10042.sh
```

This wraps `train.py` with `torchrun` + DeepSpeed ZeRO-3. Key flags (see `train.py --help` for the
full list): `--video_paths` (required), `--network` (`Qwen3-VL-2B` is the verified path),
`--gradient_accumulation_steps`, `--ds_config`, `--save_interval`, `--max_iters`, `--resume_dir`,
`--pretrain_weights`. `WANDB_API_KEY` falls back to offline logging if unset.

## Evaluation

```bash
python evaluate_v2.py --video_paths <your_video_paths.json> --network Qwen3-VL-2B --resume_dir <checkpoint_dir>
```

Evaluates alignment quality (embedding-space nearest-neighbor correspondence) between Main and
Reference videos using the same data format as training. `--resume_dir` must contain one of
`checkpoint.pth.tar`, `fp32_converted/pytorch_model.bin`, or `pytorch_model.bin` (see
`scripts/convert_ds_to_hf.sh` / `scripts/convert_ckpt_to_hf.py` for converting a DeepSpeed
checkpoint into one of these formats).
