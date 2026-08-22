# Dataset Format & Preprocessing (data_preprocessing/)

This is the first stage of the HOST pipeline (**dataset → alignment → coupling → policy_training**):
it defines the on-disk data convention that `alignment/` and `policy_training/` both consume, and
provides the tooling that constructs part of it (`task_paths.json`) from raw episode directories.
(`coupling/` is the second data-processing stage — it runs *after* `alignment/` training/eval and
produces `info_dtw.json`; see [`coupling/README.md`](../coupling/README.md).)

## 1. Task grouping

`build_task_dictionary.py` scans raw episode directories, groups them by task instruction, and
`write_task_paths.py` writes the resulting `task_paths.json` (`{"same": [...]}`) into each episode
directory. This is what lets `alignment/` sample a "Reference" episode of the same task for its
Main episode during TCC/DTW training, and what lets `policy_training/` sample a reference task
video for conditioning.

```bash
cd data_preprocessing
bash run_task_grouping.sh   # edit --input_dir / --output_path / --dataset_name inside first
```

`run_task_grouping.sh` composes two steps:

```bash
python build_task_dictionary.py \
  --input_dir /path/to/your/datasets \
  --output_path ./output/dataset_name.hdf5 \
  --dataset_name dataset_name \
  --clear

python write_task_paths.py \
  --hdf5_path ./output/dataset_name.hdf5
```

See `--help` on either script for the full argument list (`--skip_classes`, `--overwrite`,
`--dry_run`, `--max_paths`, ...). Raw data is expected in the layout described below, with an
additional session-level `instruction.json` mapping episode IDs to task instructions (used only by
`build_task_dictionary.py` to derive task groupings — not read by `policy_training/`/`alignment/`
directly).

Verification: this is a one-off, offline data-preprocessing script, not a training loop — verified
via static checks (`ast.parse`, CLI argument consistency) rather than the actual-run training
protocol used for `policy_training/`/`alignment/`.

### 1.1. Paired HumanAndRobot HDF5 conversion

`convert_human_and_robot.py` handles the paired HDF5 layout from the HumanAndRobot dataset. Each
source file contains synchronized `/cam_data/human_camera` and `/cam_data/robot_camera` arrays.
The converter writes separate robot Main and human Reference episode directories, uses all human
episodes of the same task as the robot episode's training reference pool, and fixes evaluation to
the synchronized pair from the same HDF5 file. Each robot episode receives a 7D robot action
trajectory built from `/end_position` (XYZ position plus XYZ Euler rotation) and `/gripper_state`,
together with 14D robot proprioception built from the seven joint positions in `/qpos` and seven
joint velocities in `/qvel`. The v1 `/action` field is deliberately excluded because it stores the
human hand pose rather than the robot action. Positions are converted from millimetres to metres;
Euler angles, joint positions, and joint velocities are converted from degrees/degrees per second
to radians/radians per second.

JPEG image sequences are used because they work with the image-loading path without depending on
the codecs supported by the installed `decord`/FFmpeg build.

```bash
python data_preprocessing/convert_human_and_robot.py \
  --input-dir /path/to/HumanAndRobot/data \
  --output-dir /path/to/HumanAndRobot/align_data
```

The output includes `HumanAndRobot_video_paths.json`,
`cam_mapping/HumanAndRobot_cam_mapping.json`, per-episode `task_paths.json` and
`task_paths_eval.json`, robot trajectory JSON files,
`joint_action_mapping/HumanAndRobot_joint_action_mapping.json`, instructions, and
`conversion_manifest.json`. Launch alignment with the generated camera mapping:

```bash
HOST_CAM_MAPPING_DIR=/path/to/HumanAndRobot/align_data/cam_mapping \
HOST_JOINT_ACTION_MAPPING_DIR=/path/to/HumanAndRobot/align_data/joint_action_mapping \
VIDEO_PATHS=/path/to/HumanAndRobot/align_data/HumanAndRobot_video_paths.json \
bash alignment/train_scripts/run_ds.sh
```

### 1.2. LeRobot v2 conversion

`convert_lerobot_dataset.py` handles LeRobot v2.1 datasets in the GR00T convention
(`meta/info.json` reports `codebase_version: "v2.1"`, e.g. the AgiBot A2D bimanual robot):
per-episode Parquet files under `data/chunk-*/episode_XXXXXX.parquet`, per-view MP4 videos under
`videos/chunk-*/<video_key>/episode_XXXXXX.mp4`, and optional `meta/tasks.jsonl` task
instructions. It is the robot-only counterpart of §1.1 — there is no synchronized human video,
so each episode emits a single episode directory and its `task_paths.json` lists same-task
*robot* episodes (a deterministic same-task peer goes in `task_paths_eval.json`).

Each episode receives a 14D bimanual action trajectory built from `actions.end.position` (3+3
XYZ positions, metres), `actions.end.orientation` (4+4 xyzw quaternions converted to HOST
`[roll, pitch, yaw]` Euler triples, radians), and `actions.effector.position` (2 gripper
commands, 0=closed 1=open), plus 14D proprioception from `observation.states.joint.position`
(7+7 joint angles, radians). Columns that do not map to the HOST convention (waist, head, robot
base, velocities, `actions.joint.position`) are excluded and listed in the manifest. Source
videos are AV1-coded, so frames are extracted with the `ffmpeg` CLI (required on the machine
running the converter) and stored as JPEG image sequences.

```bash
python data_preprocessing/convert_lerobot_dataset.py \
  --input-dir robot_datasets \        # a dataset dir, or a root containing several
  --output-dir ./output/align_data \
  --dataset-id RobotTask
```

`--main-view`/`--gripper-views` select which source video keys become `images/` and
`gripper_images/` (defaults: `observation.images.head` / `observation.images.hand_left`);
`--no-gripper-view` emits a single-view dataset. `--short-instructions` writes only the task
text before the first `|` to `instruction.txt`; `--max-episodes` bounds a partial run;
`--verify-only` validates an existing conversion against the source files.

The converter processes one task at a time (task-by-task discovery, so tasks may stream in
over time) and encodes the episodes of each task in parallel: `--workers` sets the number of
parallel episode encodes (default `min(4, cpu count)`).

The output mirrors §1.1: `<dataset_id>_video_paths.json`, `cam_mapping/`, per-episode
`task_paths.json`/`task_paths_eval.json`/`instruction.txt`, trajectory JSON files,
`joint_action_mapping/`, and `conversion_manifest.json`. Each task directory also receives the
session-level `instruction.json` that `build_task_dictionary.py` reads, so the converted output
can be fed straight into the §1 grouping pipeline (pass `--input_dir` as an absolute path so
the stored episode paths stay absolute). Launch alignment with the same env vars as §1.1; for
`policy_training/` add the dataset id to the data config and note the action dimensionality is
14 (20 after 6D-rotation expansion) with 14D proprioception (see §2.5).

### 1.3. Streaming OBS ingestion

`obs_streaming_convert.py` (package: `obs_ingest/`) downloads `task_XXX.tar.gz` objects from a
Huawei OBS bucket and feeds them through the §1.2 converter *as they arrive* — no need to mirror
everything locally first. All tars merge into one dataset-id. It targets large corpora (e.g. the
AgiBot Beta LeRobot tars: 183 objects, ~9 TB):

* **Pipelined stages** — download / extract / convert overlap (bounded queues): tar N converts
  while N+1 extracts and N+2 downloads.
* **Parallel ranged downloads** — each tar is fetched as `--parts-per-tar` Range GETs written
  with `os.pwrite` into one preallocated sparse file; per-part markers survive restarts, so a
  re-run resumes at byte granularity (the OBS SDK only retries connection *opening*, mid-stream
  deaths are reconnected here — see `obs_ingest/obsio.py`).
* **View spec probing** — the smallest few tars' `meta/info.json` are streamed (videos never
  fetched) to pick the run-level views: when every probe ships the `*_compress` 224×224 h264
  views they are used (≈10× cheaper than decoding the 480×640 AV1 originals); otherwise the
  full-resolution keys plus `--output-size` apply. Explicit `--main-view`/`--gripper-views`
  flags skip probing. The spec is stored in the pipeline state and reused on resume.
* **Crash-consistent resume** — per-tar state machine in `<staging>/obs_pipeline_state.json`
  (atomic writes; converted ⇒ committed + norm stats saved in one write). On resume the shared
  `DatasetWriter` is rebuilt from `conversion_manifest.json` (the commit marker) and the norm
  accumulator from the per-tar stats snapshots; `DatasetWriter.add_task` is idempotent, so
  crash-window retries cannot duplicate episodes.
* **Bounded staging disk** — `--max-staging-gb` backpressures the downloaders (one tar may
  overshoot the cap); staging files are deleted after each tar commits (`--keep-extracted` to
  keep them).
* **Failure taxonomy** — transient (network, gzip CRC → re-download) vs permanent (4xx, bad
  layout, missing views, `min_frames`/norm violations), each with `--retries` bounded and
  persisted; failed tars are listed at the end and retried with `--force-retry-failed`.

```bash
# environment: host_alignment conda env (pyarrow/numpy/ffmpeg) + obs SDK
#   (pip install esdk-obs-python, or --obs-sdk-path pointing at its src dir)
python data_preprocessing/obs_streaming_convert.py \
  --config obs_infos.txt \                       # OBS_AK/OBS_SK/OBS_ENDPOINT/OBS_BUCKET
  --prefix Agibot_Beta_Lerobot_Amap/ \
  --output-dir /path/to/align_data \
  --dataset-id AgibotA2D

python data_preprocessing/obs_streaming_convert.py --list-only    # inspect the objects
python data_preprocessing/obs_streaming_convert.py --status       # per-tar state table
python data_preprocessing/obs_streaming_convert.py --limit 1      # trial (smallest tar)
```

Useful knobs: `--download-workers 2` / `--extract-workers 2` / `--convert-workers 1` /
`--parts-per-tar 8` / `--max-staging-gb 200` / `--retries 3` / `--timeout 120` /
`--probe-tars 3`. Converter flags pass through unchanged (`--workers`, `--min-frames`,
`--jpeg-quality`, `--output-size`, `--short-instructions`, `--overwrite`); `--max-episodes`
is a **global** cap across all tars. Verification is per tar (`verify_task` at commit) plus a
source-free end-of-run check of the dataset artifacts (`--skip-final-verify` to skip); the
full `verify_dataset` that reads source Parquets is not available because sources are deleted
after each commit. After the run, feed the output into the §1 grouping pipeline
(`build_task_dictionary.py` + `write_task_paths.py`) as usual. Synthetic tests (part-download
resume, extraction safety, state machine, writer rebuild): `python
data_preprocessing/test_obs_ingest.py`.

## 2. Dataset format

`policy_training/` (self-grounded prediction) and `alignment/` (target coupling) consume the
**same on-disk data convention**: a JSON list of episode directory paths, each episode directory
following a shared layout, plus two shared sidecar mapping files (camera views, joint/action
normalization). This is the single source of truth for that format. Pass the episode-list JSON to
`alignment/train_scripts/run_ds.sh` through `VIDEO_PATHS`, and set the corresponding fields in
`policy_training/configs/data/custom*.yaml`.

Per-module differences are called out inline and summarized in [§2.6](#26-per-module-differences).

### 2.1. Video paths list — `{dataset_id}_video_paths.json`

A JSON array of episode directory paths. The dataset id is taken from the filename (e.g.
`10042_video_paths.json` → dataset id `"10042"`) and used as the lookup key into the camera
mapping, joint/action mapping, and any per-dataset config dicts (`dataset_fps`,
`dataset_image_size`, ...).

```json
["/data/episodes/task_a/episode_001", "/data/episodes/task_a/episode_002"]
```

- `policy_training/`: passed via `data.train.data_path` (comma-separated if multiple files); an
  `exclude_episode_json` field of the same format excludes listed episodes.
- `alignment/`: passed via `--video_paths` (comma-separated). Entries may also address a segment
  of a longer recording as `path:segment_id:start-end` (inclusive frame range), parsed by
  `AlignmentDataset._parse_video_path` in `alignment/datasets.py`.

### 2.2. Episode directory contents

```text
episode_001/
├── episode_001.json     # action/joint trajectory — required by policy_training/; optional for alignment/
├── instruction.txt      # plain-text task instruction — required by policy_training/; optional for alignment/
├── instruction.pt       # precomputed T5 text embedding {"context": [L,4096], "mask": [L]} — policy_training/ only
├── info_dtw.json        # {"aligned_progress": {"<frame_idx>": <0..1>, ...}} — policy_training/ only, optional
├── task_paths.json      # {"same": ["/data/episodes/task_a/episode_002", ...]} — peer episodes of the same task
├── images/              # frame sequence for one camera view: 0.jpg, 1.jpg, ...
└── gripper_images/      # frame sequence for a second camera view (if 2-view)
```

Video files (`{view}.mp4`) are also supported instead of an image-sequence subdirectory. Frame
count must match the number of entries in `episode_001.json`'s `"data"` array where that file is
used — no silent truncation.

`task_paths.json` and `info_dtw.json` aren't hand-authored — `task_paths.json` is produced by
[§1 above](#1-task-grouping), and `info_dtw.json` is produced by
[`coupling/`](../coupling/README.md) from `alignment/`'s own DTW training/eval output.

`task_paths.json` lists peer episodes of the *same task*. `policy_training/` uses it to sample a
reference task video shown alongside the primary observation; `alignment/` uses it to pick the
"Reference" video that the "Main" video's chunks are aligned against (the `"same"` pool is
preferred, a lower-quality `"100-95"` pool is a fallback if present). Without it, `alignment/`
falls back to self-alignment (Main and Reference are the same episode).

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

Rotations are `[roll, pitch, yaw]` Euler triples; `policy_training/` converts these to a 6D
rotation-matrix representation during loading when `use_6d_rotation: true`.

### 2.3. Camera mapping — `{cam_mapping_dir}/{dataset_id}_cam_mapping.json`

Maps each task directory (one level above the episode directory) to its ordered list of camera/
view names on disk:

```json
{"/data/episodes/task_a": ["images", "gripper_images"]}
```

- `policy_training/`: `num_view_probs` (e.g. `'{"2": 1.0}'`) selects how many of the listed views
  to sample per batch.
- `alignment/`: only needed for multi-view episodes (`CONFIG.DATA.MULTI_VIEW_DATASETS` /
  `CONFIG.DATA.NUM_VIEWS`); with a single view this file can be omitted (the first available
  camera / `images/` subdir is used).

### 2.4. Joint/action normalization — `{joint_action_mapping_dir}/{dataset_id}_joint_action_mapping.json`

Declares which fields to read from `episode_001.json` and their per-dimension min/delta used to
normalize to `[-1, 1]` (`norm = 2*(raw - min)/delta - 1`):

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

Both modules point their `joint_action_mapping_dir`/`JOINT_ACTION_MAPPING_DIR` config at the same
directory and read this exact `action_keys`/`joint_keys`/`norm_min_delta` structure.

### 2.5. Remaining per-dataset config fields (`policy_training/` only)

In `policy_training/configs/data/custom.yaml`, also set per dataset id: `dataset_fps` (used for
minimum-length filtering and action-frame sampling) and `dataset_image_size` (`[width, height]`).
If your dataset has a different action dimensionality than the shipped example, update
`processor.action_output_dim` / `processor.proprio_output_dim` and the corresponding `action_dim`
in your model config to match.

### 2.6. Per-module differences

| | `policy_training/` | `alignment/` |
|---|---|---|
| Action/joint data (`episode_001.json`) | **Required** | Optional — only read if `CONFIG.JOINTS.USE_JOINTS: true` |
| `instruction.txt` / `instruction.pt` | Required | Optional |
| `info_dtw.json` (progress) | Used, optional | Not used |
| `task_paths.json` | Used for reference-task sampling | Used for Main/Reference pairing (falls back to self-alignment if absent) |
| Camera mapping | Used for multi-view sampling | Only needed for multi-view episodes |
| Config location | `policy_training/configs/data/custom*.yaml` | `alignment/config.py` (`CONFIG.DATA.*`, `CONFIG.JOINTS.*`) |
