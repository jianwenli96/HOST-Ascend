# HOST dataset format

`wam/` (self-grounded prediction) and `alignment/` (target coupling) consume the **same on-disk
data convention**: a JSON list of episode directory paths, each episode directory following a
shared layout, plus two shared sidecar mapping files (camera views, joint/action normalization).
This doc is the single source of truth for that format; both modules' shipped configs
(`wam/configs/data/custom*.yaml`, `alignment/train_scripts/run_ds_10042.sh`) point at this team's
internal cluster paths — replace them with your own data in this format.

Per-module differences are called out inline and summarized in [§6](#6-per-module-differences).

## 1. Video paths list — `{dataset_id}_video_paths.json`

A JSON array of episode directory paths. The dataset id is taken from the filename (e.g.
`10042_video_paths.json` → dataset id `"10042"`) and used as the lookup key into the camera
mapping, joint/action mapping, and any per-dataset config dicts (`dataset_fps`,
`dataset_image_size`, ...).

```json
["/data/episodes/task_a/episode_001", "/data/episodes/task_a/episode_002"]
```

- `wam/`: passed via `data.train.data_path` (comma-separated if multiple files); an
  `exclude_episode_json` field of the same format excludes listed episodes.
- `alignment/`: passed via `--video_paths` (comma-separated). Entries may also address a segment
  of a longer recording as `path:segment_id:start-end` (inclusive frame range) — see
  `alignment/SEGMENTED_VIDEO_FORMAT.md` for that case.

## 2. Episode directory contents

```text
episode_001/
├── episode_001.json     # action/joint trajectory — required by wam/; optional for alignment/
├── instruction.txt      # plain-text task instruction — required by wam/; optional for alignment/
├── instruction.pt       # precomputed T5 text embedding {"context": [L,4096], "mask": [L]} — wam/ only
├── info_dtw.json        # {"aligned_progress": {"<frame_idx>": <0..1>, ...}} — wam/ only, optional
├── task_paths.json      # {"same": ["/data/episodes/task_a/episode_002", ...]} — peer episodes of the same task
├── images/              # frame sequence for one camera view: 0.jpg, 1.jpg, ...
└── gripper_images/      # frame sequence for a second camera view (if 2-view)
```

Video files (`{view}.mp4`) are also supported instead of an image-sequence subdirectory. Frame
count must match the number of entries in `episode_001.json`'s `"data"` array where that file is
used — no silent truncation.

`task_paths.json` and `info_dtw.json` aren't hand-authored — see [`coupling/`](../coupling/) for
the pipeline that produces them: `coupling/task_grouping/` writes `task_paths.json` from raw
episode directories, and `coupling/progress_alignment/` writes `info_dtw.json` from `alignment/`'s
own DTW training/eval output.

`task_paths.json` lists peer episodes of the *same task*. `wam/` uses it to sample a reference
task video shown alongside the primary observation; `alignment/` uses it to pick the "Reference"
video that the "Main" video's chunks are aligned against (the `"same"` pool is preferred, a
lower-quality `"100-95"` pool is a fallback if present). Without it, `alignment/` falls back to
self-alignment (Main and Reference are the same episode).

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

Rotations are `[roll, pitch, yaw]` Euler triples; `wam/` converts these to a 6D rotation-matrix
representation during loading when `use_6d_rotation: true`.

## 3. Camera mapping — `{cam_mapping_dir}/{dataset_id}_cam_mapping.json`

Maps each task directory (one level above the episode directory) to its ordered list of camera/
view names on disk:

```json
{"/data/episodes/task_a": ["images", "gripper_images"]}
```

- `wam/`: `num_view_probs` (e.g. `'{"2": 1.0}'`) selects how many of the listed views to sample
  per batch.
- `alignment/`: only needed for multi-view episodes (`CONFIG.DATA.MULTI_VIEW_DATASETS` /
  `CONFIG.DATA.NUM_VIEWS`); with a single view this file can be omitted (the first available
  camera / `images/` subdir is used).

## 4. Joint/action normalization — `{joint_action_mapping_dir}/{dataset_id}_joint_action_mapping.json`

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

## 5. Remaining per-dataset config fields (`wam/` only)

In `wam/configs/data/custom.yaml`, also set per dataset id: `dataset_fps` (used for
minimum-length filtering and action-frame sampling) and `dataset_image_size` (`[width, height]`).
If your dataset has a different action dimensionality than the shipped example, update
`processor.action_output_dim` / `processor.proprio_output_dim` and the corresponding `action_dim`
in your model config to match.

## 6. Per-module differences

| | `wam/` | `alignment/` |
|---|---|---|
| Action/joint data (`episode_001.json`) | **Required** | Optional — only read if `CONFIG.JOINTS.USE_JOINTS: true` |
| `instruction.txt` / `instruction.pt` | Required | Optional |
| `info_dtw.json` (progress) | Used, optional | Not used |
| `task_paths.json` | Used for reference-task sampling | Used for Main/Reference pairing (falls back to self-alignment if absent) |
| Camera mapping | Used for multi-view sampling | Only needed for multi-view episodes |
| Config location | `wam/configs/data/custom*.yaml` | `alignment/config.py` (`CONFIG.DATA.*`, `CONFIG.JOINTS.*`) |
