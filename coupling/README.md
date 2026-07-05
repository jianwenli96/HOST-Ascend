# Target Coupling — Data Pipeline (coupling/)

This directory holds the offline data-processing pipeline around `alignment/`'s target-coupling
mechanism (`\couplename` in the HOST paper): it builds the task-grouping metadata that alignment
training consumes, and converts alignment's own DTW output into the per-episode progress data
`wam/` reads as its coupled prediction target.

## Pipeline

```
task_grouping/                alignment/ train.py / evaluate_v2.py         progress_alignment/
(build task_paths.json)  -------------------------------------------->    (build info_dtw.json)
        |                       (produces high_loss_samples_*.jsonl               |
        |                        via log_and_save_high_loss_samples)              |
        v                                                                         v
  feeds alignment/'s Main/Reference episode                             consumed by wam/ as
  pairing + wam/'s task-video conditioning                              "aligned_progress"
                                                                          (target coupling output)
```

1. **`task_grouping/`** scans raw episode directories, groups them by task instruction, and writes
   a `task_paths.json` (`{"same": [...]}`) into each episode directory — the schema documented in
   [`docs/data_format.md`](../docs/data_format.md). This is what lets `alignment/` sample a
   "Reference" episode of the same task for its Main episode during TCC/DTW training, and what
   lets `wam/` sample a reference task video for conditioning.
2. Run `alignment/`'s own training or evaluation (`train.py` / `evaluate_v2.py`) — both call
   `log_and_save_high_loss_samples()` (`alignment/utils.py`), which writes DTW alignment records
   (`high_loss_samples_*.jsonl`) into the run's log directory.
3. **`progress_alignment/`** reads those `high_loss_samples_*.jsonl` records and writes
   `info_dtw.json` (`{"aligned_progress": {frame_idx: progress}}`) directly into each episode
   directory. This is the literal, materialized output of target coupling: it maps every frame of
   a robot trajectory to its corresponding position in the demonstration on the shared
   task-progress manifold. `wam/` reads this as `info_dtw.json` (see `docs/data_format.md`).

## Usage

### 1. Task grouping

```bash
cd coupling/task_grouping
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
`--dry_run`, `--max_paths`, ...). Raw data is expected in the layout described in
`docs/data_format.md`, with an additional session-level `instruction.json` mapping episode IDs to
task instructions (used only by `build_task_dictionary.py` to derive task groupings — not read by
`wam/`/`alignment/` directly).

### 2. Progress alignment

After running `alignment/`'s training or evaluation on data with `task_paths.json` already in
place (step 1), point this at the resulting log directory:

```bash
cd coupling/progress_alignment
python build_progress_info.py --log_file /path/to/alignment_run/logs/high_loss_samples_all.jsonl
```

Accepts either a single `.jsonl` file or a directory of `high_loss_samples_<N>.jsonl` files
(processed in numeric order). Records are skipped (no `info_dtw.json` written) if the alignment
loss is too high, the alignment range is too small, the alignment is non-causal, or there's too
large a frame jump — see the quality-filter thresholds in `build_progress_info.py`.

## Note on verification

These are one-off, offline data-preprocessing scripts, not training loops — verified via static
checks (`ast.parse`, CLI argument consistency) rather than the actual-run training protocol used
for `wam/`/`alignment/`.
