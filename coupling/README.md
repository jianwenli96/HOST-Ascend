# Target Coupling — Progress Alignment (coupling/)

This directory converts `alignment/`'s own DTW training/eval output into the per-episode progress
data `policy_training/` reads as its coupled prediction target — the literal materialization of
**target coupling** (`\couplename` in the HOST paper) into data.

## Pipeline position

```
data_preprocessing/            alignment/ train.py / evaluate_v2.py              coupling/
(task_paths.json)         -------------------------------------------->    (build info_dtw.json)
        |                       (produces high_loss_samples_*.jsonl               |
        |                        via log_and_save_high_loss_samples)              |
        v                                                                         v
  feeds alignment/'s Main/Reference episode                             consumed by policy_training/
  pairing (see data_preprocessing/README.md)                            as "aligned_progress"
                                                                          (target coupling output)
```

1. `data_preprocessing/` builds `task_paths.json`, which lets `alignment/` sample a "Reference"
   episode of the same task for its Main episode during TCC/DTW training.
2. Run `alignment/`'s own training or evaluation (`train.py` / `evaluate_v2.py`) — both call
   `log_and_save_high_loss_samples()` (`alignment/utils.py`), which writes DTW alignment records
   (`high_loss_samples_*.jsonl`) into the run's log directory.
3. `coupling/progress_alignment/build_progress_info.py` reads those `high_loss_samples_*.jsonl`
   records and writes `info_dtw.json` (`{"aligned_progress": {frame_idx: progress}}`) directly
   into each episode directory. This maps every frame of a robot trajectory to its corresponding
   position in the demonstration on the shared task-progress manifold. `policy_training/` reads
   this as `info_dtw.json` (see `data_preprocessing/README.md`).

## Usage

After running `alignment/`'s training or evaluation on data with `task_paths.json` already in
place, point this at the resulting log directory:

```bash
cd coupling/progress_alignment
python build_progress_info.py --log_file /path/to/alignment_run/logs/high_loss_samples_all.jsonl
```

Accepts either a single `.jsonl` file or a directory of `high_loss_samples_<N>.jsonl` files
(processed in numeric order). Records are skipped (no `info_dtw.json` written) if the alignment
loss is too high, the alignment range is too small, the alignment is non-causal, or there's too
large a frame jump — see the quality-filter thresholds in `build_progress_info.py`.

## Note on verification

This is a one-off, offline data-preprocessing script, not a training loop — verified via static
checks (`ast.parse`, CLI argument consistency) rather than the actual-run training protocol used
for `policy_training/`/`alignment/`.
