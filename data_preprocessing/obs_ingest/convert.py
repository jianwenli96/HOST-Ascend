"""Conversion worker: rebuild shared state and convert one extracted tar.

The whole run shares ONE ``DatasetWriter`` and ONE ``NormAccumulator`` across
all tars (a single dataset-id), so the dataset-level sidecars and the
joint/action normalization statistics accumulate tar by tar.  On resume both
are rebuilt without touching the (deleted) sources:

* the writer from ``conversion_manifest.json`` — it is the last of the three
  dataset-level files the writer rewrites, hence the commit marker;
* the accumulator from the per-tar stats snapshots stored in the pipeline
  state at commit time (min/max folds exactly, independent of order).

``convert_one_tar`` runs the converter's phase A (per-tar, idempotent) and
phase B (the commit, serialized under ``commit_lock`` when parallel
converters run), then marks the tar converted and deletes its staging files.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Optional

import convert_lerobot_dataset as converter
from convert_lerobot.outputs import DatasetWriter
from convert_lerobot.trajectory import NormAccumulator, load_trajectory, norm_mapping_path

from .extract import StagingPaths, remove_download_artifacts, remove_extracted_dir
from .state import (
    STATUS_CONVERTED,
    STATUS_SKIPPED,
    PipelineState,
    TarState,
)


def rebuild_writer_and_accumulator(
    output_dir: Path, args: argparse.Namespace, state: PipelineState
) -> tuple[DatasetWriter, NormAccumulator]:
    """Recreate the shared dataset writer and norm accumulator for this run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = norm_mapping_path(output_dir, args.dataset_id)
    if (output_dir / "conversion_manifest.json").is_file():
        writer = DatasetWriter.from_manifest(output_dir, args.dataset_id, args, mapping_file)
        print(
            f"resumed dataset {args.dataset_id} from conversion_manifest.json: "
            f"{len(writer.video_paths)} episodes already committed"
        )
    else:
        writer = DatasetWriter(output_dir, args.dataset_id, args, mapping_file)
    accumulator = NormAccumulator()
    snapshots = [tar.norm_stats for tar in state.tars.values() if tar.norm_stats is not None]
    accumulator.merge_stats(snapshots)
    if snapshots:
        print(f"resumed norm statistics from {len(snapshots)} converted tars")
    return writer, accumulator


def tar_norm_stats(episodes: list) -> dict:
    """Fold one tar's episodes into a snapshot via the same code path the commit uses."""
    per_tar = NormAccumulator()
    for episode in episodes:
        per_tar.update(load_trajectory(episode.source))
    return per_tar.stats_snapshot()


def convert_one_tar(
    tar: TarState,
    staging: StagingPaths,
    args: argparse.Namespace,
    output_dir: Path,
    dataset_writer: DatasetWriter,
    accumulator: NormAccumulator,
    commit_lock: Optional[threading.Lock],
    state: PipelineState,
    keep_extracted: bool,
) -> int:
    """Convert one extracted tar into the shared dataset and commit it.

    Returns the number of episodes converted.  Raises the converter's own
    exceptions on failure (the caller classifies transient vs permanent);
    the state is only touched after a successful commit.
    """
    extracted_root = staging.extracted_root(tar.tar_id)
    episodes = converter.run_conversion(
        extracted_root,
        output_dir,
        args,
        dataset_writer=dataset_writer,
        accumulator=accumulator,
        commit_lock=commit_lock,
    )
    if not episodes:
        raise ValueError(f"{tar.tar_id}: no episodes discovered under {extracted_root}")

    # Commit: norm snapshot + converted status in one atomic state save
    # (see state.py); cleanup afterwards so a crash between the two leaves at
    # worst unused staging files, never a re-download of a committed tar.
    state.mark_converted(tar, tar_norm_stats(episodes))
    if keep_extracted:
        print(f"NOTE: keeping staging files for {tar.tar_id} (--keep-extracted)")
    else:
        try:
            remove_extracted_dir(staging, tar.tar_id)
            remove_download_artifacts(staging, tar.tar_id)
        except OSError as exc:
            print(
                f"WARNING: {tar.tar_id}: staging cleanup failed ({exc}); "
                f"leftover files can be deleted manually"
            )
    return len(episodes)


def sweep_leftovers(staging: StagingPaths, state: PipelineState, keep_extracted: bool) -> None:
    """Remove staging files left behind by tars that are already terminal.

    A crash between ``mark_converted`` and cleanup leaves orphaned tar /
    extracted files that a resume would otherwise never reclaim.
    """
    for tar in state.tars.values():
        if tar.status not in (STATUS_CONVERTED, STATUS_SKIPPED):
            continue
        remove_download_artifacts(staging, tar.tar_id)
        if not keep_extracted:
            remove_extracted_dir(staging, tar.tar_id)
