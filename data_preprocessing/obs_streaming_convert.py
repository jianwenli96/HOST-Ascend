#!/usr/bin/env python3
"""Stream LeRobot tar.gz objects from Huawei OBS through the HOST converter.

Downloads the ``task_XXX.tar.gz`` objects under an OBS prefix and feeds them
through the existing LeRobot conversion pipeline (``convert_lerobot_dataset.py``)
as they arrive, instead of first mirroring everything locally.  All 183 tars
merge into ONE dataset-id: a single ``{dataset_id}_video_paths.json`` plus
shared camera mapping and joint/action normalization statistics.

The pipeline is resumable: per-tar state (including per-part download
markers) lives in the staging directory, so an interrupted run continues
where it stopped and never re-downloads a committed tar.

Performance defaults target the 9 TB AgiBot corpus: parallel ranged-GET parts
per tar (OBS per-connection streams are the bottleneck), a few tars in flight,
and — when the tars ship ``*_compress`` 224x224 h264 views (probed from the
smallest tars' meta/info.json) — conversion from those instead of decoding
the 480x640 AV1 originals (~10x cheaper).

Usage:
    python obs_streaming_convert.py --config obs_infos.txt \
        --output-dir /path/to/align_data --dataset-id AgibotA2D

    python obs_streaming_convert.py --list-only      # inspect the objects
    python obs_streaming_convert.py --status         # per-tar state table
    python obs_streaming_convert.py --limit 1        # trial run (smallest tar)

Environment: host_alignment conda env (pyarrow/numpy/ffmpeg) + the OBS SDK
(pip install esdk-obs-python, or --obs-sdk-path pointing at its src dir).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import convert_lerobot_dataset  # noqa: F401 - dependency check + shared helpers
from convert_lerobot.media import require_ffmpeg
from convert_lerobot.source import DEFAULT_GRIPPER_VIEWS, DEFAULT_MAIN_VIEW

from obs_ingest import obsio, viewspec
from obs_ingest.convert import rebuild_writer_and_accumulator, sweep_leftovers
from obs_ingest.extract import StagingPaths
from obs_ingest.pipeline import StreamingPipeline
from obs_ingest.state import PipelineState

DEFAULT_PREFIX = "Agibot_Beta_Lerobot_Amap/"
DEFAULT_PROBE_MAX_STREAM_BYTES = 4 * 1024 ** 3  # probe cap: never fetch the videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream LeRobot tar.gz objects from OBS through the HOST converter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- OBS side
    parser.add_argument("--config", default="obs_infos.txt", help="Credential file (OBS_AK/OBS_SK/OBS_ENDPOINT/OBS_BUCKET).")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Object prefix to scan.")
    parser.add_argument("--obs-sdk-path", default=None, help="Directory of the obs SDK 'src' (sys.path fallback).")
    parser.add_argument("--list-only", action="store_true", help="List matching objects and exit.")
    parser.add_argument("--refresh-keys", action="store_true", help="Ignore the listing cache and re-list.")
    parser.add_argument("--status", action="store_true", help="Print the per-tar state table and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve state/views/writer and print what would run, then exit.")
    parser.add_argument("--force-retry-failed", action="store_true", help="Re-schedule failed_permanent tars.")
    parser.add_argument("--force-respec", action="store_true", help="Ignore the stored view spec and re-probe.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N new (not-yet-converted) tars (0 = all).")
    parser.add_argument("--staging-dir", type=Path, default=None, help="Staging area (default: <output-dir>.staging).")
    parser.add_argument("--download-workers", type=int, default=2, help="Tars downloading concurrently.")
    parser.add_argument("--extract-workers", type=int, default=2, help="Tars extracting concurrently.")
    parser.add_argument("--convert-workers", type=int, default=1, help="Converters running concurrently.")
    parser.add_argument("--parts-per-tar", type=int, default=8, help="Ranged GET parts per tar.")
    parser.add_argument("--max-staging-gb", type=float, default=200.0, help="Staging disk budget (one tar may overshoot).")
    parser.add_argument("--retries", type=int, default=3, help="Transient retries per tar per stage.")
    parser.add_argument("--timeout", type=int, default=120, help="Socket timeout in seconds.")
    parser.add_argument("--probe-tars", type=int, default=3, help="Small tars probed for the view spec.")
    parser.add_argument("--probe-max-stream-bytes", type=int, default=DEFAULT_PROBE_MAX_STREAM_BYTES, help="Compressed bytes fetched per probe.")
    parser.add_argument("--keep-extracted", action="store_true", help="Keep staging files after conversion (debugging).")
    parser.add_argument("--skip-final-verify", action="store_true", help="Skip the end-of-run artifact consistency check.")
    parser.add_argument("--verify-episode-trajectories", action="store_true", help="Re-parse every episode trajectory JSON in the final check.")
    # --- converter pass-throughs
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination HOST data directory (not needed for --list-only/--status).")
    parser.add_argument("--dataset-id", default="RobotTask", help="Dataset id for <id>_video_paths.json etc.")
    parser.add_argument("--main-view", default=DEFAULT_MAIN_VIEW, help="Source video key for the main camera.")
    parser.add_argument("--gripper-views", nargs="+", default=list(DEFAULT_GRIPPER_VIEWS), help="Source video keys for gripper cameras.")
    parser.add_argument("--no-gripper-view", action="store_true", help="Emit a single-view dataset (main view only).")
    parser.add_argument("--min-frames", type=int, default=24, help="Reject episodes shorter than alignment CONFIG.TRAIN.NUM_FRAMES.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 1..100.")
    parser.add_argument("--output-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"), help="Output image size HEIGHT WIDTH.")
    parser.add_argument("--short-instructions", action="store_true", help="Write only the task text before the first '|'.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing image sequences on re-conversion.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Parallel episode encodes within one converter.")
    parser.add_argument("--max-episodes", type=int, default=None, help="GLOBAL cap on episodes across all tars.")
    args = parser.parse_args()

    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in the range 1..100")
    for name, value in (
        ("--download-workers", args.download_workers),
        ("--extract-workers", args.extract_workers),
        ("--convert-workers", args.convert_workers),
        ("--parts-per-tar", args.parts_per_tar),
        ("--retries", args.retries),
        ("--workers", args.workers),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if not args.list_only and not args.status and args.output_dir is None:
        parser.error("--output-dir is required for a conversion run")
    return args


def probe_view_infos(args, infos, bucket, keys, state) -> list[dict]:
    """Stream meta/info.json out of the smallest few tars (videos never fetched)."""
    probed: list[dict] = []
    for key, size in sorted(keys, key=lambda item: item[1])[: args.probe_tars]:
        factory = obsio.make_stream_factory(infos, bucket, key, args.timeout)
        try:
            info = obsio.probe_info_json(factory, max_bytes=min(size, args.probe_max_stream_bytes))
        except RuntimeError as exc:
            print(f"WARNING: probe of {key} failed: {exc}")
            continue
        if info is None:
            print(f"WARNING: {key}: no meta/info.json found within the probe cap")
            continue
        probed.append(info)
        tar = state.add_tar(key, size)
        state.update(tar, info_json=info)
    return probed


def resolve_run_view_spec(args, infos, bucket, keys, state) -> dict:
    """Resolve the run-level view spec: explicit flags > stored spec > probing."""
    if viewspec.views_explicitly_set(args):
        return viewspec.resolve_view_spec([], args)
    if not args.force_respec and state.view_spec:
        spec = state.view_spec
        print(f"reusing stored view spec ({spec.get('source')}): {spec.get('main_view')}")
        return spec
    probed = probe_view_infos(args, infos, bucket, keys, state)
    spec = viewspec.resolve_view_spec(probed, args)
    state.view_spec = spec
    state.save()
    return spec


def main() -> int:
    args = parse_args()
    obs = obsio.import_obs(args.obs_sdk_path)
    infos = obsio.load_infos(Path(args.config))
    bucket = infos["OBS_BUCKET"]
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir is None:
        staging_dir = args.staging_dir or Path("output/obs_streaming.staging")
    else:
        staging_dir = args.staging_dir or output_dir.parent / f"{output_dir.name}.staging"
    staging = StagingPaths(staging_dir.resolve())
    state = PipelineState(staging.root / "obs_pipeline_state.json")
    keys_cache = staging.root / ".obs_tar_keys.json"

    try:
        client = obs.ObsClient(
            access_key_id=infos["OBS_AK"],
            secret_access_key=infos["OBS_SK"],
            server=infos["OBS_ENDPOINT"],
            timeout=args.timeout,
        )
        try:
            if args.list_only:
                keys = obsio.load_tar_keys(client, bucket, args.prefix, keys_cache, args.refresh_keys)
                total = sum(size for _, size in keys)
                print(f"{len(keys)} tar.gz objects under {args.prefix!r} ({total / 1024 ** 3:.2f} GiB)")
                for key, size in sorted(keys, key=lambda item: item[1]):
                    print(f"  {size / 1024 ** 3:10.2f} GiB  {key}")
                return 0
            if args.status:
                state.print_status()
                return 0

            require_ffmpeg()
            keys = obsio.load_tar_keys(client, bucket, args.prefix, keys_cache, args.refresh_keys)
        finally:
            client.close()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        staging.ensure_dirs()
        output_dir.mkdir(parents=True, exist_ok=True)

        spec = resolve_run_view_spec(args, infos, bucket, keys, state)
        viewspec.apply_view_spec(args, spec)
        if args.output_size is not None:
            print(f"output image size: {args.output_size[0]}x{args.output_size[1]}")
        print(
            f"dataset id: {args.dataset_id} | views: {args.main_view} / "
            f"{' '.join(args.gripper_views) if not args.no_gripper_view else '(none)'} | "
            f"output: {output_dir} | staging: {staging.root}"
        )

        # One dataset across all tars: rebuild the shared writer/accumulator.
        writer, accumulator = rebuild_writer_and_accumulator(output_dir, args, state)
        sweep_leftovers(staging, state, args.keep_extracted)

        # The global --max-episodes cap lives in the scheduler; per-tar
        # conversion calls must not apply it (iter_task_units would otherwise
        # cap each tar individually).
        episode_cap = args.max_episodes
        args.max_episodes = None

        pipeline = StreamingPipeline(
            args, infos, bucket, keys, state, staging, output_dir, writer,
            accumulator, episode_cap, committed_episodes=len(writer.video_paths),
        )
        if args.dry_run:
            pipeline.describe_schedule()
            return 0
        ok = pipeline.run()

        if not args.skip_final_verify:
            from convert_lerobot.outputs import verify_dataset_artifacts

            verify_dataset_artifacts(
                output_dir, args.dataset_id,
                verify_episode_trajectories=args.verify_episode_trajectories,
            )
        return 0 if ok else 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
