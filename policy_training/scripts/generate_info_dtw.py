#!/usr/bin/env python3
"""Generate info_dtw.json with linear progress for eval episodes.

For each episode in the video_paths JSON, counts frames from the first
available mp4 (or image directory), then writes info_dtw.json with
linear progress 0 → 1.

Usage:
    python scripts/generate_info_dtw.py \
        --video_paths /path/to/10042_video_paths.json \
        --views leftImg rightImg \
        [--overwrite]
"""
import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def count_frames(episode_dir, views):
    """Count frames from the first available view (mp4 or image dir)."""
    for view in views:
        mp4_path = os.path.join(episode_dir, f"{view}.mp4")
        img_dir = os.path.join(episode_dir, view)

        if os.path.exists(mp4_path):
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(mp4_path, num_threads=1)
            n = len(vr)
            del vr
            return n

        if os.path.isdir(img_dir):
            import glob
            files = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                    glob.glob(os.path.join(img_dir, "*.png"))
            return len(files)

    return None


def generate_info_dtw(episode_dir, n_frames):
    """Generate info_dtw.json with linear progress 0 → 1."""
    aligned_progress = {}
    for i in range(n_frames):
        progress = i / max(n_frames - 1, 1)
        aligned_progress[str(i)] = round(progress, 6)

    return {"aligned_progress": aligned_progress}


def main():
    parser = argparse.ArgumentParser(description="Generate info_dtw.json for eval episodes")
    parser.add_argument("--video_paths", required=True, help="Path to video_paths JSON file")
    parser.add_argument("--views", nargs="+", default=["leftImg", "rightImg"],
                        help="Camera view names for frame counting")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing info_dtw.json")
    args = parser.parse_args()

    with open(args.video_paths) as f:
        episodes = json.load(f)

    logger.info("Processing %d episodes from %s", len(episodes), args.video_paths)

    for ep_path in episodes:
        ep_dir = ep_path if isinstance(ep_path, str) else ep_path.get("video_dir", "")
        info_path = os.path.join(ep_dir, "info_dtw.json")

        if os.path.exists(info_path) and not args.overwrite:
            # Check if existing file matches frame count
            n_frames = count_frames(ep_dir, args.views)
            if n_frames is None:
                logger.warning("[SKIP] No video data in %s", ep_dir)
                continue
            with open(info_path) as f:
                existing = json.load(f)
            n_existing = len(existing.get("aligned_progress", {}))
            if n_existing == n_frames:
                logger.info("[OK] %s — %d frames, matches existing", os.path.basename(ep_dir), n_frames)
                continue
            else:
                logger.warning("[MISMATCH] %s — existing %d keys, actual %d frames, regenerating",
                               os.path.basename(ep_dir), n_existing, n_frames)

        n_frames = count_frames(ep_dir, args.views)
        if n_frames is None:
            logger.warning("[SKIP] No video data in %s", ep_dir)
            continue

        info_dtw = generate_info_dtw(ep_dir, n_frames)
        with open(info_path, "w") as f:
            json.dump(info_dtw, f, indent=2)

        logger.info("[WROTE] %s — %d frames", os.path.basename(ep_dir), n_frames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
