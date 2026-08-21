"""Stage B of the LeRobot converter: frame decoding and image-sequence writing.

Decodes the source MP4s (AV1-coded, so via the ffmpeg CLI) into per-view JPEG
sequences, with ffprobe validation of the source videos and of the written
frames.  ``encode_views`` operates on a single episode and keeps all state in
per-episode directories, so a task's episodes can be encoded in parallel via
``encode_task``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .source import Episode, natural_key


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(
            "Missing required tools: " + ", ".join(missing) + ". The source videos are "
            "AV1-coded, so frames are extracted with the ffmpeg CLI. Install it (e.g. "
            "'brew install ffmpeg') or run on a machine where it is available."
        )


def jpeg_quality_to_qv(quality: int) -> int:
    """Map 1..100 JPEG quality to ffmpeg mjpeg -q:v 31..1 (95 -> 3)."""
    return 31 - round((quality - 1) * 30 / 99)


def ffprobe_json(path: Path, show_entries: str) -> list[dict]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        show_entries,
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe produced unparseable output for {path}") from exc


def validate_source_video(video: Path, view_dir: str, episode: Episode) -> None:
    streams = ffprobe_json(video, "stream=nb_frames,width,height,codec_name")
    if not streams:
        raise ValueError(f"{video}: no video stream found")
    stream = streams[0]
    expected_shape = episode.view_shapes[view_dir]
    actual_shape = (int(stream.get("height", 0)), int(stream.get("width", 0)))
    if actual_shape != expected_shape:
        raise ValueError(
            f"{video}: expected frame shape {expected_shape}, got {actual_shape} "
            f"(codec {stream.get('codec_name')})"
        )
    try:
        nb_frames = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError):
        return  # container does not store a frame count; the JPEG count check still applies
    if nb_frames != episode.frame_count:
        raise ValueError(
            f"{video}: video has {nb_frames} frames but the Parquet trajectory has "
            f"{episode.frame_count}; refusing to silently truncate"
        )


def assert_images_match(path: Path, frame_count: int, shape: tuple[int, int]) -> None:
    if not path.is_dir():
        raise ValueError(f"Missing output image directory: {path}")
    files = sorted(path.glob("*.jpg"), key=natural_key)
    if len(files) != frame_count:
        raise ValueError(f"Invalid image count in {path}: expected {frame_count}, got {len(files)}")
    expected_names = [f"{index}.jpg" for index in range(frame_count)]
    if [file.name for file in files] != expected_names:
        raise ValueError(f"Unexpected image filenames in {path}; expected contiguous 0.jpg..N.jpg")
    height, width = shape
    for file in (files[0], files[len(files) // 2], files[-1]):
        streams = ffprobe_json(file, "stream=width,height")
        actual = None
        if streams:
            actual = (int(streams[0].get("height", 0)), int(streams[0].get("width", 0)))
        if actual != shape:
            raise ValueError(f"Invalid image {file}: expected {(height, width, 3)}, got {actual}")


def encode_views(episode: Episode, jpeg_quality: int, overwrite: bool, output_size: Optional[tuple[int, int]] = None) -> None:
    needs_write: dict[str, bool] = {}
    for view_dir, video in episode.videos.items():
        target = episode.episode_dir / view_dir
        if target.exists() and not overwrite:
            # Reuse check must validate against the shape this run would write
            # (--output-size may have resized the frames), not the source shape.
            reuse_shape = (
                output_size if output_size is not None else episode.view_shapes[view_dir]
            )
            assert_images_match(target, episode.frame_count, reuse_shape)
            needs_write[view_dir] = False
        else:
            needs_write[view_dir] = True

    stale_videos = [episode.episode_dir / f"{view_dir}.mp4" for view_dir in episode.videos]
    if not overwrite and any(path.exists() for path in stale_videos):
        raise ValueError(
            "Stale MP4 output conflicts with image-sequence loading; rerun with --overwrite "
            "to replace it"
        )

    if not any(needs_write.values()):
        print(f"[reuse] {episode.task_dir.name}/{episode.episode_name}")
        return

    temporary = {
        view_dir: target.with_name(f".{target.name}.converting")
        for view_dir, target in (
            (view_dir, episode.episode_dir / view_dir) for view_dir in episode.videos
        )
        if needs_write[view_dir]
    }
    try:
        for temp_path in temporary.values():
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True)

        for view_dir, temp_path in temporary.items():
            validate_source_video(episode.videos[view_dir], view_dir, episode)
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(episode.videos[view_dir]),
            ]
            if output_size is not None:
                height, width = output_size
                command.extend(["-vf", f"scale={width}:{height}"])
            command.extend([
                "-q:v",
                str(jpeg_quality_to_qv(jpeg_quality)),
                "-start_number",
                "0",
                str(temp_path / "%d.jpg"),
            ])
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed on {episode.videos[view_dir]}: {completed.stderr.strip()[-500:]}"
                )
            output_shape = output_size if output_size is not None else episode.view_shapes[view_dir]
            assert_images_match(temp_path, episode.frame_count, output_shape)

        for view_dir, temp_path in temporary.items():
            target = episode.episode_dir / view_dir
            if target.exists():
                shutil.rmtree(target)
            temp_path.rename(target)

        for stale_video in stale_videos:
            if stale_video.exists():
                stale_video.unlink()
        views = "/".join(episode.videos)
        if output_size is not None:
            output_shapes = {view_dir: output_size for view_dir in episode.videos}
        else:
            output_shapes = episode.view_shapes
        shapes = " ".join(f"{width}x{height}" for height, width in output_shapes.values())
        print(
            f"[write] {episode.task_dir.name}/{episode.episode_name}: "
            f"{episode.frame_count} JPEG frames per view ({views}), {shapes}"
        )
    finally:
        for temp_path in temporary.values():
            if temp_path.exists():
                shutil.rmtree(temp_path)


def encode_task(
    episodes: list[Episode],
    jpeg_quality: int,
    overwrite: bool,
    output_size: Optional[tuple[int, int]],
    workers: int,
) -> None:
    """Encode every episode of a task, in parallel across episodes.

    Each episode writes to its own directories (temporary names included), so
    the per-episode ``encode_views`` calls are independent.  ffmpeg/ffprobe
    run as subprocesses, which release the GIL while waiting, so threads are
    enough to overlap the decoding work.
    """
    if workers <= 1 or len(episodes) <= 1:
        for episode in episodes:
            encode_views(episode, jpeg_quality, overwrite, output_size)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(encode_views, episode, jpeg_quality, overwrite, output_size)
            for episode in episodes
        ]
        for future in futures:
            future.result()  # re-raise the first failure
