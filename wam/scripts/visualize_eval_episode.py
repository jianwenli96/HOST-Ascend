"""
Visualization script for SelfGroundedPredictor evaluation episodes.

Usage:
    python scripts/visualize_eval_episode.py <eval_json_path> [options]

    --output <mp4_path>       Output path (default: <cwd>/<json_stem>.mp4)
    --fps <int>               Output FPS (default: 8)
    --cam-mapping-dir <dir>   Camera mapping JSON dir (default: /open_data/cgy/cam_mapping/2views)
    --height <px>             Per-camera frame height in pixels (default: 224)

The eval JSON must contain:
    "agent_episode_dir"  – agent rollout path (may have "video://" prefix)
    "task_episode_dir"   – reference task episode path

Output video has 3 rows:
    Row 1: Agent cameras | Task cameras (progress-matched)
    Row 2: Agent cameras | Agent's eval-ref cameras (from task_paths_eval.json, progress-matched)
    Row 3: Task cameras  | Task's eval-ref cameras  (from task_paths_eval.json, progress-matched)

Each row has a progress bar band at its bottom.
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_MACRO_BLOCK = 16  # libx264 + yuv420p requires dimensions divisible by 16

# ─── Video I/O ───────────────────────────────────────────────────────────────

def _pad_to_macroblock(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = (-h) % _MACRO_BLOCK
    pad_w = (-w) % _MACRO_BLOCK
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_mp4(frames, path: str, fps: int = 8):
    """Write an iterable of PIL Images (or HxWx3 uint8 arrays) to an MP4 via ffmpeg."""
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)

    frame_iter = iter(frames)
    try:
        first = next(frame_iter)
    except StopIteration:
        log.warning("No frames to write.")
        return

    def to_array(f) -> np.ndarray:
        if isinstance(f, Image.Image):
            return np.array(f.convert("RGB"), dtype=np.uint8)
        return np.asarray(f, dtype=np.uint8)

    first_arr = _pad_to_macroblock(to_array(first))
    h, w = first_arr.shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(max(fps, 1)), "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        proc.stdin.write(first_arr.tobytes())
        for f in frame_iter:
            arr = _pad_to_macroblock(to_array(f))
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        log.error("ffmpeg process closed unexpectedly. Re-run with stderr enabled for details.")
        proc.wait()
        raise


# ─── Path utilities ───────────────────────────────────────────────────────────

def strip_video_prefix(path: str) -> str:
    if path.startswith("video://"):
        return path[len("video://"):]
    return path


def extract_dataset_name(episode_dir: str) -> str | None:
    """Extract the 5-digit dataset folder from the path (e.g. '10042')."""
    for part in Path(episode_dir).parts:
        if re.match(r"^\d{5}$", part):
            return part
    return None


def get_path_transforms(dataset_name: str | None) -> dict:
    """
    Return {old_prefix: new_prefix} for the given dataset.
    Tries to import from the project; falls back to a hardcoded default.
    """
    if dataset_name is None:
        return {}
    try:
        _src = osp.join(osp.dirname(__file__), "..", "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from self_grounded_prediction.datasets.custom.path_transforms_config import get_path_transforms as _gpt
        return _gpt(dataset_name)
    except Exception:
        # Common default for x2robot datasets
        return {"/x2robot_data/zhengwei": "/open_data/cgy/anns/real_data_new"}


def apply_path_transform(path: str, transforms: dict) -> str:
    """Apply all {old: new} prefix substitutions to path."""
    for old, new in transforms.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


# ─── Camera mapping ───────────────────────────────────────────────────────────

def load_cam_mapping(cam_mapping_dir: str, dataset_name: str) -> dict:
    mapping_path = osp.join(cam_mapping_dir, f"{dataset_name}_cam_mapping.json")
    if not osp.exists(mapping_path):
        log.warning(f"Camera mapping not found: {mapping_path}. Will auto-detect cameras.")
        return {}
    with open(mapping_path) as f:
        return json.load(f)


def get_cam_list(cam_mapping: dict, episode_dir: str) -> list[str] | None:
    """Lookup camera list by task-level key; returns None if not found."""
    if not cam_mapping:
        return None
    task_path = osp.normpath(osp.dirname(episode_dir.rstrip("/")))
    if task_path in cam_mapping:
        return cam_mapping[task_path]
    log.warning(f"Task path '{task_path}' not in cam_mapping. Will auto-detect cameras.")
    return None


def auto_detect_cameras(episode_dir: str) -> list[str]:
    """Find camera names from .mp4 files in episode_dir (excluding 'preview')."""
    try:
        mp4s = sorted(p.stem for p in Path(episode_dir).iterdir()
                      if p.suffix == ".mp4" and p.stem != "preview")
        if mp4s:
            return mp4s
        subdirs = sorted(p.name for p in Path(episode_dir).iterdir() if p.is_dir())
        return subdirs
    except FileNotFoundError:
        return []


# ─── Video / frame loading ────────────────────────────────────────────────────

def load_video_frames(episode_dir: str, cameras: list[str]) -> dict[str, np.ndarray]:
    """
    Load all frames for each camera. Returns {cam_name: np.ndarray [T, H, W, 3]}.
    Uses decord for MP4s; PIL for image directories.
    """
    try:
        import decord
        _decord_ok = True
    except ImportError:
        _decord_ok = False
        log.warning("decord not available; PIL fallback for image dirs only.")

    result: dict[str, np.ndarray] = {}
    for cam in cameras:
        mp4_path = osp.join(episode_dir, f"{cam}.mp4")
        img_dir  = osp.join(episode_dir, cam)

        if osp.isfile(mp4_path):
            if _decord_ok:
                vr = decord.VideoReader(mp4_path, ctx=decord.cpu(0))
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T, H, W, 3]
            else:
                log.error(f"decord required to read MP4: {mp4_path}")
                continue
            result[cam] = frames
            log.info(f"  Loaded {len(frames)} frames from {mp4_path}")

        elif osp.isdir(img_dir):
            img_paths = sorted(
                [osp.join(img_dir, p) for p in os.listdir(img_dir)
                 if p.lower().endswith((".jpg", ".jpeg", ".png"))],
                key=lambda x: int(re.sub(r"\D", "", osp.basename(x)) or "0"),
            )
            if not img_paths:
                log.warning(f"  No images in {img_dir}")
                continue
            frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in img_paths])
            result[cam] = frames
            log.info(f"  Loaded {len(frames)} frames from {img_dir}/")

        else:
            log.warning(f"  Camera '{cam}' not found in {episode_dir}")

    return result


# ─── Progress info ────────────────────────────────────────────────────────────

def load_progress_info(episode_dir: str, transforms: dict | None = None) -> list[float] | None:
    """
    Load aligned_progress from info_dtw.json (or info.json).
    Searches:
      1. episode_dir itself
      2. parent of episode_dir
      3. path-transformed variants of the above
    Returns sorted list of progress floats, or None if not found.
    """
    transforms = transforms or {}

    def _candidate_dirs(base: str) -> list[str]:
        dirs = [base, osp.dirname(base.rstrip("/"))]
        transformed = [apply_path_transform(d, transforms) for d in dirs]
        # Deduplicate while preserving order
        seen = set()
        result = []
        for d in dirs + transformed:
            if d not in seen:
                seen.add(d)
                result.append(d)
        return result

    for search_dir in _candidate_dirs(episode_dir):
        if osp.exists(osp.join(search_dir, "bad_info_dtw.json")):
            log.warning(f"bad_info_dtw.json in {search_dir}; treating as bad episode.")
            return None

        for fname in ("info_dtw.json", "info.json"):
            info_path = osp.join(search_dir, fname)
            if not osp.exists(info_path):
                continue
            with open(info_path) as f:
                data = json.load(f)
            if "aligned_progress" not in data:
                log.warning(f"'aligned_progress' missing in {info_path}")
                continue
            items = sorted((int(k), float(v)) for k, v in data["aligned_progress"].items())
            progress = [v for _, v in items]
            log.info(f"  Loaded {len(progress)} progress values from {info_path}")
            return progress

    log.warning(f"No progress info found for: {episode_dir}")
    return None


def uniform_progress(n_frames: int) -> list[float]:
    if n_frames <= 1:
        return [0.0]
    return [i / (n_frames - 1) for i in range(n_frames)]


def align_progress(progress: list[float], n_frames: int) -> list[float]:
    if len(progress) == n_frames:
        return progress
    if len(progress) > n_frames:
        log.warning(f"Progress length {len(progress)} > frames {n_frames}; truncating.")
        return progress[:n_frames]
    log.warning(f"Progress length {len(progress)} < frames {n_frames}; extending.")
    return progress + uniform_progress(n_frames - len(progress))


# ─── task_paths_eval.json ─────────────────────────────────────────────────────

def load_task_paths_eval(episode_dir: str, transforms: dict | None = None) -> str | None:
    """
    Load the first episode path from task_paths_eval.json["same"].
    Searches episode_dir and its parent, preferring the path-transformed location
    (where annotation files like task_paths_eval.json are stored).
    Returns the episode directory string (original path format), or None if not found.
    """
    transforms = transforms or {}

    def _search_dirs(base: str) -> list[str]:
        dirs = [base, osp.dirname(base.rstrip("/"))]
        transformed = [apply_path_transform(d, transforms) for d in dirs]
        seen, result = set(), []
        # Prefer transformed dirs first (that's where annotation files live)
        for d in transformed + dirs:
            if d not in seen:
                seen.add(d)
                result.append(d)
        return result

    for search_dir in _search_dirs(episode_dir):
        json_path = osp.join(search_dir, "task_paths_eval.json")
        if not osp.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        candidates = data.get("same", [])
        if not candidates:
            log.warning(f"task_paths_eval.json has no 'same' entries in {search_dir}")
            return None
        chosen = candidates[0]
        log.info(f"  task_paths_eval.json → eval ref: {chosen}")
        return chosen

    log.warning(f"task_paths_eval.json not found for: {episode_dir}")
    return None


# ─── Frame correspondence ─────────────────────────────────────────────────────

def match_frames(src_progress: list[float], ref_progress: list[float]) -> list[int]:
    """For each source frame, find the ref frame with the closest progress."""
    ref_arr = np.array(ref_progress)
    return [int(np.argmin(np.abs(ref_arr - p))) for p in src_progress]


# ─── Visualization helpers ────────────────────────────────────────────────────

def _get_font(size: int = 16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if osp.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


_FONT = None

def _font():
    global _FONT
    if _FONT is None:
        _FONT = _get_font(16)
    return _FONT


def _draw_progress_bar(draw, x, y, width, progress, label, bar_color):
    font = _font()
    draw.text((x, y), label, fill=(220, 220, 220), font=font)
    bar_y = y + 18
    draw.rectangle([x, bar_y, x + width, bar_y + 12], fill=(60, 60, 60))
    filled_w = int(width * max(0.0, min(1.0, progress)))
    if filled_w > 0:
        draw.rectangle([x, bar_y, x + filled_w, bar_y + 12], fill=bar_color)
    draw.text((x + width + 6, bar_y - 2), f"{progress:.3f}", fill=(220, 220, 220), font=font)


def _resize_to_height(arr: np.ndarray, target_h: int) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    return img.resize((new_w, target_h), Image.BILINEAR)


def build_row(
    left_frames:  list[np.ndarray],
    right_frames: list[np.ndarray],
    left_progress:  float,
    right_progress: float,
    left_label:   str,
    right_label:  str,
    left_color:   tuple,
    right_color:  tuple,
    target_h: int,
    bar_h:    int = 50,
    divider_w: int = 4,
) -> Image.Image:
    """
    Build one row image:
        [left_cam1 | left_cam2 | divider | right_cam1 | right_cam2]
        [left progress bar            | right progress bar         ]
    """
    left_imgs  = [_resize_to_height(f, target_h) for f in left_frames]
    right_imgs = [_resize_to_height(f, target_h) for f in right_frames]

    left_w  = sum(img.width for img in left_imgs)
    right_w = sum(img.width for img in right_imgs)
    total_w = left_w + divider_w + right_w
    total_h = target_h + bar_h

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))

    # Paste left cameras
    cx = 0
    for img in left_imgs:
        canvas.paste(img, (cx, 0))
        cx += img.width

    # Divider
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([cx, 0, cx + divider_w - 1, target_h - 1], fill=(200, 200, 200))
    cx += divider_w

    # Paste right cameras
    for img in right_imgs:
        canvas.paste(img, (cx, 0))
        cx += img.width

    # Progress bars — each takes half the width
    half_bar_w = (total_w - 220) // 2
    bar_y = target_h + 4
    _draw_progress_bar(draw, 4,              bar_y, half_bar_w, left_progress,  left_label,  left_color)
    _draw_progress_bar(draw, total_w // 2,   bar_y, half_bar_w, right_progress, right_label, right_color)

    return canvas


def stack_rows(rows: list[Image.Image], gap: int = 2) -> Image.Image:
    """Stack row images vertically, padding narrower rows to the max width."""
    max_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (max_w, total_h), color=(15, 15, 15))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    return canvas


# ─── Episode helper ───────────────────────────────────────────────────────────

class Episode:
    """Holds all loaded data for one episode."""
    def __init__(self, label: str, episode_dir: str, cameras: list[str], transforms: dict):
        self.label = label
        self.episode_dir = episode_dir
        log.info(f"Loading video frames for [{label}]: {episode_dir}")
        self.videos = load_video_frames(episode_dir, cameras)
        self.cams_ok = [c for c in cameras if c in self.videos]
        if not self.cams_ok:
            log.error(f"No camera videos loaded for [{label}]: {episode_dir}")
        self.n_frames = len(self.videos[self.cams_ok[0]]) if self.cams_ok else 0

        log.info(f"Loading progress info for [{label}]...")
        raw = load_progress_info(episode_dir, transforms)
        if raw is None:
            log.warning(f"  Using uniform progress for [{label}].")
            raw = uniform_progress(self.n_frames)
        self.progress = align_progress(raw, self.n_frames)

    def frames_at(self, idx: int) -> list[np.ndarray]:
        """Return one frame per camera at the given index (clamped)."""
        return [self.videos[c][min(idx, len(self.videos[c]) - 1)] for c in self.cams_ok]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize SelfGroundedPredictor eval episode pair (3 rows).")
    parser.add_argument("json_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cam-mapping-dir", default="/open_data/cgy/cam_mapping/2views")
    parser.add_argument("--height", type=int, default=224)
    args = parser.parse_args()

    json_path = osp.abspath(args.json_path)
    if not osp.isfile(json_path):
        log.error(f"JSON not found: {json_path}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        stem = osp.splitext(osp.basename(json_path))[0]
        output_path = osp.join(os.getcwd(), stem + ".mp4")

    # ── 1. Load eval JSON ──────────────────────────────────────────────────────
    with open(json_path) as f:
        eval_data = json.load(f)

    agent_dir = strip_video_prefix(eval_data["agent_episode_dir"])
    task_dir  = eval_data["task_episode_dir"]
    log.info(f"Agent episode: {agent_dir}")
    log.info(f"Task  episode: {task_dir}")

    # ── 2. Dataset + path transforms ──────────────────────────────────────────
    dataset_name = extract_dataset_name(agent_dir) or extract_dataset_name(task_dir)
    if dataset_name is None:
        log.warning("Could not extract dataset name from episode paths.")
    transforms = get_path_transforms(dataset_name)
    log.info(f"Dataset: {dataset_name}, path transforms: {transforms}")

    # ── 3. Camera mapping ──────────────────────────────────────────────────────
    cam_mapping: dict = {}
    if dataset_name:
        cam_mapping = load_cam_mapping(args.cam_mapping_dir, dataset_name)

    def get_cams(ep_dir: str) -> list[str]:
        cams = get_cam_list(cam_mapping, ep_dir)
        if cams is None:
            cams = auto_detect_cameras(ep_dir)
            log.info(f"  Auto-detected cameras for {ep_dir}: {cams}")
        else:
            log.info(f"  Cam-mapping cameras for {ep_dir}: {cams}")
        return cams

    agent_cams = get_cams(agent_dir)
    task_cams  = get_cams(task_dir)

    if not agent_cams:
        log.error(f"No cameras for agent episode: {agent_dir}")
        sys.exit(1)
    if not task_cams:
        log.error(f"No cameras for task episode: {task_dir}")
        sys.exit(1)

    # ── 4. Load eval-ref episodes from task_paths_eval.json ───────────────────
    log.info("Loading task_paths_eval.json for agent episode...")
    agent_eval_ref_dir = load_task_paths_eval(agent_dir, transforms)
    log.info("Loading task_paths_eval.json for task episode...")
    task_eval_ref_dir  = load_task_paths_eval(task_dir, transforms)

    # Camera lists for eval-ref episodes
    agent_eval_ref_cams: list[str] = []
    task_eval_ref_cams:  list[str] = []
    if agent_eval_ref_dir:
        agent_eval_ref_cams = get_cams(agent_eval_ref_dir)
    if task_eval_ref_dir:
        task_eval_ref_cams = get_cams(task_eval_ref_dir)

    # ── 5. Load all episodes ───────────────────────────────────────────────────
    agent_ep = Episode("Agent",     agent_dir,            agent_cams,          transforms)
    task_ep  = Episode("Task",      task_dir,             task_cams,           transforms)

    agent_eval_ref_ep: Episode | None = None
    task_eval_ref_ep:  Episode | None = None

    if agent_eval_ref_dir and agent_eval_ref_cams:
        agent_eval_ref_ep = Episode("Agent-EvalRef", agent_eval_ref_dir, agent_eval_ref_cams, transforms)
    else:
        log.warning("Skipping agent eval-ref (not found or no cameras).")

    # Reuse the same Episode object if both ref dirs point to the same path
    if task_eval_ref_dir and task_eval_ref_cams:
        if task_eval_ref_dir == agent_eval_ref_dir:
            task_eval_ref_ep = agent_eval_ref_ep
            log.info("Task eval-ref is the same episode as agent eval-ref; reusing.")
        else:
            task_eval_ref_ep = Episode("Task-EvalRef", task_eval_ref_dir, task_eval_ref_cams, transforms)
    else:
        log.warning("Skipping task eval-ref (not found or no cameras).")

    # ── 6. Frame correspondence ────────────────────────────────────────────────
    # Row 1: agent frame i → task frame
    task_matched = match_frames(agent_ep.progress, task_ep.progress)

    # Row 2: agent frame i → agent_eval_ref frame
    agent_ref_matched: list[int] = []
    if agent_eval_ref_ep:
        agent_ref_matched = match_frames(agent_ep.progress, agent_eval_ref_ep.progress)

    # Row 3: task frame (t_idx) → task_eval_ref frame
    task_ref_matched: list[int] = []
    if task_eval_ref_ep:
        # Match from task_ep's progress at the already-matched task frames
        # (so row 3 tracks the same task moment as row 1's right side)
        task_ref_matched = match_frames(task_ep.progress, task_eval_ref_ep.progress)

    # ── 7. Build and save video ────────────────────────────────────────────────
    n_out = agent_ep.n_frames
    log.info(f"Building visualization ({n_out} frames, 3 rows)...")

    def frame_generator():
        for i in range(n_out):
            t_idx = task_matched[i]

            # Row 1: Agent vs Task
            row1 = build_row(
                left_frames   = agent_ep.frames_at(i),
                right_frames  = task_ep.frames_at(t_idx),
                left_progress  = agent_ep.progress[i],
                right_progress = task_ep.progress[t_idx],
                left_label    = "Agent",
                right_label   = "Task",
                left_color    = (100, 180, 255),
                right_color   = (100, 255, 150),
                target_h      = args.height,
            )

            rows = [row1]

            # Row 2: Agent vs Agent's eval-ref
            if agent_eval_ref_ep and agent_ref_matched:
                ar_idx = agent_ref_matched[i]
                row2 = build_row(
                    left_frames   = agent_ep.frames_at(i),
                    right_frames  = agent_eval_ref_ep.frames_at(ar_idx),
                    left_progress  = agent_ep.progress[i],
                    right_progress = agent_eval_ref_ep.progress[ar_idx],
                    left_label    = "Agent",
                    right_label   = "Agent-EvalRef",
                    left_color    = (100, 180, 255),
                    right_color   = (255, 200, 80),
                    target_h      = args.height,
                )
                rows.append(row2)

            # Row 3: Task vs Task's eval-ref
            if task_eval_ref_ep and task_ref_matched:
                tr_idx = task_ref_matched[t_idx]
                row3 = build_row(
                    left_frames   = task_ep.frames_at(t_idx),
                    right_frames  = task_eval_ref_ep.frames_at(tr_idx),
                    left_progress  = task_ep.progress[t_idx],
                    right_progress = task_eval_ref_ep.progress[tr_idx],
                    left_label    = "Task",
                    right_label   = "Task-EvalRef",
                    left_color    = (100, 255, 150),
                    right_color   = (255, 200, 80),
                    target_h      = args.height,
                )
                rows.append(row3)

            yield stack_rows(rows)

    log.info(f"Saving to {output_path} at {args.fps} fps...")
    save_mp4(frame_generator(), output_path, fps=args.fps)
    log.info("Done.")


if __name__ == "__main__":
    main()
