import json
import os
import re
import glob
import logging
import sys
import numpy as np
import argparse
import cv2
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

_HIGH_LOSS_SAMPLES_JSONL = re.compile(r"^high_loss_samples_(\d+)\.jsonl$")


def resolve_log_jsonl_paths(log_path: str) -> list[str]:
    """Resolve a single jsonl file or a directory of high_loss_samples_<N>.jsonl files."""
    p = Path(log_path).resolve()
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        matched: list[tuple[int, str]] = []
        for child in p.iterdir():
            if not child.is_file():
                continue
            m = _HIGH_LOSS_SAMPLES_JSONL.match(child.name)
            if m:
                matched.append((int(m.group(1)), str(child.resolve())))
        matched.sort(key=lambda x: x[0])
        return [path for _, path in matched]
    return []


def parse_frame_path(path):
    """Parse a frame path into (episode_dir, image_dir, frame_index)."""
    try:
        if path.startswith('video://'):
            stripped = path[len('video://'):]
            parts = stripped.split('::')
            mp4_path = parts[0]
            frame_idx = int(parts[1])
            media_dir = str(Path(mp4_path).parent)
            return media_dir, media_dir, frame_idx

        path_obj = Path(path)
        if path_obj.parent.name in ('images', 'rgb_static'):
            base_dir = path_obj.parent.parent
            img_dir = path_obj.parent
        else:
            base_dir = path_obj.parent
            img_dir = path_obj.parent
        frame_idx = int(path_obj.stem)
        return str(base_dir), str(img_dir), frame_idx
    except Exception:
        return None, None, None


def parse_video_segment_path(video_path_str):
    """Parse a video segment string like base:segment:start-end."""
    if not video_path_str:
        return video_path_str, None, None, None
    parts = video_path_str.rsplit(':', 2)
    if len(parts) == 3:
        try:
            start, end = map(int, parts[2].split('-'))
            return parts[0], parts[1], start, end
        except ValueError:
            pass
    return video_path_str, None, None, None


def count_total_frames(directory):
    """Count frames from images or the first mp4 in a directory."""
    types = ('*.jpg', '*.png', '*.jpeg')
    files = []
    for t in types:
        files.extend(glob.glob(os.path.join(directory, t)))
    if files:
        return len(files)

    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
    if mp4_files:
        cap = cv2.VideoCapture(mp4_files[0])
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count > 0:
            return count
        logging.warning(f"OpenCV reported 0 frames for mp4: {mp4_files[0]}")
    return 0


def process_jsonl(log_file):
    """Read a DTW alignment jsonl log and write info_dtw.json into each episode directory."""
    logging.info(f"Reading log file: {log_file}")

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logging.info(f"Found {len(lines)} records. Processing...")

    skipped_loss = 0
    skipped_range = 0
    skipped_causal = 0
    skipped_jump = 0
    skipped_other = 0
    written = 0

    for line in tqdm(lines):
        try:
            record = json.loads(line)
        except Exception:
            skipped_other += 1
            continue

        raw_main_paths = record.get('frame_paths', [])
        raw_ref_paths = record.get('ref_frame_paths', [])
        alignment = record.get('forward_argmax_indices', [])

        if not raw_main_paths or not raw_ref_paths or not alignment:
            skipped_other += 1
            continue

        # Subsample every 4th frame starting at index 3
        extracted_main = raw_main_paths[3::4]
        extracted_ref = raw_ref_paths[3::4]

        limit = min(len(extracted_main), len(alignment))
        if limit == 0:
            skipped_other += 1
            continue

        alignment_slice = alignment[:limit]

        loss = record.get('loss', float('inf'))
        if loss > 0.15:
            skipped_loss += 1
            continue

        len_ref = len(extracted_ref)
        align_max = max(alignment_slice)
        align_min = min(alignment_slice)
        if (align_max - align_min) <= 0.8 * len_ref:
            skipped_range += 1
            continue

        if any(alignment_slice[i] < alignment_slice[i - 1] for i in range(1, limit)):
            skipped_causal += 1
            continue

        max_jump = 7.0 / 24.0 * len_ref
        if any(alignment_slice[i] - alignment_slice[i - 1] > max_jump for i in range(1, limit)):
            skipped_jump += 1
            continue

        is_video = raw_main_paths[0].startswith('video://')

        if is_video:
            main_vp = record.get('main_video_path') or ''
            main_base, main_seg_idx, main_start, main_end = parse_video_segment_path(main_vp)
            if not main_base:
                _, main_base, _ = parse_frame_path(raw_main_paths[0])

            if main_seg_idx is not None:
                main_root_dir = os.path.join(main_base, main_seg_idx)
                all_main_indices = list(range(main_start, main_end + 1))
                if not main_base or not os.path.exists(main_base):
                    skipped_other += 1
                    continue
            else:
                main_root_dir = main_base
                if not main_root_dir or not os.path.exists(main_root_dir):
                    skipped_other += 1
                    continue
                total_main_frames = count_total_frames(main_root_dir)
                if total_main_frames == 0:
                    skipped_other += 1
                    continue
                all_main_indices = list(range(total_main_frames))
        else:
            main_root_dir, main_img_dir, _ = parse_frame_path(raw_main_paths[0])
            if not main_root_dir or not os.path.exists(main_root_dir):
                skipped_other += 1
                continue
            main_files = (
                glob.glob(os.path.join(main_img_dir, '*.jpg')) +
                glob.glob(os.path.join(main_img_dir, '*.png'))
            )
            if not main_files:
                skipped_other += 1
                continue
            all_main_indices = sorted([int(Path(p).stem) for p in main_files])

        ref_start_frame = 0

        if is_video:
            ref_vp = record.get('ref_video_path', '')
            ref_root_str = ref_vp
            ref_base, ref_seg_idx, ref_start, ref_end = parse_video_segment_path(ref_vp)

            if ref_seg_idx is not None:
                total_ref_frames = ref_end - ref_start + 1
                ref_start_frame = ref_start
            else:
                if not ref_base:
                    for rp in extracted_ref:
                        if 'DUSTBIN' not in rp:
                            _, ref_base, _ = parse_frame_path(rp)
                            if ref_base:
                                break
                if not ref_base:
                    skipped_other += 1
                    continue
                total_ref_frames = count_total_frames(ref_base)
                if total_ref_frames == 0:
                    skipped_other += 1
                    continue
        else:
            ref_root_str = ''
            ref_media_dir_for_count = None
            for rp in extracted_ref:
                if 'DUSTBIN' not in rp:
                    ref_base_img, ref_img_dir, _ = parse_frame_path(rp)
                    if ref_base_img and os.path.exists(ref_img_dir):
                        ref_root_str = str(ref_base_img)
                        ref_media_dir_for_count = ref_img_dir
                        break

            if not ref_media_dir_for_count:
                skipped_other += 1
                continue

            total_ref_frames = count_total_frames(ref_media_dir_for_count)
            if total_ref_frames == 0:
                skipped_other += 1
                continue

        sparse_progress = {}
        prev_alignment_val = None

        for i in range(limit):
            cur_align = alignment_slice[i]

            if prev_alignment_val is not None and cur_align == prev_alignment_val:
                continue
            prev_alignment_val = cur_align

            _, _, main_idx = parse_frame_path(extracted_main[i])
            if main_idx is None:
                continue

            if cur_align >= len(extracted_ref):
                continue

            target_ref_path = extracted_ref[cur_align]
            if 'DUSTBIN' in target_ref_path:
                continue

            _, _, ref_frame_num = parse_frame_path(target_ref_path)
            if ref_frame_num is not None:
                progress = (ref_frame_num - ref_start_frame + 1) / total_ref_frames
                sparse_progress[main_idx] = progress

        if not sparse_progress:
            skipped_other += 1
            continue

        known_indices = sorted(sparse_progress.keys())
        known_values = [sparse_progress[k] for k in known_indices]

        final_progress_dict = {}
        for target_idx in all_main_indices:
            if target_idx in sparse_progress:
                final_progress_dict[str(target_idx)] = sparse_progress[target_idx]
            else:
                interp_val = np.interp(target_idx, known_indices, known_values)
                final_progress_dict[str(target_idx)] = float(f"{interp_val:.5f}")

        info_data = {
            "ref_path": ref_root_str,
            "aligned_progress": final_progress_dict,
        }

        info_path = os.path.join(main_root_dir, "info_dtw.json")

        try:
            os.makedirs(main_root_dir, exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as json_file:
                json.dump(info_data, json_file, indent=4)
            written += 1
        except Exception as e:
            logging.warning(f"Failed to write {info_path}: {e}")

    logging.info(
        f"Done. written={written}, "
        f"skipped(loss)={skipped_loss}, "
        f"skipped(range)={skipped_range}, "
        f"skipped(causal)={skipped_causal}, "
        f"skipped(jump)={skipped_jump}, "
        f"skipped(other)={skipped_other}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate info_dtw.json from DTW alignment logs and write directly to episode directories.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        required=True,
        help='Path to a jsonl log file, or a directory containing high_loss_samples_<N>.jsonl files',
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        logging.error(f"Path not found: {args.log_file}")
        sys.exit(1)

    paths = resolve_log_jsonl_paths(args.log_file)
    if not paths:
        if os.path.isdir(args.log_file):
            logging.error(
                f"No high_loss_samples_<N>.jsonl files found in directory: {args.log_file}"
            )
        else:
            logging.error(f"Invalid log path: {args.log_file}")
        sys.exit(1)

    logging.info(f"Will process {len(paths)} jsonl file(s): {[Path(p).name for p in paths]}")
    for log_path in paths:
        logging.info(f"Processing: {log_path}")
        process_jsonl(log_path)
