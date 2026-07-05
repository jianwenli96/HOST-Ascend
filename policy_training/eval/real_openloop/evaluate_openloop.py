"""
SelfGroundedPredictor open-loop evaluation script.

Chunk-based evaluation: for each episode, iterate through chunks of
action_frames steps, call infer_real(), compare predicted actions against
GT actions from the dataset.

Usage:
    python evaluate_openloop.py \
        --checkpoint_dir ./logs/real_joint_2cam_224_1e-4/2026-04-25_14-55-59 \
        --eval_data_path /open_data/.../10042_video_paths.json \
        --dataset_name 10042 \
        --views leftImg rightImg
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import random
import sys
import time
import traceback
from typing import Optional

from omegaconf import OmegaConf

import numpy as np

# Ensure project src is importable
_project_src = osp.join(osp.dirname(__file__), "..", "..", "src")
if _project_src not in sys.path:
    sys.path.insert(0, _project_src)
_eval_dir = osp.dirname(__file__)
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from eval_dataset import SelfGroundedPredictorEvalDataset
from self_grounded_predictor_eval import SelfGroundedPredictorEval
from action_utils import build_action_col_map, decode_6d_to_euler
from self_grounded_prediction.datasets.custom.path_transforms_config import get_path_transforms

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def mae_rmse(pred: np.ndarray, gt: np.ndarray):
    """Compute MAE and RMSE between pred and gt arrays."""
    diff = pred - gt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def compute_episode_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
) -> dict:
    """Compute per-episode action metrics.

    Args:
        preds: [N, D] predicted actions (raw, denormalized)
        gts:   [N, D] GT actions (raw, denormalized)

    Returns:
        dict with metric names → float values
    """
    mae, rmse = mae_rmse(preds, gts)
    metrics = {
        "action_mae_raw": mae,
        "action_rmse_raw": rmse,
        "n_steps": len(preds),
    }

    D = preds.shape[1]
    # Per-arm metrics for 14D dual-arm (pos[3] + rot[3] + grip[1]) × 2
    if D == 14:
        for prefix, start in [("left", 0), ("right", 7)]:
            pos_err = np.linalg.norm(
                preds[:, start:start + 3] - gts[:, start:start + 3], axis=-1
            ).mean()
            rot_err = np.linalg.norm(
                preds[:, start + 3:start + 6] - gts[:, start + 3:start + 6], axis=-1
            ).mean()
            grip_acc = float(np.mean(
                np.sign(preds[:, start + 6]) == np.sign(gts[:, start + 6])
            ))
            metrics[f"{prefix}_pos_err_m"] = float(pos_err)
            metrics[f"{prefix}_rot_err_rad"] = float(rot_err)
            metrics[f"{prefix}_rot_err_deg"] = float(rot_err * 180.0 / np.pi)
            metrics[f"{prefix}_grip_acc"] = grip_acc

        metrics["pos_err_m"] = (
            metrics["left_pos_err_m"] + metrics["right_pos_err_m"]
        ) / 2
        metrics["rot_err_deg"] = (
            metrics["left_rot_err_deg"] + metrics["right_rot_err_deg"]
        ) / 2
        metrics["grip_acc"] = (
            metrics["left_grip_acc"] + metrics["right_grip_acc"]
        ) / 2

    return metrics


def aggregate_metrics(all_metrics: list) -> dict:
    """Average scalar metrics across episodes."""
    SKIP = {"n_steps", "n_chunks", "episode_idx", "episode_dir", "task_description"}
    all_keys = set()
    for m in all_metrics:
        all_keys.update(
            k for k in m if k not in SKIP and isinstance(m[k], (int, float))
        )
    agg = {
        k: float(np.mean([m[k] for m in all_metrics if k in m]))
        for k in sorted(all_keys)
    }
    agg["n_episodes"] = len(all_metrics)
    agg["total_steps"] = int(sum(m.get("n_steps", 0) for m in all_metrics))
    return agg


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------

def evaluate(
    wrapper: SelfGroundedPredictorEval,
    dataset: SelfGroundedPredictorEvalDataset,
    log_dir: str,
    *,
    max_episodes: Optional[int] = None,
    prediction_stride: Optional[int] = None,
    use_task_video: bool = True,
    seed: int = 42,
    num_inference_steps: Optional[int] = None,
    visualize: bool = False,
):
    """Run chunk-based open-loop evaluation.

    For each episode: iterate chunks of K=action_frames steps, call
    wrapper.infer_real(), compare predicted vs GT actions.
    """
    os.makedirs(log_dir, exist_ok=True)
    run_id = f"openloop-{time.strftime('%Y%m%d_%H%M%S')}"
    log_path = osp.join(log_dir, f"{run_id}.log")
    log_file = open(log_path, "w", buffering=1)

    K = wrapper.action_frames
    stride = prediction_stride if prediction_stride is not None else K

    n_total = len(dataset)
    if max_episodes is not None:
        n_total = min(n_total, max_episodes)

    all_metrics = []
    logger.info("Starting evaluation: %d episodes, K=%d, stride=%d", n_total, K, stride)

    for ep_idx in range(n_total):
        try:
            episode = dataset[ep_idx]
        except Exception as e:
            logger.warning("[ep %d] Failed to load: %s", ep_idx, e)
            log_file.write(f"[SKIP] ep {ep_idx}: {e}\n")
            continue

        ep_dir = episode["episode_dir"]
        task_desc = episode["task_description"]
        gt_actions = episode["gt_actions_raw"]
        gt_actions_normed = episode.get("gt_actions_normed")
        joint_entries_all = episode.get("joint_entries", [])
        progress_list = episode.get("progress_list")

        # Select task video peer episode via task_paths.json
        task_video_dir = None
        if use_task_video:
            _transforms = get_path_transforms(episode.get("dataset_name", ""))
            def _apply_pt(p):
                for old_pfx, new_pfx in _transforms.items():
                    if p.startswith(old_pfx):
                        return p.replace(old_pfx, new_pfx, 1)
                return p

            _ep_dir_t = _apply_pt(ep_dir)
            _task_paths_file = osp.join(_ep_dir_t, "task_paths.json")
            if not osp.exists(_task_paths_file):
                _task_paths_file = osp.join(osp.dirname(_ep_dir_t), "task_paths.json")
            if osp.exists(_task_paths_file):
                with open(_task_paths_file) as _f:
                    _candidates = json.load(_f).get("same", [])
                if _candidates:
                    task_video_dir = random.choice(_candidates)
            if task_video_dir is None:
                logger.warning("[ep %d] No task_paths.json, using ep_dir", ep_idx)
                task_video_dir = ep_dir

        # Convert GT from 6D rotation to Euler so it matches model output (14D)
        if wrapper.use_6d_rotation and wrapper._action_col_map is not None and gt_actions is not None:
            gt_actions = decode_6d_to_euler(gt_actions, wrapper._action_col_map)

        if gt_actions is None or len(gt_actions) < K:
            logger.warning("[ep %d] GT actions too short (%s), skipping",
                           ep_idx, len(gt_actions) if gt_actions is not None else "None")
            continue

        # Determine episode length from frame files
        view_files = next(
            (v for v in episode["frame_files"].values() if v), None
        )
        if view_files is None:
            logger.warning("[ep %d] No frame files, skipping", ep_idx)
            continue
        T = len(view_files)

        logger.info("[ep %d] %s | T=%d | %s", ep_idx, task_desc[:60], T, ep_dir[-50:])

        # Reset wrapper state for new episode
        wrapper.reset()

        # Load task video once per episode (before chunk loop)
        if use_task_video and task_video_dir is not None:
            try:
                wrapper.set_task_video(task_video_dir)
            except Exception as e:
                logger.warning("[ep %d] Failed to load task video from %s: %s", ep_idx, task_video_dir, e)
                continue

        # Collect predictions
        preds_list, gts_list = [], []
        preds_normed_list = []

        max_start = min(len(gt_actions) - K, T - 1)
        for start_idx in range(0, max_start + 1, stride):
            try:
                # Build observation dict
                obs = {}
                for view_name, files in episode["frame_files"].items():
                    if start_idx < len(files):
                        frame = dataset.load_frame_numpy(files[start_idx])
                        if frame is not None:
                            obs[view_name] = frame

                if len(obs) != len(wrapper.views):
                    logger.warning(
                        "[ep %d t=%d] Incomplete obs (%d/%d views), skipping chunk",
                        ep_idx, start_idx, len(obs), len(wrapper.views),
                    )
                    continue

                # Joint state (dict keyed by joint field name)
                robot_state = None
                if joint_entries_all and start_idx < len(joint_entries_all):
                    robot_state = joint_entries_all[start_idx]

                # Load GT frames for visualization (all views, vertically stacked).
                # Sample at the same temporal stride as predicted video so the
                # visual motion range matches (action_video_freq_ratio downsampling).
                _gt_viz_frames = None
                if visualize:
                    # Match predicted video frame count
                    _n_vid = max(5, wrapper.frames)
                    if _n_vid % 4 != 1:
                        _n_vid = (_n_vid // 4) * 4 + 1
                    _end_fi = min(start_idx + K - 1, len(view_files) - 1)
                    _gt_indices = np.linspace(start_idx, _end_fi, _n_vid, dtype=int)

                    _gt_viz_frames = []
                    for fi in _gt_indices:
                        view_stack = []
                        for vn in wrapper.views:
                            _vfiles = episode["frame_files"].get(vn, [])
                            if fi < len(_vfiles):
                                _f = dataset.load_frame_numpy(_vfiles[fi])
                                if _f is not None:
                                    view_stack.append(_f)
                        if len(view_stack) == len(wrapper.views):
                            _gt_viz_frames.append(np.concatenate(view_stack, axis=0))

                # Inference
                _viz_dir = osp.join(log_dir, f"ep_{ep_idx:04d}") if visualize else None
                t0 = time.perf_counter()
                gt_chunk = gt_actions[start_idx:start_idx + K].astype(np.float32)

                result = wrapper.infer_real(
                    observations=obs,
                    joints=robot_state,
                    instruction=task_desc,
                    episode_dir=task_video_dir if use_task_video else None,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    visualize=visualize,
                    viz_dir=_viz_dir,
                    step_count=start_idx,
                    gt_frames=_gt_viz_frames,
                    gt_progress_list=progress_list,
                    current_agent_idx=int(list(episode["frame_files"].values())[0][start_idx].split("::")[-1]),
                    gt_action_chunk=gt_chunk if visualize else None,
                )
                step_s = time.perf_counter() - t0

                pred_chunk = result["action_raw"][:K].astype(np.float32)

                if len(pred_chunk) < K or len(gt_chunk) < K:
                    continue

                preds_list.append(pred_chunk)
                gts_list.append(gt_chunk)

                if result.get("action_normed") is not None:
                    preds_normed_list.append(result["action_normed"][:K])

                # Per-chunk log
                l1, l2 = mae_rmse(pred_chunk, gt_chunk)
                logger.info(
                    "  [t=%d] raw L1=%.6f L2=%.6f  step=%.2fs",
                    start_idx, l1, l2, step_s,
                )

            except Exception as e:
                tb_str = traceback.format_exc()
                logger.warning("[ep %d t=%d] Exception: %s\n%s", ep_idx, start_idx, e, tb_str)
                log_file.write(f"  [t={start_idx}] {e}\n{tb_str}")

        # Episode-level metrics
        if not preds_list:
            logger.warning("[ep %d] No valid chunks", ep_idx)
            continue

        preds_all = np.concatenate(preds_list, axis=0)
        gts_all = np.concatenate(gts_list, axis=0)
        ep_metrics = compute_episode_metrics(preds_all, gts_all)
        ep_metrics["n_chunks"] = len(preds_list)
        ep_metrics["episode_idx"] = ep_idx
        ep_metrics["episode_dir"] = ep_dir
        ep_metrics["task_description"] = task_desc

        # Normed-space metrics
        if preds_normed_list and gt_actions_normed is not None:
            preds_normed_all = np.concatenate(preds_normed_list, axis=0)
            # Re-slice GT normed to match chunks
            gts_normed_chunks = []
            for i, start in enumerate(range(0, max_start + 1, stride)):
                if i < len(preds_normed_list):
                    gts_normed_chunks.append(
                        gt_actions_normed[start:start + K].astype(np.float32)
                    )
            if gts_normed_chunks:
                gts_normed_all = np.concatenate(gts_normed_chunks, axis=0)
                n_min = min(len(preds_normed_all), len(gts_normed_all))
                mae_n, rmse_n = mae_rmse(preds_normed_all[:n_min], gts_normed_all[:n_min])
                ep_metrics["action_mae_normed"] = mae_n
                ep_metrics["action_rmse_normed"] = rmse_n

        all_metrics.append(ep_metrics)

        # Log episode summary
        _summary = (
            f"[ep {ep_idx}] chunks={ep_metrics['n_chunks']} "
            f"mae_raw={ep_metrics['action_mae_raw']:.6f} "
            f"rmse_raw={ep_metrics['action_rmse_raw']:.6f}"
        )
        if "pos_err_m" in ep_metrics:
            _summary += (
                f" pos={ep_metrics['pos_err_m']:.4f}m"
                f" rot={ep_metrics['rot_err_deg']:.2f}deg"
                f" grip={ep_metrics['grip_acc']:.3f}"
            )
        logger.info(_summary)
        log_file.write(_summary + "\n")

        # Save per-episode JSON
        ep_json = json.dumps({k: v for k, v in ep_metrics.items()
                              if isinstance(v, (int, float, str))})
        log_file.write(f"EPISODE_RESULT: {ep_json}\n")

    # Drain any remaining async visualization work
    wrapper.flush_viz()

    # Global aggregation
    if all_metrics:
        agg = aggregate_metrics(all_metrics)
        logger.info("=== Aggregate ===")
        for k, v in sorted(agg.items()):
            logger.info("  %s: %s", k, f"{v:.6f}" if isinstance(v, float) else v)

        agg_path = osp.join(log_dir, f"{run_id}_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        logger.info("Saved aggregate metrics to %s", agg_path)
    else:
        logger.warning("No valid episodes evaluated.")

    log_file.close()
    return all_metrics


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SelfGroundedPredictor open-loop evaluation")
    p.add_argument("--checkpoint_dir", required=True,
                    help="Training output dir with config.yaml + checkpoints/")
    p.add_argument("--eval_data_path", required=True,
                    help="Comma-separated JSON path list (same format as training)")
    p.add_argument("--dataset_name", required=True,
                    help="Dataset id for norm lookup (e.g. '10042')")
    p.add_argument("--views", nargs="+", required=True,
                    help="Camera view names (e.g. leftImg rightImg)")
    p.add_argument("--log_dir", default=None,
                    help="Output directory (default: checkpoint_dir/eval_openloop/)")
    p.add_argument("--max_episodes", type=int, default=None,
                    help="Limit number of episodes to evaluate")
    p.add_argument("--prediction_stride", type=int, default=None,
                    help="Chunk stride (default: action_frames)")
    p.add_argument("--checkpoint_step", type=int, default=None,
                    help="Specific checkpoint step (default: latest)")
    p.add_argument("--diffusion_steps", type=int, default=20,
                    help="Denoising steps for inference")
    p.add_argument("--use_task_video", action="store_true", default=True,
                    help="Use task video conditioning")
    p.add_argument("--no_task_video", action="store_true",
                    help="Disable task video conditioning")
    p.add_argument("--use_task_description", action="store_true", default=True,
                    help="Use text description (only when task_video is present)")
    p.add_argument("--no_task_description", action="store_true",
                    help="Disable text description (tv_only mode)")
    p.add_argument("--visualize", action="store_true",
                    help="Save per-step stitched video + meta.json")
    p.add_argument("--remove_static_frames", action="store_true", default=None,
                    dest="remove_static_frames",
                    help="Enable static frame removal (default: from checkpoint config)")
    p.add_argument("--no_remove_static_frames", action="store_false",
                    dest="remove_static_frames",
                    help="Disable static frame removal")
    p.add_argument("--task_video_static_threshold_ratio", type=float, default=0.5,
                    help="Task video static threshold = training threshold * this ratio")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    use_task_video = args.use_task_video and not args.no_task_video
    use_task_description = args.use_task_description and not args.no_task_description
    log_dir = args.log_dir or osp.join(args.checkpoint_dir, "eval_openloop")

    # Read static-frame thresholds from training config (need config before wrapper init)
    _tmp_cfg = OmegaConf.load(osp.join(args.checkpoint_dir, "config.yaml"))
    _dcfg = _tmp_cfg.data.train
    _remove_static = (
        args.remove_static_frames
        if args.remove_static_frames is not None
        else bool(getattr(_dcfg, "remove_static_frames", False))
    )
    _static_rot_thresh = float(getattr(_dcfg, "static_rot_threshold", 0.01))
    _static_trans_thresh = float(getattr(_dcfg, "static_trans_threshold", 0.01))
    _static_grip_thresh = float(getattr(_dcfg, "static_gripper_threshold", 0.01))

    logger.info("Building model wrapper...")
    wrapper = SelfGroundedPredictorEval(
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset_name,
        views=args.views,
        checkpoint_step=args.checkpoint_step,
        diffusion_steps=args.diffusion_steps,
        use_task_video=use_task_video,
        use_task_description=use_task_description,
        remove_static_frames=_remove_static,
        static_rot_threshold=_static_rot_thresh,
        static_trans_threshold=_static_trans_thresh,
        static_gripper_threshold=_static_grip_thresh,
        task_video_static_threshold_ratio=args.task_video_static_threshold_ratio,
    )

    logger.info("Building evaluation dataset (remove_static_frames=%s)...", _remove_static)
    dataset = SelfGroundedPredictorEvalDataset(
        data_path=args.eval_data_path,
        views=args.views,
        joint_action_mapping_dir=wrapper.joint_action_mapping_dir,
        dataset_name=args.dataset_name,
        use_6d_rotation=wrapper.use_6d_rotation,
        use_relative_action=wrapper.use_relative_action,
        dataset_image_size=wrapper.dataset_image_size,
        context_len=wrapper.context_len,
        cam_mapping_dir=wrapper.cam_mapping_dir,
        dataset_fps=wrapper.dataset_fps,
        task_max_frames=wrapper.task_max_frames,
        remove_static_frames=_remove_static,
        static_rot_threshold=_static_rot_thresh,
        static_trans_threshold=_static_trans_thresh,
        static_gripper_threshold=_static_grip_thresh,
    )

    logger.info("Starting evaluation (use_task_video=%s)...", use_task_video)
    evaluate(
        wrapper=wrapper,
        dataset=dataset,
        log_dir=log_dir,
        max_episodes=args.max_episodes,
        prediction_stride=args.prediction_stride,
        use_task_video=use_task_video,
        seed=args.seed,
        num_inference_steps=args.diffusion_steps,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
