"""
Triple-view video → Wan2.2 VAE encode → decode → visualization.

Reads three camera-view videos (left, right, face), horizontally
concatenates them, encodes through Wan2.2 VAE, decodes back,
and saves a side-by-side comparison (original vs. reconstruction).

Usage:
    python wanvae_dual_view.py
    python wanvae_dual_view.py --max_frames 65 --frame_h 256 --frame_w 256
    python wanvae_dual_view.py --video_dir /path/to/folder --views cam0.mp4 cam1.mp4 cam2.mp4
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wan.modules.vae2_2 import Wan2_2_VAE


def read_video_frames(video_path: str, frame_h: int, frame_w: int,
                      max_frames: int | None = None) -> tuple[np.ndarray, float]:
    assert os.path.exists(video_path), f"Not found: {video_path}"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_w, frame_h))
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames), fps


def frames_to_tensor(frames_np: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (3, T, H, W) float32 in [-1, 1]."""
    t = torch.from_numpy(frames_np).float() / 255.0
    t = t * 2.0 - 1.0
    return t.permute(3, 0, 1, 2).contiguous()


def tensor_to_frames(t: torch.Tensor) -> np.ndarray:
    """(3, T, H, W) float in [-1, 1] → (T, H, W, 3) uint8."""
    t = t.float().clamp(-1, 1)
    t = (t + 1.0) / 2.0 * 255.0
    return t.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)


def write_video(path: str, frames: np.ndarray, fps: float):
    """Write (T, H, W, 3) uint8 frames to mp4 via OpenCV."""
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for i in range(T):
        writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    writer.release()


def make_frame_grid(orig: np.ndarray, recon: np.ndarray, n_samples: int = 8) -> np.ndarray:
    """Sample frames and tile: each column is [orig / recon]."""
    T = min(len(orig), len(recon))
    indices = np.linspace(0, T - 1, n_samples, dtype=int)
    cols = []
    for idx in indices:
        col = np.concatenate([orig[idx], recon[idx]], axis=0)
        cols.append(col)
    return np.concatenate(cols, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 VAE triple-view encode/decode test")
    parser.add_argument("--video_dir", type=str,
                        default="/x2robot_v2/harryjhou/project/Wan2.2/"
                                "20260302-day-desk_place_tissue_on_tray"
                                "@MASTER_SLAVE_MODE@2026_03_02_10_50_59")
    parser.add_argument("--views", type=str, nargs="+",
                        default=["leftImg.mp4", "rightImg.mp4", "faceImg.mp4"],
                        help="Video filenames for each view (concatenated left-to-right)")
    parser.add_argument("--vae_path", type=str,
                        default="/x2robot_v2/harryjhou/project/Wan2.2/"
                                "checkpoint/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--max_frames", type=int, default=444,
                        help="Frames to use. Best if T = 1+4k (5,9,13,...,33,65,...)")
    parser.add_argument("--frame_h", type=int, default=256,
                        help="Height per view (must be divisible by 16)")
    parser.add_argument("--frame_w", type=int, default=640,
                        help="Width per view (concat width = frame_w * N_views, must be divisible by 16)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    n_views = len(args.views)
    assert args.frame_h % 16 == 0, f"frame_h must be divisible by 16, got {args.frame_h}"
    assert args.frame_w % 16 == 0, f"frame_w must be divisible by 16, got {args.frame_w}"
    concat_w = args.frame_w * n_views
    assert concat_w % 16 == 0, (
        f"Concatenated width {concat_w} must be divisible by 16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # ---- 1. Read & concatenate multi-view video ----
    all_view_frames = []
    fps = 20.0
    for i, view_name in enumerate(args.views):
        view_path = os.path.join(args.video_dir, view_name)
        print(f"[1/5] Reading view {i}: {view_path}")
        frames, view_fps = read_video_frames(view_path, args.frame_h, args.frame_w, args.max_frames)
        all_view_frames.append(frames)
        if i == 0:
            fps = view_fps

    T = min(len(f) for f in all_view_frames)
    all_view_frames = [f[:T] for f in all_view_frames]
    concat_frames = np.concatenate(all_view_frames, axis=2)  # (T, H, N*W, 3)
    print(f"       {n_views}-view concat: {T} frames, "
          f"{concat_frames.shape[1]}x{concat_frames.shape[2]}")

    # ---- 2. Prepare tensor ----
    video_tensor = frames_to_tensor(concat_frames).to(device)  # (3, T, H, N*W)
    print(f"[2/5] Input tensor: {list(video_tensor.shape)}, "
          f"range [{video_tensor.min():.2f}, {video_tensor.max():.2f}]")

    # ---- 3. Load VAE & encode ----
    print(f"[3/5] Loading Wan2.2 VAE ({args.dtype}) ...")
    vae = Wan2_2_VAE(vae_pth=args.vae_path, dtype=dtype, device=str(device))
    print("       VAE loaded.")

    print("[3/5] Encoding ...")
    with torch.no_grad():
        latents = vae.encode([video_tensor])
    z = latents[0]
    print(f"       Latent shape: {list(z.shape)}")
    print(f"       Latent stats: mean={z.mean():.4f}, std={z.std():.4f}, "
          f"min={z.min():.4f}, max={z.max():.4f}")
    print(f"       Compression: {video_tensor.numel()} → {z.numel()} "
          f"({video_tensor.numel() / z.numel():.1f}x)")

    # ---- 4. Decode ----
    print("[4/5] Decoding ...")
    with torch.no_grad():
        decoded = vae.decode([z])
    recon_tensor = decoded[0]  # (3, T', H, N*W)
    print(f"       Decoded shape: {list(recon_tensor.shape)}")

    recon_frames = tensor_to_frames(recon_tensor)
    T_out = min(T, recon_frames.shape[0])
    orig_vis = concat_frames[:T_out]
    recon_vis = recon_frames[:T_out]

    # ---- 5. Save outputs ----
    out_dir = args.output_dir or os.path.join(args.video_dir, "vae_output")
    os.makedirs(out_dir, exist_ok=True)

    comparison_frames = np.concatenate([orig_vis, recon_vis], axis=1)  # vertical stack

    comp_path = os.path.join(out_dir, "comparison.mp4")
    print(f"[5/5] Saving comparison video ({T_out} frames) → {comp_path}")
    write_video(comp_path, comparison_frames, fps)

    orig_path = os.path.join(out_dir, "original.mp4")
    recon_path = os.path.join(out_dir, "reconstructed.mp4")
    write_video(orig_path, orig_vis, fps)
    write_video(recon_path, recon_vis, fps)
    print(f"       Original       → {orig_path}")
    print(f"       Reconstructed  → {recon_path}")

    grid = make_frame_grid(orig_vis, recon_vis)
    grid_path = os.path.join(out_dir, "frame_grid.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"       Frame grid     → {grid_path}")

    print(f"\nDone! All outputs in {out_dir}")


if __name__ == "__main__":
    main()
