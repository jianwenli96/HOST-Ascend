#!/usr/bin/env python3
"""Precompute in-place instruction.pt files for CustomDataset episodes."""

import argparse
import json
import os
import uuid
from pathlib import Path

import _ensure_project_src

_ensure_project_src.ensure()

import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception:
    pass

from self_grounded_prediction.models.wan22.helpers.loader import _load_registered_model
from self_grounded_prediction.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer


PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-paths-json", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_episode_prompts(video_paths_json: Path) -> dict[str, list[Path]]:
    episode_dirs = json.loads(video_paths_json.read_text(encoding="utf-8"))
    if not isinstance(episode_dirs, list) or not episode_dirs:
        raise ValueError(f"Expected a non-empty path list in {video_paths_json}")

    prompts: dict[str, list[Path]] = {}
    for raw_dir in episode_dirs:
        episode_dir = Path(raw_dir)
        instruction_path = episode_dir / "instruction.txt"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"Missing instruction: {instruction_path}")
        task = instruction_path.read_text(encoding="utf-8").strip()
        if not task:
            raise ValueError(f"Empty instruction: {instruction_path}")
        prompt = PROMPT_TEMPLATE.format(task=task)
        prompts.setdefault(prompt, []).append(episode_dir)
    return prompts


def _atomic_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp.{uuid.uuid4().hex}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


def main() -> None:
    args = _parse_args()
    model_dir = args.model_dir.resolve()
    text_weights = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_dir = model_dir / "google" / "umt5-xxl"
    if not text_weights.is_file():
        raise FileNotFoundError(f"Missing text encoder weights: {text_weights}")
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_dir}")

    prompts_to_dirs = _load_episode_prompts(args.video_paths_json.resolve())
    pending = {
        prompt: [d for d in dirs if args.overwrite or not (d / "instruction.pt").exists()]
        for prompt, dirs in prompts_to_dirs.items()
    }
    pending = {prompt: dirs for prompt, dirs in pending.items() if dirs}
    if not pending:
        print("[INFO] All instruction.pt files already exist; nothing to do.")
        return

    print(
        f"[INFO] Loading Wan UMT5 encoder from {text_weights} on {args.device}; "
        f"unique_prompts={len(pending)} episodes={sum(map(len, pending.values()))}"
    )
    text_encoder = _load_registered_model(
        str(text_weights),
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=args.device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_dir),
        seq_len=int(args.context_len),
        clean="whitespace",
    )

    prompts = list(pending)
    with torch.no_grad():
        ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(args.device)
        mask = mask.to(device=args.device, dtype=torch.bool)
        context = text_encoder(ids, mask)

    written = 0
    for index, prompt in enumerate(prompts):
        valid_len = max(int(mask[index].sum().item()), 1)
        payload = {
            "context": context[index, :valid_len].detach().to(
                "cpu", dtype=torch.bfloat16
            ).contiguous(),
            "mask": mask[index, :valid_len].detach().to(
                "cpu", dtype=torch.bool
            ).contiguous(),
        }
        for episode_dir in pending[prompt]:
            _atomic_save(payload, episode_dir / "instruction.pt")
            written += 1
    print(f"[INFO] Wrote {written} instruction.pt files from {len(prompts)} unique prompts.")


if __name__ == "__main__":
    main()
