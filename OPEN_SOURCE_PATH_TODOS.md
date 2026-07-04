# Open-Source Path Cleanup TODOs

Tracking list for internal-cluster paths left in place during the P1 cleanup pass
because they are **load-bearing for the actual-run verification** completed this
session (WAM: 22,230+ real training steps; alignment: stable past step 5 — both using
the exact configs/paths below). Every location is marked in-code with a
`TODO(open-source)` comment; `grep -rn "TODO(open-source)"` from the repo root finds
all of them.

**Do not touch any of these without re-running the full actual-run verification
afterward** (real training steps printed, not just import/parse checks) — that's the
whole reason this pass deferred them instead of "fixing" them blind.

## Load-bearing (must re-verify after changing)

| File | What | Planned fix |
|---|---|---|
| `wam/configs/model/fastwam_joint_cross_attn_ve.yaml` (`backbone_local_repo`, `backbone_weights_path`, `siglip_local_weights_path`) | DINOv2/SigLIP visual-encoder weights, internal cluster paths | Code already has an automatic fallback: `visual_encoder.py`'s `_load_dino`/`_load_siglip` download from `facebookresearch/dinov2` (torch.hub) and TIMM automatically when these fields are unset/`None`. Simplest fix is likely just **deleting these 3 lines** — but must be re-verified end-to-end since it changes what actually gets loaded (local internal weights → public download). |
| `wam/src/fastwam/datasets/custom/path_transforms_config.py` (`PATH_TRANSFORMS` dict, esp. `"10042"` entry) | Legacy dataset path-prefix remapping | Not part of the README's documented external repro path (README's Dataset Download section only covers LIBERO/RoboTwin) — this only matters for the internal real-robot/web-scale datasets. Likely fix: extract the common `/open_data/cgy/anns` root into an env-var-overridable constant, or leave as-is with a clarifying "internal-only, not part of external repro" note if not worth the effort. |
| `wam/configs/data/custom_cross_all.yaml` (`data_path`, `cam_mapping_dir`, `joint_action_mapping_dir`, all in the `train:` block) | Internal dataset/annotation paths, actively read at training time | Same scope question as above — this data config isn't part of the documented LIBERO/RoboTwin external path. Decide whether to leave as an "internal-config example" or make paths env-var-overridable. |
| `alignment/models.py` (Qwen3-VL-Embedding-8B path, `elif 'Qwen3-VL' in network:` branch) | 8B checkpoint path, no fallback today | Confirmed by the user: this is genuinely the public model at https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B. Fix: read an env var (e.g. `QWEN3_VL_CKPT_PATH`) defaulting to this exact hardcoded path (zero behavior change here), falling back to the public HF repo ID if neither the env var nor the hardcoded path exists, with a `logging.warning` when falling back (no silent behavior change). |
| `alignment/train.py:~247` (same Qwen3-VL path, in the `AutoProcessor.from_pretrained` call) | Duplicate of the above, in the actual training script | Same fix as `models.py` — keep both in sync. |
| `alignment/config.py` (`CONFIG.DATA.TASK_PATHS_TRANSFORMS`, esp. `"10042"` entry) | Mirrors wam's `PATH_TRANSFORMS` | Same treatment as `path_transforms_config.py` above. |

## Not load-bearing (lower priority — verified run never reaches these branches)

| File | What | Why lower priority |
|---|---|---|
| `alignment/models.py` (`Resnet50`/`Resnet50_pretrained` branch, `dinov2_vitb14`/`dinov2_vitl14` branches) | Hardcoded internal ResNet50/DINOv2 paths | Verified run uses `NETWORK=Qwen3-VL-2B`, never reaches these `elif` branches. Already have graceful fallback to a public download URL / torch.hub if the local path is missing — reasonably safe for external users as-is. |
| `alignment/evaluate_v2.py:~185` | Same Qwen3-VL path, eval-only | Eval wasn't part of this session's verification scope. Apply the same fix as `models.py` when doing the consolidated pass, for consistency. |
| `alignment/extract_embeddings.py:~47` | Same Qwen3-VL path, utility script | Not part of the training or eval path exercised this session. |
| `alignment/scripts/convert_ds_to_hf.sh:9`, `alignment/scripts/convert_ckpt_to_hf.py:104` | Same Qwen3-VL path, already accept a CLI override (`${2:-...}` / `argparse default=`) with this as the fallback default | Already partially configurable; lowest priority — just update the default alongside the others for consistency. |
| `wam/eval/real_openloop/eval_dataset.py:~52` (`cam_mapping_dir` default) | Internal cam-mapping path default | Only exercised by real-robot open-loop eval, not the verified training run. |

## Confirmed as "nothing to do" (do not re-investigate)

- **`wam/checkpoints` symlink** (→ internal `open_ckpts/fastwam_ckpts`): confirmed
  **untracked by git** — `checkpoints/` is in `.gitignore`, `git ls-files` returns 0
  hits under it. It exists only in this session's local working copy for our own
  verification runs. External users never see it; the README's `mkdir -p checkpoints`
  + ActionDiT-extraction steps already produce the equivalent for them.

## Process for the eventual consolidated pass

1. Implement the "Planned fix" for each load-bearing row above — prefer additive,
   backward-compatible changes (env var overrides that default to today's hardcoded
   value) so nothing changes unless the override is explicitly set.
2. Run **one** full actual-run verification afterward (WAM: real training steps
   printed; alignment: real training steps printed) — same bar used this session.
3. Only then push.
