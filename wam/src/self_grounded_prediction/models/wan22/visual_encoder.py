"""Lightweight, trainable visual encoder over the first frame.

Aligned with the input layout used by Physical Intelligence's pi0:
    - each camera image is processed independently at 224x224 by the vision
      backbone (DINOv2-S/14 by default), giving 16x16 = 256 patch tokens per
      camera at native fidelity (no spatial pooling);
    - tokens from all cameras are concatenated, with a learned per-camera id
      embedding added so the action expert can disambiguate which camera each
      token came from;
    - the resulting [B, num_cameras * 256, feat_dim] sequence is projected to
      text_dim through a 2-layer MLP whose final Linear is zero-init, so that
      at the very first step the visual tokens are zero-valued and contribute
      essentially nothing to the action expert's cross-attention output.

Supports three backbone modes via ``backbone_name``:
    - ``dinov2_vits14`` / ``dinov2_vitb14``: DINOv2-only (2-layer MLP projector)
    - ``siglip_so400m_14``: SigLIP-only (2-layer MLP projector)
    - ``dinosiglip_vitb14_so400m`` / ``dinosiglip_vits14_so400m``:
      DINO+SigLIP fused (PrismaticVLM-style channel concatenation + 3-layer
      FusedMLP projector)

The full image fed in is the wide concatenation of the camera views along the
width axis (e.g. x2robot's 224x672 = 3 cameras of 224x224 each); this module
splits it back into per-camera crops before forwarding the backbone.
"""

from __future__ import annotations

import fcntl
import os
from functools import partial
from pathlib import Path

import timm
import torch
import torch.nn as nn

from self_grounded_prediction.utils.logging_config import get_logger

logger = get_logger(__name__)


# ImageNet normalization. DINOv2 was trained with these stats.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# SigLIP normalization.
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


# ── DINOv2 (torch.hub) loading ──────────────────────────────────────────────

def _hub_load_serialized(repo: str, entry: str):
    """torch.hub.load serialized via a per-node fcntl exclusive file lock.

    Without this, all `nproc_per_node` ranks call torch.hub.load concurrently
    and race on `~/.cache/torch/hub/`, which manifests as `FileExistsError` /
    `OSError(39, 'Directory not empty')` during zip extraction. (We can't use
    `dist.barrier()` here because the process group has not yet been
    initialized at the time the model is constructed — Accelerator initializes
    it in the trainer, after model build.)

    The first rank to acquire the lock downloads + extracts; subsequent ranks
    on the same node enter the critical section in turn but see a warm cache
    and return immediately.
    """
    cache_root = Path(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")))
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_repo = repo.replace("/", "_")
    lock_path = cache_root / f".hub_lock_{safe_repo}_{entry}"
    with open(lock_path, "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            return torch.hub.load(repo, entry, source="github", trust_repo=True)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _local_load(local_repo: str, entry: str, weights_path: str):
    """Load DINOv2 backbone from a local repo clone + local weights file.

    Uses torch.hub.load with source="local" to build the model architecture
    from the cloned repo, then loads pretrained weights from a local .pth file.
    """
    local_repo = str(local_repo)
    if not Path(local_repo).is_dir():
        raise FileNotFoundError(f"Local dinov2 repo not found: {local_repo}")
    weights_path = str(weights_path)
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Local weights file not found: {weights_path}")

    # Build model architecture from local repo (pretrained=False to skip download)
    model = torch.hub.load(local_repo, entry, source="local", pretrained=False)
    # Load weights from local file
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded DINOv2 backbone `%s` from local: repo=%s weights=%s", entry, local_repo, weights_path)
    return model


# ── SigLIP (TIMM) loading ───────────────────────────────────────────────────

def _timm_load_siglip(timm_model_id: str, image_size: int,
                       local_weights_path: str | None = None):
    """Load a SigLIP backbone via TIMM.

    Uses fcntl lock for thread-safe weight downloading (same pattern as DINOv2).
    """
    cache_root = Path(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")))
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_id = timm_model_id.replace("/", "_")
    lock_path = cache_root / f".timm_lock_{safe_id}"

    pretrained_cfg_overlay = None
    if local_weights_path is not None:
        local_weights_path = str(local_weights_path)
        if not Path(local_weights_path).is_file():
            raise FileNotFoundError(f"Local SigLIP weights not found: {local_weights_path}")
        pretrained_cfg_overlay = {"file": local_weights_path}
        logger.info("Loading SigLIP `%s` from local: %s", timm_model_id, local_weights_path)

    with open(lock_path, "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            model = timm.create_model(
                timm_model_id,
                pretrained=True,
                num_classes=0,
                img_size=image_size,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
            )
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    if local_weights_path is None:
        logger.info("Loaded SigLIP `%s` from TIMM hub", timm_model_id)
    return model


# ── Backbone registries ─────────────────────────────────────────────────────

# DINOv2 via torch.hub: (repo, entry, feat_dim, patch_size)
_DINO_ENTRYPOINTS = {
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14", 384, 14),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", 768, 14),
}

# SigLIP via TIMM: (timm_model_id, feat_dim, patch_size)
_SIGLIP_ENTRYPOINTS = {
    "siglip_so400m_14": ("vit_so400m_patch14_siglip_224", 1152, 14),
}

# Fused DINO+SigLIP: references into the above two registries
_FUSED_ENTRYPOINTS = {
    "dinosiglip_vitb14_so400m": {"dino": "dinov2_vitb14", "siglip": "siglip_so400m_14"},
    "dinosiglip_vits14_so400m": {"dino": "dinov2_vits14", "siglip": "siglip_so400m_14"},
}

# Backward-compat alias (referenced in existing code/checkpoints)
_HUB_ENTRYPOINTS = _DINO_ENTRYPOINTS


def _resolve_backbone(backbone_name: str):
    """Return (mode, patch_size, feat_dim, dino_cfg|None, siglip_cfg|None).

    mode is one of "dino", "siglip", "fused".
    """
    if backbone_name in _DINO_ENTRYPOINTS:
        repo, entry, feat_dim, ps = _DINO_ENTRYPOINTS[backbone_name]
        return "dino", ps, feat_dim, {"repo": repo, "entry": entry, "feat_dim": feat_dim}, None

    if backbone_name in _SIGLIP_ENTRYPOINTS:
        timm_id, feat_dim, ps = _SIGLIP_ENTRYPOINTS[backbone_name]
        return "siglip", ps, feat_dim, None, {"timm_id": timm_id, "feat_dim": feat_dim}

    if backbone_name in _FUSED_ENTRYPOINTS:
        fused = _FUSED_ENTRYPOINTS[backbone_name]
        dino_repo, dino_entry, dino_dim, dino_ps = _DINO_ENTRYPOINTS[fused["dino"]]
        siglip_timm, siglip_dim, siglip_ps = _SIGLIP_ENTRYPOINTS[fused["siglip"]]
        assert dino_ps == siglip_ps, (
            f"Patch size mismatch: DINO={dino_ps}, SigLIP={siglip_ps}"
        )
        fused_dim = dino_dim + siglip_dim
        return (
            "fused", dino_ps, fused_dim,
            {"repo": dino_repo, "entry": dino_entry, "feat_dim": dino_dim},
            {"timm_id": siglip_timm, "feat_dim": siglip_dim},
        )

    all_names = sorted(
        list(_DINO_ENTRYPOINTS) + list(_SIGLIP_ENTRYPOINTS) + list(_FUSED_ENTRYPOINTS)
    )
    raise ValueError(
        f"Unknown backbone `{backbone_name}`; supported: {all_names}"
    )


class FirstFrameVisualEncoder(nn.Module):
    """Trainable per-camera visual encoder (pi0-style layout).

    Supports DINOv2-only, SigLIP-only, or DINO+SigLIP fused backbones.

    Input: RGB tensor in [-1, 1], shape [B, 3, H, W], where H == camera_input_size
    and W == camera_input_size * num_cameras (cameras concatenated horizontally).
    H/camera_input_size must equal patch_size's grid alignment.

    Output: tokens [B, num_tokens, text_dim], with
        num_tokens = num_cameras * (camera_input_size // patch_size) ** 2.
    """

    def __init__(
        self,
        text_dim: int,
        backbone_name: str = "dinov2_vits14",
        num_cameras: int = 3,
        camera_input_size: int = 224,
        proj_hidden: int = 1024,
        dropout: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        backbone_local_repo: str | None = None,
        backbone_weights_path: str | None = None,
        siglip_local_weights_path: str | None = None,
    ):
        super().__init__()

        mode, patch_size, feat_dim, dino_cfg, siglip_cfg = _resolve_backbone(backbone_name)
        if int(camera_input_size) % int(patch_size) != 0:
            raise ValueError(
                f"`camera_input_size`={camera_input_size} must be divisible by patch={patch_size}"
            )
        if int(num_cameras) <= 0:
            raise ValueError(f"`num_cameras` must be positive, got {num_cameras}")

        self.mode = mode  # "dino", "siglip", or "fused"
        self.backbone_name = backbone_name
        self.feat_dim = int(feat_dim)
        self.patch_size = int(patch_size)
        self.text_dim = int(text_dim)
        self.num_cameras = int(num_cameras)
        self.camera_input_size = int(camera_input_size)
        self.patches_per_side = self.camera_input_size // self.patch_size
        self.patches_per_cam = self.patches_per_side * self.patches_per_side
        self.num_tokens = self.patches_per_cam * self.num_cameras

        # ── Load backbone(s) ────────────────────────────────────────────
        grid_str = f"{self.patches_per_side}x{self.patches_per_side}"

        if mode == "dino":
            self.dino_backbone = self._load_dino(
                dino_cfg, backbone_local_repo, backbone_weights_path, torch_dtype
            )
            self.siglip_backbone = None
            logger.info(
                "Visual encoder mode=dino `%s`; %d camera(s) at %dx%d, %s patch grid.",
                backbone_name, num_cameras, camera_input_size, camera_input_size, grid_str,
            )

        elif mode == "siglip":
            self.dino_backbone = None
            self.siglip_backbone = self._load_siglip(
                siglip_cfg, camera_input_size, siglip_local_weights_path, torch_dtype
            )
            logger.info(
                "Visual encoder mode=siglip `%s`; %d camera(s) at %dx%d, %s patch grid.",
                backbone_name, num_cameras, camera_input_size, camera_input_size, grid_str,
            )

        elif mode == "fused":
            self.dino_backbone = self._load_dino(
                dino_cfg, backbone_local_repo, backbone_weights_path, torch_dtype
            )
            self.siglip_backbone = self._load_siglip(
                siglip_cfg, camera_input_size, siglip_local_weights_path, torch_dtype
            )
            # Monkey-patch SigLIP forward to extract second-to-last layer
            # features, following PrismaticVLM.
            n_blocks = len(self.siglip_backbone.blocks)
            self.siglip_backbone._get_intermediate = partial(
                self.siglip_backbone.get_intermediate_layers,
                n={n_blocks - 2},
            )
            logger.info(
                "Visual encoder mode=fused `%s`; DINO(%s, %d-d) + SigLIP(%s, %d-d) → %d-d; "
                "%d camera(s) at %dx%d, %s patch grid.",
                backbone_name,
                dino_cfg["entry"], dino_cfg["feat_dim"],
                siglip_cfg["timm_id"], siglip_cfg["feat_dim"],
                self.feat_dim,
                num_cameras, camera_input_size, camera_input_size, grid_str,
            )

        # Keep backward-compat alias
        if mode == "dino":
            self.backbone = self.dino_backbone
        elif mode == "siglip":
            self.backbone = self.siglip_backbone
        else:
            self.backbone = None

        # ── Per-camera ID embedding ─────────────────────────────────────
        self.camera_emb = nn.Parameter(
            torch.zeros(self.num_cameras, self.feat_dim, dtype=torch_dtype)
        )
        nn.init.trunc_normal_(self.camera_emb, std=0.02)

        # ── Projector ──────────────────────────────────────────────────
        self.norm = nn.LayerNorm(self.feat_dim).to(torch_dtype)

        if mode == "fused":
            # 3-layer FusedMLP (PrismaticVLM-style):
            #   Linear(feat_dim, 4*feat_dim) → GELU →
            #   Linear(4*feat_dim, proj_hidden) → GELU →
            #   Linear(proj_hidden, text_dim) → zero-init
            fused_expand = 4 * self.feat_dim
            self.fc1 = nn.Linear(self.feat_dim, fused_expand).to(torch_dtype)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(fused_expand, proj_hidden).to(torch_dtype)
            self.act2 = nn.GELU()
            self.fc3 = nn.Linear(proj_hidden, self.text_dim).to(torch_dtype)
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)
            # Aliases for checkpoint compat (unused in fused forward, but
            # present so state_dict key names don't break mixed loads).
            self.act = self.act1
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            # 2-layer MLP (original):
            #   Linear(feat_dim, proj_hidden) → GELU → Dropout →
            #   Linear(proj_hidden, text_dim) → zero-init
            self.fc1 = nn.Linear(self.feat_dim, proj_hidden).to(torch_dtype)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.fc2 = nn.Linear(proj_hidden, self.text_dim).to(torch_dtype)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
            # Placeholders so state_dict shape is consistent
            self.act1 = self.act
            self.act2 = None
            self.fc3 = None

        # ── Learned positional embedding (post-projector, in text_dim) ──
        # Zero-init so visual tokens remain zero-valued at step 0,
        # consistent with the zero-init projector final layer.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.text_dim, dtype=torch_dtype)
        )

        # ── Normalization buffers ───────────────────────────────────────
        if mode in ("dino", "fused"):
            self.register_buffer(
                "_dino_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "_dino_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
            )
        if mode in ("siglip", "fused"):
            self.register_buffer(
                "_siglip_mean", torch.tensor(_SIGLIP_MEAN).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "_siglip_std", torch.tensor(_SIGLIP_STD).view(1, 3, 1, 1), persistent=False
            )

        # Backward-compat aliases for dino-only mode
        if mode == "dino":
            self.register_buffer("_mean", self._dino_mean, persistent=False)
            self.register_buffer("_std", self._dino_std, persistent=False)

    # ── Backbone loading helpers ────────────────────────────────────────

    @staticmethod
    def _load_dino(cfg: dict, local_repo, weights_path, dtype):
        """Load DINOv2 backbone via torch.hub (or local)."""
        if local_repo is not None and weights_path is not None:
            backbone = _local_load(str(local_repo), cfg["entry"], str(weights_path))
        else:
            backbone = _hub_load_serialized(cfg["repo"], cfg["entry"])
        backbone.to(dtype=dtype)
        return backbone

    @staticmethod
    def _load_siglip(cfg: dict, image_size: int, local_weights_path, dtype):
        """Load SigLIP backbone via TIMM."""
        backbone = _timm_load_siglip(
            cfg["timm_id"], image_size,
            local_weights_path=local_weights_path,
        )
        backbone.to(dtype=dtype)
        return backbone

    # ── Normalization ───────────────────────────────────────────────────

    def _normalize_dino(self, rgb_m1_1: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → ImageNet-normalized tensor in backbone dtype."""
        x = (rgb_m1_1.float() + 1.0) * 0.5  # → [0, 1]
        x = x.clamp(0.0, 1.0)
        x = (x - self._dino_mean) / self._dino_std
        return x.to(dtype=self.fc1.weight.dtype)

    def _normalize_siglip(self, rgb_m1_1: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → SigLIP-normalized tensor in backbone dtype."""
        x = (rgb_m1_1.float() + 1.0) * 0.5  # → [0, 1]
        x = x.clamp(0.0, 1.0)
        x = (x - self._siglip_mean) / self._siglip_std
        return x.to(dtype=self.fc1.weight.dtype)

    def _normalize(self, rgb_m1_1: torch.Tensor) -> torch.Tensor:
        """Backward-compat: DINOv2 normalization (dino-only mode)."""
        if rgb_m1_1.ndim != 4 or rgb_m1_1.shape[1] != 3:
            raise ValueError(
                f"`rgb_m1_1` must be [B, 3, H, W], got {tuple(rgb_m1_1.shape)}"
            )
        return self._normalize_dino(rgb_m1_1)

    # ── Per-camera crop + resize ────────────────────────────────────────

    def _split_cameras(self, rgb_m1_1: torch.Tensor):
        """Split multi-view [B, 3, num_cams*H, W] → [B*num_cams, 3, H_cam, W],
        then resize to (camera_input_size, camera_input_size).

        Returns the un-normalized flat tensor.
        """
        b, _, h, w = rgb_m1_1.shape
        if h % self.num_cameras != 0:
            raise ValueError(
                f"Input H={h} must be divisible by num_cameras={self.num_cameras} "
                f"(vertical concat layout), got HxW=({h},{w})."
            )
        per_cam_h = h // self.num_cameras
        # [B, 3, num_cams*h_cam, W] -> [B, num_cams, 3, h_cam, W]
        x_split = rgb_m1_1.view(b, 3, self.num_cameras, per_cam_h, w)
        x_split = x_split.permute(0, 2, 1, 3, 4).contiguous()
        # -> [B * num_cams, 3, h_cam, W]
        x_flat = x_split.view(b * self.num_cameras, 3, per_cam_h, w)
        # Resize to square input expected by backbones
        if per_cam_h != self.camera_input_size or w != self.camera_input_size:
            x_flat = torch.nn.functional.interpolate(
                x_flat,
                size=(self.camera_input_size, self.camera_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return x_flat  # still in [-1, 1], un-normalized

    # ── Feature extraction ──────────────────────────────────────────────

    def _extract_dino(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 on [B*C, 3, H, W] (already normalized) → [B*C, P, D_dino]."""
        out = self.dino_backbone.forward_features(x_flat)
        if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
            raise RuntimeError(
                f"DINOv2 backbone did not return `x_norm_patchtokens`; "
                f"got keys={list(out) if isinstance(out, dict) else type(out)}"
            )
        tokens = out["x_norm_patchtokens"]  # [B*C, P, D_dino]
        assert tokens.shape[1] == self.patches_per_cam, (
            f"DINOv2 patch count {tokens.shape[1]} != expected {self.patches_per_cam}"
        )
        return tokens

    def _extract_siglip(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Run SigLIP on [B*C, 3, H, W] (already normalized) → [B*C, P, D_siglip]."""
        if self.mode == "fused":
            # Use second-to-last layer features (PrismaticVLM-style)
            out_tuple = self.siglip_backbone._get_intermediate(x_flat)
            tokens = out_tuple[0]  # [B*C, P, D_siglip]
        else:
            # SigLIP-only mode: use forward_features (last layer)
            tokens = self.siglip_backbone.forward_features(x_flat)
        assert tokens.shape[1] == self.patches_per_cam, (
            f"SigLIP patch count {tokens.shape[1]} != expected {self.patches_per_cam}"
        )
        return tokens

    # ── Projection ──────────────────────────────────────────────────────

    def _project_fused(self, tokens: torch.Tensor) -> torch.Tensor:
        """3-layer FusedMLP: [B, N, feat_dim] → [B, N, text_dim]."""
        z = self.norm(tokens)
        z = self.fc1(z)
        z = self.act1(z)
        z = self.fc2(z)
        z = self.act2(z)
        z = self.dropout(z)
        z = self.fc3(z)
        return z

    def _project_single(self, tokens: torch.Tensor) -> torch.Tensor:
        """2-layer MLP: [B, N, feat_dim] → [B, N, text_dim]."""
        z = self.norm(tokens)
        z = self.fc1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, rgb_m1_1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_m1_1: [B, 3, H, W] with H = num_cameras * cam_h, values in [-1, 1].

        Returns:
            tokens: [B, num_tokens, text_dim]
        """
        b = rgb_m1_1.shape[0]
        x_flat = self._split_cameras(rgb_m1_1)  # [B*C, 3, H_cam, W_cam], still [-1, 1]

        if self.mode == "fused":
            # Dual-backbone: DINO + SigLIP → channel concat
            x_dino = self._normalize_dino(x_flat)
            x_siglip = self._normalize_siglip(x_flat)
            dino_tokens = self._extract_dino(x_dino)      # [B*C, P, D_dino]
            siglip_tokens = self._extract_siglip(x_siglip)  # [B*C, P, D_siglip]
            patch_tokens = torch.cat([dino_tokens, siglip_tokens], dim=-1)  # [B*C, P, D_fused]

        elif self.mode == "dino":
            x_dino = self._normalize_dino(x_flat)
            patch_tokens = self._extract_dino(x_dino)  # [B*C, P, D_dino]

        elif self.mode == "siglip":
            x_siglip = self._normalize_siglip(x_flat)
            patch_tokens = self._extract_siglip(x_siglip)  # [B*C, P, D_siglip]

        # Reshape to [B, num_cameras, P, feat_dim] and add camera embeddings
        patch_tokens = patch_tokens.view(
            b, self.num_cameras, self.patches_per_cam, self.feat_dim
        )
        patch_tokens = patch_tokens + self.camera_emb.view(
            1, self.num_cameras, 1, self.feat_dim
        ).to(patch_tokens.dtype)
        tokens = patch_tokens.reshape(b, self.num_tokens, self.feat_dim)

        # Project to text_dim
        if self.mode == "fused":
            tokens = self._project_fused(tokens)
        else:
            tokens = self._project_single(tokens)

        # Add learned positional embedding  # [B, num_tokens, text_dim]
        tokens = tokens + self.pos_embed.to(tokens.dtype)
        return tokens
