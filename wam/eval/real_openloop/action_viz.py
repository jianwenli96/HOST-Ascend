"""Shared action visualization utilities for FastWAM eval (training + open-loop)."""
from __future__ import annotations

import math
import os

import numpy as np

N_COLS = 4  # number of columns in the dim grid


def save_action_plot(
    save_path: str,
    pred: np.ndarray,
    gt: np.ndarray,
    title: str = "Action",
    labels: list[str] | None = None,
):
    """Save per-chunk action comparison plot (pred vs GT).

    Layout: uniform N_COLS-column grid, one subplot per dimension.

    Args:
        save_path: Output PNG path.
        pred:      [T, D] numpy array — predicted actions.
        gt:        [T, D] numpy array — ground-truth actions.
        title:     Plot super-title.
        labels:    Per-dimension labels (len=D). Defaults to "dim_i".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    D = pred.shape[1]
    T = len(pred)

    if labels is None:
        labels = [f"dim_{i}" for i in range(D)]

    n_cols = min(N_COLS, D)
    n_rows = math.ceil(D / n_cols)

    _xticks = [0, T // 2, T - 1] if T > 2 else list(range(T))
    steps = np.arange(T)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 1.8 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for d in range(D):
        row, col = divmod(d, n_cols)
        ax = axes[row][col]
        ax.plot(steps, gt[:, d],   "r-",  linewidth=1.2, label="GT")
        ax.plot(steps, pred[:, d], "b--", linewidth=1.0, label="Pred")
        ax.set_ylabel(labels[d], fontsize=7)
        ax.set_xticks(_xticks)
        ax.tick_params(labelsize=6)
        if d == 0:
            ax.legend(fontsize=6, loc="upper right")
        if row == n_rows - 1:
            ax.set_xlabel(f"step (horizon={T})", fontsize=7)

    # Hide unused subplots
    for d in range(D, n_rows * n_cols):
        row, col = divmod(d, n_cols)
        axes[row][col].set_visible(False)

    mae  = np.abs(pred - gt).mean()
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    fig.suptitle(f"{title}  |  MAE={mae:.4f}  RMSE={rmse:.4f}", fontsize=9)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
