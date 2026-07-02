"""Ensure ``<project_root>/src`` is on ``sys.path`` so ``import fastwam`` works without pip.

Project root is inferred as the parent of the ``scripts/`` directory containing this
file — no absolute paths, safe if the repo is moved or symlinked.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure() -> Path:
    """Insert ``<project_root>/src`` at the front of ``sys.path`` if not already present.

    Returns the resolved project root (parent of ``scripts/``).
    """
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    src = project_root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return project_root
