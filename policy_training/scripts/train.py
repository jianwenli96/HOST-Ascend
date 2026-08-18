import _ensure_project_src

_ensure_project_src.ensure()

import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception:
    pass

import hydra
from omegaconf import DictConfig

from self_grounded_prediction.runtime import run_training
from self_grounded_prediction.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
