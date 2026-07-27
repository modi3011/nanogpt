"""
Train the stories model.

Usage:
    python -m models.stories.train
    python -m models.stories.train configs/stories/train_config.py
    python -m models.stories.train --max_iters=5000 --batch_size=64

This script can also be run under torchrun for DDP, exactly as the original
train.py supported.
"""

import os
import pickle

from core.data_utils import get_batch_aligned
from core.trainer import run_training
from core.utils import default_dtype, load_config
from models.stories.model import default_model_args

DATA_DIR = os.path.join("data", "stories")

DEFAULT_CONFIG = {
    # I/O
    "out_dir": os.path.join("checkpoints", "stories"),
    "eval_interval": 200,
    "log_interval": 10,
    "eval_iters": 200,
    "eval_only": False,
    "always_save_checkpoint": True,
    "init_from": "scratch",  # 'scratch' | 'resume' | 'gpt2*'

    # wandb
    "wandb_log": False,
    "wandb_project": "nanogpt-stories",
    "wandb_run_name": "stories",

    # data / batching
    "gradient_accumulation_steps": 1,
    "batch_size": 128,
    "block_size": 128,

    # optimizer
    "learning_rate": 2.5e-4,
    "max_iters": 12000,
    "weight_decay": 0.06,
    "beta1": 0.9,
    "beta2": 0.99,
    "grad_clip": 1.0,

    # LR schedule
    "decay_lr": True,
    "warmup_iters": 200,
    "lr_decay_iters": 12000,
    "min_lr": 2e-5,

    # DDP
    "backend": "nccl",

    # system
    "device": "cuda",
    "dtype": default_dtype(),
    "compile": False,
    "seed": 1337,
}

def main() -> None:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(default_model_args())
    cfg = load_config(cfg)

    # attempt to derive vocab_size from data/stories/meta.pkl, if present
    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # batch sampler aligned to story boundaries (so windows tend to start at
    # the beginning of a story rather than mid-story)
    start_positions_cache: dict = {}

    def get_batch_fn(split: str):
        return get_batch_aligned(
            data_dir=DATA_DIR,
            split=split,
            batch_size=cfg["batch_size"],
            block_size=cfg["block_size"],
            device=cfg["device"],
            device_type="cuda" if "cuda" in cfg["device"] else "cpu",
            start_positions_cache=start_positions_cache,
        )

    run_training(cfg, get_batch_fn, meta_vocab_size=meta_vocab_size)


if __name__ == "__main__":
    main()
