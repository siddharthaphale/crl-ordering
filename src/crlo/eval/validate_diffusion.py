"""Evaluate a saved diffusion policy on the official validation split."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crlo.envs.ogbench_loader import load_crl_env_and_dataset
from crlo.training.checkpoints import load_diffusion
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import RunLogger


def validate_diffusion(configuration: ConfigMap) -> int:
    """Measure denoising error from model inference on disjoint data."""
    cfg = configuration
    if cfg.n_samples < 1 or cfg.max_denoising_loss <= 0:
        raise ValueError("Validation sample count and loss bound must be positive")
    np.random.seed(cfg.seed)
    env, dataset = load_crl_env_and_dataset(
        cfg.task, cfg.discount, split="validation", dataset_dir=cfg.dataset_dir
    )
    env.close()
    policy = load_diffusion(Path(cfg.checkpoint_dir), seed=cfg.seed)
    batch = dataset.sample(cfg.n_samples)
    losses = -policy.negative_denoising_loss(
        batch["actions"], batch["observations"], batch["value_goals"]
    )
    value = float(np.mean(losses))
    passed = bool(np.isfinite(losses).all() and value <= cfg.max_denoising_loss)
    RunLogger(cfg.output_dir).log_event(
        "validation",
        {
            "validation_denoising_loss": value,
            "n_samples": cfg.n_samples,
            "split": "official_validation",
            "seed": cfg.seed,
            "max_denoising_loss": cfg.max_denoising_loss,
            "passed": passed,
            "protocol_version": 2,
        },
    )
    return 0 if passed else 1
