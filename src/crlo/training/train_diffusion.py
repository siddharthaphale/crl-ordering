"""Train a DDPM policy and measure denoising loss on disjoint validation data."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.diffusion_behavior_agent import (  # noqa: E402
    make_diffusion_behavior_agent,
)
from crlo.training.config_schemas import TrainDiffusionConfig


def _train_real(cfg: TrainDiffusionConfig) -> int:
    from crlo.envs.ogbench_loader import load_crl_env_and_dataset

    _, train_dataset = load_crl_env_and_dataset(
        env_name=cfg.task,
        discount=cfg.discount,
        dataset_dir=cfg.dataset_dir,
    )
    example = train_dataset.sample(8)
    agent = make_diffusion_behavior_agent(
        seed=cfg.seed,
        example_batch=example,
        denoising_steps=cfg.denoising_steps,
        lr=cfg.lr,
        hidden_dims=tuple(cfg.hidden_dims),
        schedule=cfg.schedule,
        time_embed_dim=cfg.time_embed_dim,
    )

    print(
        f"[train_diffusion] task={cfg.task} steps={cfg.steps} "
        f"batch={cfg.batch_size} T={cfg.denoising_steps} seed={cfg.seed}",
        flush=True,
    )
    start = time.time()
    last_loss = float("nan")
    for step in range(1, cfg.steps + 1):
        batch = train_dataset.sample(cfg.batch_size)
        info = agent.update(batch)
        last_loss = info["ddpm_loss"]
        if step % cfg.log_interval == 0:
            elapsed = (time.time() - start) / 60.0
            print(
                f"Step {step:>7} | ddpm_loss: {last_loss:.4f} | elapsed: {elapsed:.1f}min",
                flush=True,
            )

    validation_env, validation_dataset = load_crl_env_and_dataset(
        env_name=cfg.task,
        discount=cfg.discount,
        split="validation",
        dataset_dir=cfg.dataset_dir,
    )
    validation_env.close()
    train_batch = train_dataset.sample(cfg.validation_samples)
    holdout_batch = validation_dataset.sample(cfg.validation_samples)
    train_denoising_loss = -float(
        agent.negative_denoising_loss(
            train_batch["actions"],
            train_batch["observations"],
            train_batch["value_goals"],
        ).mean()
    )
    validation_denoising_loss = -float(
        agent.negative_denoising_loss(
            holdout_batch["actions"],
            holdout_batch["observations"],
            holdout_batch["value_goals"],
        ).mean()
    )
    gap = validation_denoising_loss - train_denoising_loss

    from crlo.training.checkpoints import save_diffusion  # noqa: PLC0415
    from crlo.utils.logging import RunLogger

    save_diffusion(
        agent,
        cfg.output_dir,
        meta={"task": cfg.task, "seed": cfg.seed, "steps": cfg.steps},
    )
    RunLogger(cfg.output_dir).log_event(
        "validation",
        {
            "train_denoising_loss": train_denoising_loss,
            "validation_denoising_loss": validation_denoising_loss,
            "gap": gap,
            "split": "official_validation",
            "protocol_version": 2,
            "final_train_loss": last_loss,
        },
    )
    print(
        f"[train_diffusion] done. train_denoising_loss={train_denoising_loss:.4f} "
        f"validation_denoising_loss={validation_denoising_loss:.4f} gap={gap:.4f}",
        flush=True,
    )
    return 0


def train_diffusion(configuration: ConfigMap) -> int:
    """Train a real diffusion proposal and validate on the official split."""
    cfg = TrainDiffusionConfig.from_mapping(
        configuration,
    )
    np.random.seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return _train_real(cfg)
