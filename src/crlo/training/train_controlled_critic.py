"""Train controlled critics once for geometry, selection, and candidate-count evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from crlo.agents.critic_factory import build_critic
from crlo.envs.toy_2d import Toy2DConfig, Toy2DEnv
from crlo.training.checkpoints import save_critic
from crlo.training.controlled_data import sample_controlled_batch
from crlo.utils.config_loader import ConfigMap


def train_controlled_critic(configuration: ConfigMap) -> int:
    """Train declared critics and save architecture plus sampling provenance."""
    from crlo.training.config_schemas import ControlledCriticConfig, CriticModelConfig

    cfg = ControlledCriticConfig.from_mapping(configuration)
    model = CriticModelConfig(**cfg.model)
    env_cfg = Toy2DConfig()
    if env_cfg.noise_std != 0:
        raise ValueError("Exact controlled values require noise_std=0")
    data = Toy2DEnv(config=env_cfg, seed=cfg.seed).sample_offline_data(cfg.n_train, cfg.seed)
    arrays = [data[k] for k in ("states", "actions", "goals", "next_states", "rewards")]
    if cfg.seed_protocol not in ("geometry", "selection"):
        raise ValueError("seed_protocol must be geometry or selection")
    if cfg.steps < 1 or cfg.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    for index, critic in enumerate(cfg.critics):
        rng = np.random.default_rng(cfg.seed + index * cfg.critic_seed_stride)
        example = sample_controlled_batch(rng, *arrays, batch_size=1)
        seed = (
            cfg.seed
            if cfg.seed_protocol == "geometry"
            else cfg.seed * cfg.model_seed_stride + index
        )
        agent = build_critic(
            critic,
            seed,
            example,
            cfg.bc_alpha,
            cfg.discount,
            hidden_dims=tuple(model.hidden_dims),
            latent_dim=model.latent_dim,
            model_config=model.agent_config(critic, batch_size=cfg.batch_size, controlled=True),
            controlled=True,
        )
        for step in range(cfg.steps):
            agent, info = agent.update(
                sample_controlled_batch(rng, *arrays, batch_size=cfg.batch_size)
            )
            if not all(np.isfinite(np.asarray(v)).all() for v in info.values()):
                raise FloatingPointError(f"Nonfinite update for {critic} at {step}")
        output = Path(cfg.output_dir) / critic
        save_critic(
            agent,
            output,
            {
                "critic": critic,
                "seed": seed,
                "steps": cfg.steps,
                "obs_dim": 2,
                "action_dim": 2,
                "bc_alpha": cfg.bc_alpha,
                "discount": cfg.discount,
                "hidden_dims": tuple(model.hidden_dims),
                "latent_dim": model.latent_dim,
                "controlled": True,
                "environment": asdict(env_cfg),
                "seed_protocol": cfg.seed_protocol,
                "sampling_rng": rng.bit_generator.state,
            },
        )
        print(f"Saved {critic} after {cfg.steps} steps to {output}", flush=True)
    return 0
