"""Initialize a JAX diffusion behavior policy from explicit model parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import Array

    from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior


def make_diffusion_behavior_agent(
    seed: int,
    example_batch: Mapping[str, np.ndarray | Array] | None = None,
    action_dim: int | None = None,
    state_dim: int | None = None,
    goal_dim: int | None = None,
    denoising_steps: int | None = None,
    lr: float | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    schedule: str | None = None,
    time_embed_dim: int | None = None,
) -> DiffusionQLBehavior:
    """Initialize a diffusion behavior policy from explicit dimensions or a batch.

    Args:
        seed: JAX initialization seed.
        example_batch: Arrays with trailing feature dimensions. ``actions``,
            ``observations``, and ``value_goals`` supply missing action, state,
            and goal dimensions, respectively.
        action_dim: Positive action width; overrides the corresponding batch width.
        state_dim: Positive state width; overrides the corresponding batch width.
        goal_dim: Positive goal width; overrides the corresponding batch width.
        denoising_steps: Diffusion steps; ``None`` uses the policy default of 20.
        lr: Adam learning rate; ``None`` uses 0.0003.
        hidden_dims: MLP widths; ``None`` uses ``(256, 256)``.
        schedule: ``"cosine"`` or ``"linear"``; ``None`` uses ``"cosine"``.
        time_embed_dim: Positive even time-embedding width; ``None`` uses 32.

    Returns:
        A newly initialized policy with fresh optimizer and RNG state. No
        training or checkpoint loading occurs, and no configuration file is read.

    Raises:
        ValueError: A dimension is missing or nonpositive, the schedule is
            unsupported, or the time-embedding width is invalid.
    """
    from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior

    batch = example_batch or {}
    dimensions = {
        "action_dim": (action_dim, "actions"),
        "state_dim": (state_dim, "observations"),
        "goal_dim": (goal_dim, "value_goals"),
    }
    resolved = {}
    for name, (value, key) in dimensions.items():
        if value is None:
            if key not in batch:
                raise ValueError(f"Supply {name} or example_batch[{key!r}]")
            value = int(batch[key].shape[-1])
        if value < 1:
            raise ValueError(f"{name} must be positive")
        resolved[name] = value
    return DiffusionQLBehavior.create(
        seed=seed,
        **resolved,
        denoising_steps=denoising_steps,
        lr=lr,
        hidden_dims=hidden_dims,
        schedule=schedule,
        time_embed_dim=time_embed_dim,
    )
