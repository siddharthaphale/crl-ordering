"""One train step decreases loss on a simple synthetic batch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior


def _toy_batch(
    action_dim: int, state_dim: int, goal_dim: int, batch_size: int, seed: int
) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observations": jnp.asarray(rng.normal(size=(batch_size, state_dim))),
        "actions": jnp.asarray(rng.normal(size=(batch_size, action_dim))),
        "value_goals": jnp.asarray(rng.normal(size=(batch_size, goal_dim))),
    }


def test_single_train_step_changes_params_and_returns_finite_loss() -> None:
    policy = DiffusionQLBehavior.create(
        seed=0,
        action_dim=2,
        state_dim=2,
        goal_dim=2,
        denoising_steps=4,
        hidden_dims=(16,),
    )
    batch = _toy_batch(2, 2, 2, batch_size=32, seed=1)
    before = jax.tree_util.tree_map(lambda p: jnp.asarray(p), policy.params)
    info = policy.update(batch)
    assert np.isfinite(info["ddpm_loss"])
    after = policy.params
    diffs = jax.tree_util.tree_map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))),
        after,
        before,
    )
    assert any(d > 0 for d in jax.tree_util.tree_leaves(diffs))


def test_loss_decreases_over_multiple_steps() -> None:
    policy = DiffusionQLBehavior.create(
        seed=0,
        action_dim=2,
        state_dim=2,
        goal_dim=2,
        denoising_steps=4,
        hidden_dims=(16, 16),
        lr=1e-3,
    )
    batch = _toy_batch(2, 2, 2, batch_size=64, seed=2)
    losses = []
    for _ in range(20):
        losses.append(policy.update(batch)["ddpm_loss"])
    assert losses[-1] < losses[0]
