"""CUDA diffusion optimizer wiring, deterministic sampling, and exact resume."""

from __future__ import annotations

import jax
import numpy as np

from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior
from crlo.eval.candidate_returns import (
    DeterministicDiffusionContinuation,
    StochasticDiffusionContinuation,
)
from crlo.training.checkpoints import load_diffusion, save_diffusion


def test_diffusion_resume_equivalence(tmp_path):
    policy = DiffusionQLBehavior.create(5, 2, 3, 3, 4, (16,), lr=0.002, schedule="linear")
    batch = {
        "observations": np.ones((8, 3), np.float32),
        "value_goals": np.zeros((8, 3), np.float32),
        "actions": np.full((8, 2), 0.2, np.float32),
    }
    for _ in range(3):
        assert np.isfinite(policy.update(batch)["ddpm_loss"])
    save_diffusion(policy, tmp_path)
    restored = load_diffusion(tmp_path, resume=True)
    assert restored.lr == 0.002 and restored.schedule == "linear" and restored.steps == 3
    assert all(next(iter(x.devices())).platform == "gpu" for x in jax.tree.leaves(restored.params))
    policy.update(batch)
    restored.update(batch)
    for a, b in zip(jax.tree.leaves(policy.params), jax.tree.leaves(restored.params), strict=True):
        np.testing.assert_array_equal(a, b)
    s, g = batch["observations"][0], batch["value_goals"][0]
    np.testing.assert_array_equal(policy.sample(s, g, 4), restored.sample(s, g, 4))
    deterministic = DeterministicDiffusionContinuation(policy)
    np.testing.assert_array_equal(deterministic.sample(s, g, 1), deterministic.sample(s, g, 1))
    first, second = StochasticDiffusionContinuation(policy), StochasticDiffusionContinuation(policy)
    np.testing.assert_array_equal(first.sample(s, g, 4), second.sample(s, g, 4))


def test_configured_learning_rate_changes_update():
    slow = DiffusionQLBehavior.create(5, 2, 3, 3, 4, (16,), lr=0.0001)
    fast = DiffusionQLBehavior.create(5, 2, 3, 3, 4, (16,), lr=0.001)
    initial = [np.asarray(value).copy() for value in jax.tree.leaves(slow.params)]
    batch = dict(
        observations=np.ones((8, 3), np.float32),
        value_goals=np.zeros((8, 3), np.float32),
        actions=np.full((8, 2), 0.2, np.float32),
    )
    slow.update(batch)
    fast.update(batch)

    def displacement(policy):
        return np.sqrt(
            sum(
                np.square(np.asarray(after) - before).sum()
                for before, after in zip(initial, jax.tree.leaves(policy.params), strict=True)
            )
        )

    np.testing.assert_allclose(displacement(fast) / displacement(slow), 10.0, rtol=1e-3)
