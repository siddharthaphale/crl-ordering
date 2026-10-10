"""Actual CUDA updates and checkpoint score parity for every retained critic."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from crlo.agents.critic_factory import build_critic, build_critic_from_metadata
from crlo.agents.model_names import CRITICS
from crlo.eval.critic_scoring import critic_score
from crlo.training.checkpoints import load_critic, save_critic


@pytest.mark.parametrize("critic", CRITICS)
def test_gpu_update_and_checkpoint(critic, tmp_path):
    rng = np.random.default_rng(19)
    obs = rng.normal(size=(8, 3)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
    batch = dict(
        observations=obs,
        actions=actions,
        next_observations=obs * 0.9,
        value_goals=obs[::-1].copy(),
        actor_goals=obs[::-1].copy(),
        rewards=np.zeros(8, np.float32),
        masks=np.ones(8, np.float32),
    )
    agent = build_critic(
        critic,
        17,
        batch,
        3.0,
        0.99,
        hidden_dims=(16, 16),
        latent_dim=8,
        model_config={"td_warmup_steps": 2} if critic == "hybrid" else None,
    )
    before = jax.tree.leaves(agent.network.params)
    for step in range(3):
        agent, info = agent.update(batch)
        if critic == "hybrid":
            assert float(info["td_weight"]) == float(step >= 1)
        assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    after = jax.tree.leaves(agent.network.params)
    assert all(next(iter(v.devices())).platform == "gpu" for v in after)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    scores = critic_score(agent, critic, obs, batch["value_goals"], actions)
    save_critic(
        agent,
        tmp_path,
        dict(
            critic=critic,
            seed=17,
            bc_alpha=3.0,
            discount=0.99,
            obs_dim=3,
            action_dim=2,
            hidden_dims=(16, 16),
            latent_dim=8,
        ),
    )
    restored, _ = load_critic(tmp_path, build_critic_from_metadata, resume=True)
    np.testing.assert_allclose(
        critic_score(restored, critic, obs, batch["value_goals"], actions), scores, atol=1e-6
    )
    expected, _ = agent.update(batch)
    actual, _ = restored.update(batch)
    for a, b in zip(
        jax.tree.leaves(expected.network.params),
        jax.tree.leaves(actual.network.params),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "critic,expect_gradient",
    [
        ("two_head_separate_encoders", False),
        ("two_head_shared_encoder_stopped_gradient", False),
        ("two_head_shared_encoder", True),
        ("two_head_shared_encoder_td_only", True),
    ],
)
def test_cuda_two_head_gradient_isolation(critic, expect_gradient):
    rng = np.random.default_rng(11)
    observations = rng.normal(size=(8, 3)).astype(np.float32)
    batch = dict(
        observations=observations,
        next_observations=observations[::-1].copy(),
        actions=rng.normal(size=(8, 2)).astype(np.float32),
        value_goals=np.zeros_like(observations),
        actor_goals=observations,
        rewards=np.ones(8, np.float32),
        masks=np.ones(8, np.float32),
    )
    agent = build_critic(critic, 3, batch, 3.0, 0.99, hidden_dims=(16,), latent_dim=8)
    gradient = jax.grad(
        lambda params: agent.value_loss(batch, params)[0] + agent.critic_loss(batch, params)[0]
    )(agent.network.params)

    def norm(key):
        return sum(float(np.square(np.asarray(x)).sum()) for x in jax.tree.leaves(gradient[key]))

    assert (norm("modules_critic") > 1e-12) == expect_gradient
    assert norm("modules_target_q_td") < 1e-16
