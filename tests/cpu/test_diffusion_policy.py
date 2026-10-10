"""DiffusionQLBehavior satisfies the sampler protocol on small-K, small-T."""

from __future__ import annotations

import numpy as np

from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior
from crlo.agents.diffusion_behavior_agent import make_diffusion_behavior_agent
from crlo.eval.diffusion_sampler import DiffusionSampler


def test_policy_create_and_sample_shapes() -> None:
    policy = DiffusionQLBehavior.create(
        seed=0,
        action_dim=4,
        state_dim=3,
        goal_dim=3,
        denoising_steps=4,
        hidden_dims=(16,),
    )
    s = np.zeros(3)
    g = np.ones(3)
    samples = policy.sample(s, g, K=8)
    assert samples.shape == (8, 4)
    ll = policy.negative_denoising_loss(samples, s, g)
    assert ll.shape == (8,)


def test_factory_non_placeholder_returns_diffusion_ql() -> None:
    agent = make_diffusion_behavior_agent(
        seed=0,
        action_dim=4,
        state_dim=3,
        goal_dim=3,
        denoising_steps=2,
    )
    assert isinstance(agent, DiffusionSampler)
    assert agent.action_dim == 4


def test_policy_inferred_dims_from_example_batch() -> None:
    example_batch = {
        "actions": np.zeros((1, 5)),
        "observations": np.zeros((1, 6)),
        "value_goals": np.zeros((1, 6)),
    }
    agent = make_diffusion_behavior_agent(
        seed=0,
        example_batch=example_batch,
        denoising_steps=2,
    )
    assert agent.action_dim == 5
