"""The public factory builds a real trainable policy with explicit dimensions."""

from __future__ import annotations

import numpy as np
import pytest

from crlo.agents.diffusion_behavior_agent import make_diffusion_behavior_agent


def test_dimensions_and_optimizer_are_forwarded():
    batch = {
        "actions": np.zeros((1, 2)),
        "observations": np.zeros((1, 3)),
        "value_goals": np.zeros((1, 3)),
    }
    policy = make_diffusion_behavior_agent(
        0, batch, denoising_steps=2, hidden_dims=(8,), lr=0.002, schedule="linear"
    )
    assert (policy.action_dim, policy.state_dim, policy.goal_dim) == (2, 3, 3)
    assert policy.lr == 0.002
    assert policy.schedule == "linear"


def test_missing_dimensions_are_rejected():
    with pytest.raises(ValueError, match="action_dim"):
        make_diffusion_behavior_agent(0)
