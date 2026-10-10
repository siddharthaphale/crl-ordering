"""Option-B Seed replication0: AWR actor wired; agent still steps; back-compat.

*_full variants: one update returns a finite total loss including an `actor/...`
term. Base (#4) variants: no actor term (train_actor=False) -> back-compat.
"""

import numpy as np
import pytest

from crlo.agents.two_head_critic import make_two_head_critic

OBS_D, ACT_D, B = 4, 2, 16
FULL = [
    "two_head_separate_encoders_with_actor",
    "two_head_shared_encoder_stopped_gradient_with_actor",
    "two_head_shared_encoder_with_actor",
]
BASE = ["two_head_separate_encoders", "two_head_shared_encoder"]


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda d: rng.standard_normal((B, d)).astype(np.float32)  # noqa: E731
    return {
        "observations": f(OBS_D),
        "actions": f(ACT_D),
        "next_observations": f(OBS_D),
        "value_goals": f(OBS_D),
        "actor_goals": f(OBS_D),
        "rewards": rng.standard_normal(B).astype(np.float32),
        "masks": np.ones(B, np.float32),
    }


def _agent(variant):
    ex = {
        "observations": np.zeros((1, OBS_D), np.float32),
        "actions": np.zeros((1, ACT_D), np.float32),
    }
    return make_two_head_critic(
        0, ex, bc_alpha=3.0, discount=0.99, variant=variant, hidden_dims=(64, 64), latent_dim=32
    )


@pytest.mark.parametrize("variant", FULL)
def test_full_variant_trains_actor(variant):
    agent, info = _agent(variant).update(_batch())
    assert np.isfinite(float(info["total_loss"])), f"{variant}: non-finite"
    assert "actor/actor_loss" in info, f"{variant}: no actor term"
    assert np.isfinite(float(info["actor/actor_loss"]))


@pytest.mark.parametrize("variant", BASE)
def test_base_variant_back_compat(variant):
    agent, info = _agent(variant).update(_batch())
    assert np.isfinite(float(info["total_loss"]))
    assert not any(k.startswith("actor/") for k in info), (
        f"{variant}: actor term leaked with train_actor=False"
    )
