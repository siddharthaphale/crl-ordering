"""#4 Gate G1: two-head agent builds, steps, and exposes a free scalar q_td.

Checks (per variant): modules q_td/v_td/target_q_td present; one update returns a
finite loss; q_td forward (2,B) and v_td forward (B,); critic_score routes through
q_td (== mean q_td, != v_td); q_td is a free-scalar GCValue MLP (structural).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from crlo.agents.two_head_critic import make_two_head_critic
from crlo.eval.critic_scoring import critic_score

VARIANTS = [
    "two_head_separate_encoders",
    "two_head_shared_encoder_stopped_gradient",
    "two_head_shared_encoder",
    "two_head_shared_encoder_td_only",
]
OBS_D, ACT_D, B = 4, 2, 16


def _batch(seed=0):
    rng = np.random.default_rng(seed)

    def f(d):
        return rng.standard_normal((B, d)).astype(np.float32)

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
        0,
        ex,
        bc_alpha=3.0,
        discount=0.99,
        variant=variant,
        hidden_dims=(64, 64),
        latent_dim=32,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_build_step_and_routing(variant):
    agent = _agent(variant)
    p = agent.network.params
    for m in ("modules_q_td", "modules_v_td", "modules_target_q_td"):
        assert m in p, f"{variant}: missing {m}"
    # Free scalar: q_td is a GCValue MLP (has value_net), not a cosine/bilinear op.
    assert "value_net" in str(p["modules_q_td"]), f"{variant}: q_td not a free MLP"

    b = _batch()
    agent2, info = agent.update(b)
    assert np.isfinite(float(info["total_loss"])), f"{variant}: non-finite loss"

    sj = jnp.asarray(b["observations"])
    gj = jnp.asarray(b["value_goals"])
    aj = jnp.asarray(b["actions"])
    q = np.asarray(agent2._q_td(sj, gj, aj))
    assert q.shape == (2, B), f"{variant}: q_td shape {q.shape}"
    v = np.asarray(agent2.network.select("v_td")(sj, gj))
    assert v.shape == (B,), f"{variant}: v_td shape {v.shape}"

    cs = critic_score(agent2, variant, b["observations"], b["value_goals"], b["actions"])
    assert np.allclose(cs, q.mean(0), atol=1e-5), f"{variant}: score != q_td"
    assert not np.allclose(cs, v, atol=1e-3), f"{variant}: score routed via v_td"
