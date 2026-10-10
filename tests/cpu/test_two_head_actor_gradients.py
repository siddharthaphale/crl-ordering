"""Option-B Seed replication1: the AWR actor term updates ONLY the actor.

grad of actor_loss w.r.t. modules_actor > 1e-6; w.r.t. modules_q_td, modules_v_td,
modules_critic, modules_target_q_td all < 1e-8 (the advantage is a fixed signal,
so policy learning cannot perturb the value head / calibration).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crlo.agents.two_head_critic import make_two_head_critic

OBS_D, ACT_D, B = 4, 2, 16
ZERO, NONZERO = 1e-8, 1e-6
FULL = [
    "two_head_separate_encoders_with_actor",
    "two_head_shared_encoder_stopped_gradient_with_actor",
    "two_head_shared_encoder_with_actor",
]


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda d: jnp.asarray(rng.standard_normal((B, d)), jnp.float32)  # noqa: E731
    return {
        "observations": f(OBS_D),
        "actions": f(ACT_D),
        "next_observations": f(OBS_D),
        "value_goals": f(OBS_D),
        "actor_goals": f(OBS_D),
        "rewards": jnp.asarray(rng.standard_normal(B), jnp.float32),
        "masks": jnp.ones(B, jnp.float32),
    }


def _agent(variant):
    ex = {
        "observations": np.zeros((1, OBS_D), np.float32),
        "actions": np.zeros((1, ACT_D), np.float32),
    }
    return make_two_head_critic(
        0, ex, bc_alpha=3.0, discount=0.99, variant=variant, hidden_dims=(64, 64), latent_dim=32
    )


def _norm(grad, key):
    leaves = jax.tree_util.tree_leaves(grad[key])
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in leaves)))


@pytest.mark.parametrize("variant", FULL)
def test_actor_grad_isolation(variant):
    agent = _agent(variant)
    b = _batch()
    g = jax.grad(lambda p: agent.actor_loss(b, p)[0])(agent.network.params)
    assert _norm(g, "modules_actor") > NONZERO, f"{variant}: actor not trained"
    for head in ("modules_q_td", "modules_v_td", "modules_critic", "modules_target_q_td"):
        assert _norm(g, head) < ZERO, (
            f"{variant}: actor loss perturbs {head} (calibration drift risk)"
        )
