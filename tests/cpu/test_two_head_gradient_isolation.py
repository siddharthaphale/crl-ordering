"""#4 Gate G4: gradient routing & target isolation.

Because v_td uses raw [s,g], the TD gradient into the contrastive encoder comes
only through the q_td critic-loss path. Per variant, grad of the IQL value+critic
loss w.r.t. the contrastive `critic` (phi,psi) params:
  A:0  B1:0  B2:>0  B0:>0.
Also: contrastive loss does not update q_td/v_td/target_q_td; target_q_td gets no
gradient from any loss (changes only via target_update).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crlo.agents.two_head_critic import make_two_head_critic

OBS_D, ACT_D, B = 4, 2, 16
ZERO, NONZERO = 1e-8, 1e-6


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


def _subtree_norm(grad, key):
    leaves = jax.tree_util.tree_leaves(grad[key])
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in leaves)))


@pytest.mark.parametrize(
    "variant,expect_grad",
    [
        ("two_head_separate_encoders", False),
        ("two_head_shared_encoder_stopped_gradient", False),
        ("two_head_shared_encoder", True),
        ("two_head_shared_encoder_td_only", True),
    ],
)
def test_td_gradient_into_contrastive_encoder(variant, expect_grad):
    agent = _agent(variant)
    b = _batch()

    def td_loss(params):
        return agent.value_loss(b, params)[0] + agent.critic_loss(b, params)[0]

    g = jax.grad(td_loss)(agent.network.params)
    n_critic = _subtree_norm(g, "modules_critic")
    if expect_grad:
        assert n_critic > NONZERO, f"{variant}: TD grad into encoder={n_critic}"
    else:
        assert n_critic < ZERO, f"{variant}: leaked TD grad into encoder={n_critic}"
    # target_q_td must receive no gradient from the TD loss
    assert _subtree_norm(g, "modules_target_q_td") < ZERO, (
        f"{variant}: target_q_td received TD gradient"
    )


@pytest.mark.parametrize(
    "variant", ["two_head_shared_encoder_stopped_gradient", "two_head_shared_encoder"]
)
def test_contrastive_loss_does_not_touch_td_heads(variant):
    agent = _agent(variant)
    b = _batch()

    def c_loss(params):
        return agent.contrastive_loss(b, params, "critic")[0]

    g = jax.grad(c_loss)(agent.network.params)
    for head in ("modules_q_td", "modules_v_td", "modules_target_q_td"):
        assert _subtree_norm(g, head) < ZERO, f"{variant}: contrastive loss updated {head}"
