"""E4 function-class control (campaign 4, C4.3a): construction, scoring,
parameter matching, and one-step training sanity.

The 2x2 the control completes:
  contrastive x bilinear  = raw_bilinear (existing)
  contrastive x MLP       = contrastive_mlp (new; must match td_q params)
  TD x bilinear           = td_bilinear (new; must match raw_bilinear's
                            deployed-scalar params)
  TD x MLP                = td_q / GCIQL (existing)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crlo.agents.critic_factory import build_critic
from crlo.eval.critic_scoring import critic_score

S_DIM, A_DIM, B = 6, 3, 8
EX = {"observations": np.zeros((1, S_DIM), np.float32), "actions": np.zeros((1, A_DIM), np.float32)}


def _build(cid):
    return build_critic(cid, seed=0, example_batch=EX, bc_alpha=3.0, discount=0.99)


def _rand(n, d, seed):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _param_count(tree):
    return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize("cid", ["td_bilinear", "contrastive_mlp"])
def test_scores_finite_and_shaped(cid):
    agent = _build(cid)
    s, g, a = _rand(B, S_DIM, 0), _rand(B, S_DIM, 1), _rand(B, A_DIM, 2)
    cs = critic_score(agent, cid, s, g, a)
    assert cs.shape == (B,)
    assert np.all(np.isfinite(cs))


def test_td_bilinear_score_is_ensemble_mean_bilinear():
    agent = _build("td_bilinear")
    s, g, a = _rand(B, S_DIM, 3), _rand(B, S_DIM, 4), _rand(B, A_DIM, 5)
    cs = critic_score(agent, "td_bilinear", s, g, a)
    v = np.asarray(
        agent.network.select("critic")(jnp.asarray(s), jnp.asarray(g), actions=jnp.asarray(a))
    )
    assert v.shape == (2, B)  # twin (ensemble) bilinear
    assert np.allclose(cs, v.mean(0), atol=1e-6)


def test_contrastive_mlp_score_is_ensemble_mean_logit():
    agent = _build("contrastive_mlp")
    s, g, a = _rand(B, S_DIM, 6), _rand(B, S_DIM, 7), _rand(B, A_DIM, 8)
    cs = critic_score(agent, "contrastive_mlp", s, g, a)
    q1, q2 = agent.network.select("critic")(jnp.asarray(s), jnp.asarray(g), jnp.asarray(a))
    assert np.allclose(cs, (np.asarray(q1) + np.asarray(q2)) / 2, atol=1e-6)


def test_contrastive_mlp_params_match_td_q():
    """The new contrastive cell must be parameter-matched to deployed TD-Q's
    critic (GCIQL twin-Q MLP on concat(s, g, a)) — that IS the control."""
    mlp = _build("contrastive_mlp")
    gciql = _build("td_q")
    n_mlp = _param_count(mlp.network.params["modules_critic"])
    n_tdq = _param_count(gciql.network.params["modules_critic"])
    assert n_mlp == n_tdq, (n_mlp, n_tdq)


def test_td_bilinear_params_match_raw_bilinear():
    """The new TD cell must have the raw_bilinear deployed scalar's function
    class: identical GCBilinearValue parameterization."""
    tdb = _build("td_bilinear")
    raw = _build("raw_bilinear")
    n_tdb = _param_count(tdb.network.params["modules_critic"])
    n_raw = _param_count(raw.network.params["modules_critic"])
    assert n_tdb == n_raw, (n_tdb, n_raw)


@pytest.mark.parametrize("cid", ["td_bilinear", "contrastive_mlp"])
def test_one_update_step_changes_critic_and_is_finite(cid):
    agent = _build(cid)
    rng = np.random.default_rng(9)
    batch = {
        "observations": rng.standard_normal((B, S_DIM)).astype(np.float32),
        "next_observations": rng.standard_normal((B, S_DIM)).astype(np.float32),
        "actions": rng.standard_normal((B, A_DIM)).astype(np.float32),
        "value_goals": rng.standard_normal((B, S_DIM)).astype(np.float32),
        "rewards": rng.integers(0, 2, B).astype(np.float32),
        "masks": np.ones(B, np.float32),
    }
    before = jax.tree_util.tree_map(
        lambda p: np.asarray(p).copy(), agent.network.params["modules_critic"]
    )
    agent2, info = agent.update(batch)
    assert np.isfinite(float(info["total_loss"]))
    after = agent2.network.params["modules_critic"]
    diffs = jax.tree_util.tree_map(
        lambda a_, b_: float(np.abs(np.asarray(a_) - np.asarray(b_)).max()), after, before
    )
    assert max(jax.tree_util.tree_leaves(diffs)) > 0.0
