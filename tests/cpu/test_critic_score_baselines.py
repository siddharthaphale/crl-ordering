"""Preserved critic-score baselines and explicit value-head readout checks."""

import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from crlo.agents.critic_factory import build_critic
from crlo.eval.critic_scoring import critic_score, normalize_embeddings

GOLD = np.load(Path(__file__).with_name("critic_score_baselines.npz"))
S, G, A = GOLD["s"], GOLD["g"], GOLD["a"]
EXISTING = ["raw_bilinear", "cosine", "td_q", "hybrid"]
TWOHEAD = [
    "two_head_separate_encoders",
    "two_head_shared_encoder_stopped_gradient",
    "two_head_shared_encoder",
    "two_head_shared_encoder_td_only",
]
EX = {
    "observations": np.zeros((1, S.shape[1]), np.float32),
    "actions": np.zeros((1, A.shape[1]), np.float32),
}


def _build(cid):
    return build_critic(cid, seed=0, example_batch=EX, bc_alpha=3.0, discount=0.99)


@pytest.mark.parametrize("cid", EXISTING + TWOHEAD)
def test_golden_regression(cid):
    cs = critic_score(_build(cid), cid, S, G, A)
    assert np.allclose(cs, GOLD[cid], atol=1e-5), f"{cid}: score drifted from golden"


def test_critic_score_never_uses_v_td():
    assert "v_td" not in inspect.getsource(critic_score), (
        "critic_score must never route a deployed score through v_td"
    )


@pytest.mark.parametrize("variant", TWOHEAD)
def test_twohead_scored_by_free_qtd(variant):
    agent = _build(variant)
    cs = critic_score(agent, variant, S, G, A)
    sj, gj, aj = jnp.asarray(S), jnp.asarray(G), jnp.asarray(A)
    # (a) equals mean-ensemble q_td via the raw network call
    q = np.asarray(agent._q_td(sj, gj, aj))
    assert np.allclose(cs, q.mean(0), atol=1e-5), f"{variant}: score != q_td"
    # (b) free scalar: not the contrastive cosine score
    _, phi, psi = agent.network.select("critic")(sj, gj, actions=aj, info=True)
    cos = np.asarray(jnp.sum(normalize_embeddings(phi) * normalize_embeddings(psi), axis=-1))
    cos = cos.mean(0) if cos.ndim == 2 else cos
    assert not np.allclose(cs, cos, atol=1e-4), f"{variant}: score == cosine"
