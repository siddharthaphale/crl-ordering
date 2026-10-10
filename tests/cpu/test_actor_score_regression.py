"""Actor addition preserves the original critic scores and value-head baselines."""

from pathlib import Path

import numpy as np
import pytest

from crlo.agents.critic_factory import build_critic
from crlo.eval.critic_scoring import critic_score

GOLD = np.load(Path(__file__).with_name("critic_score_baselines.npz"))
S, G, A = GOLD["s"], GOLD["g"], GOLD["a"]
ORIGINAL = [
    "raw_bilinear",
    "cosine",
    "td_q",
    "hybrid",
    "two_head_separate_encoders",
    "two_head_shared_encoder_stopped_gradient",
    "two_head_shared_encoder",
    "two_head_shared_encoder_td_only",
]
FULL_TO_BASE = {
    "two_head_separate_encoders_with_actor": "two_head_separate_encoders",
    "two_head_shared_encoder_stopped_gradient_with_actor": "two_head_shared_encoder_stopped_gradient",
    "two_head_shared_encoder_with_actor": "two_head_shared_encoder",
    "two_head_shared_encoder_td_only_with_actor": "two_head_shared_encoder_td_only",
}
EX = {
    "observations": np.zeros((1, S.shape[1]), np.float32),
    "actions": np.zeros((1, A.shape[1]), np.float32),
}


def _score(cid):
    ag = build_critic(cid, seed=0, example_batch=EX, bc_alpha=3.0, discount=0.99)
    return critic_score(ag, cid, S, G, A)


@pytest.mark.parametrize("cid", ORIGINAL)
def test_original_goldens_unchanged(cid):
    assert np.allclose(_score(cid), GOLD[cid], atol=1e-5), f"{cid}: drifted"


@pytest.mark.parametrize("full,base", FULL_TO_BASE.items())
def test_full_score_equals_base_golden(full, base):
    # q_td score path is identical to the base variant at init (actor additive).
    assert np.allclose(_score(full), GOLD[base], atol=1e-5), (
        f"{full}: deployed q_td differs from {base}"
    )
