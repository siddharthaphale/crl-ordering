"""#4 Gate G6: checkpoint round-trip is lossless for all two-head variants.

Build (default capacity, matching the rebuild hook), step a few times so
target_q_td diverges from q_td, score a fixed batch, save, reload via
run_ranking_metrics.critic_rebuild_function, re-score: scores must match < 1e-6.
"""

import numpy as np
import pytest

from crlo.agents.critic_factory import critic_rebuild_function
from crlo.agents.two_head_critic import make_two_head_critic
from crlo.eval.critic_scoring import critic_score
from crlo.training.checkpoints import load_critic, save_critic

OBS_D, ACT_D, B = 4, 2, 12
VARIANTS = [
    "two_head_separate_encoders",
    "two_head_shared_encoder_stopped_gradient",
    "two_head_shared_encoder",
    "two_head_shared_encoder_td_only",
]


def _batch(seed):
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_roundtrip(variant, tmp_path):
    ex = {
        "observations": np.zeros((1, OBS_D), np.float32),
        "actions": np.zeros((1, ACT_D), np.float32),
    }
    # Default capacity (no hidden_dims override) to match critic_rebuild_function.
    agent = make_two_head_critic(0, ex, bc_alpha=3.0, discount=0.99, variant=variant)
    b = _batch(0)
    for _ in range(3):  # diverge target_q_td from q_td
        agent, _ = agent.update(b)

    eb = _batch(5)
    before = critic_score(agent, variant, eb["observations"], eb["value_goals"], eb["actions"])
    meta = {
        "critic": variant,
        "obs_dim": OBS_D,
        "action_dim": ACT_D,
        "bc_alpha": 3.0,
        "discount": 0.99,
    }
    save_critic(agent, tmp_path, meta)
    restored, _ = load_critic(tmp_path, critic_rebuild_function(variant))
    after = critic_score(restored, variant, eb["observations"], eb["value_goals"], eb["actions"])
    assert np.max(np.abs(before - after)) < 1e-6, f"{variant}: round-trip drift"
