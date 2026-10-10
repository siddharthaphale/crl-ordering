"""#4 Gate G3: the IQL/Bellman machinery learns a known value.

discount=0 -> target = r = -||s+a-g|| (exact). Train the variant-A head (q_td on
raw [s,a,g]) on toy transitions and require kendall(q_td, Q*) > 0.85 plus a TD
residual drop. Isolates the TD code from embedding quality (raw input can
represent Q*). discount=0 means this validates the IQL machinery, not multi-step
bootstrapping (that is G7/G8).
"""

import numpy as np
from scipy.stats import kendalltau

from crlo.agents.two_head_critic import make_two_head_critic
from crlo.envs.toy_2d import Toy2DConfig, Toy2DEnv
from crlo.eval.critic_scoring import critic_score
from crlo.training.controlled_data import sample_controlled_batch


def test_td_learns_known_value():
    cfg = Toy2DConfig()
    env = Toy2DEnv(config=cfg, seed=0)
    d = env.sample_offline_data(n=20_000, seed=0)
    S, A, G, NS, Rw = (d["states"], d["actions"], d["goals"], d["next_states"], d["rewards"])
    ex = sample_controlled_batch(np.random.default_rng(0), S, A, G, NS, Rw, batch_size=1)
    agent = make_two_head_critic(
        0,
        ex,
        bc_alpha=3.0,
        discount=0.0,
        variant="two_head_separate_encoders",
        hidden_dims=(64, 64),
        latent_dim=32,
    )

    rng = np.random.default_rng(1)
    losses = []
    for _ in range(5000):
        b = sample_controlled_batch(rng, S, A, G, NS, Rw, batch_size=256)
        agent, info = agent.update(b)
        losses.append(float(info["critic_td/critic_td_loss"]))

    er = np.random.default_rng(99)
    s = er.uniform(cfg.state_low, cfg.state_high, (500, 2)).astype(np.float32)
    a = np.clip(er.normal(scale=0.5, size=(500, 2)), cfg.action_low, cfg.action_high).astype(
        np.float32
    )
    g = er.uniform(cfg.state_low, cfg.state_high, (500, 2)).astype(np.float32)
    qstar = -np.linalg.norm(s + a - g, axis=1)
    qhat = critic_score(agent, "two_head_separate_encoders", s, g, a)

    kt = float(kendalltau(qhat, qstar).statistic)
    first, last = np.mean(losses[:500]), np.mean(losses[-500:])
    assert kt > 0.85, f"kendall(q_td,Q*)={kt:.3f} (want >0.85)"
    assert last < 0.5 * first, f"TD residual {first:.3f}->{last:.3f}"
