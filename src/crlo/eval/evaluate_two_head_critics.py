"""Compare value and contrastive readouts of two-head critics."""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.stats import kendalltau

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.envs.ogbench_loader import load_crl_env_and_dataset  # noqa: E402
from crlo.eval.critic_scoring import critic_score, normalize_embeddings  # noqa: E402
from crlo.eval.trajectory_queries import sample_trajectory_queries  # noqa: E402
from crlo.training.checkpoints import load_critic  # noqa: E402


def _auc(pos, neg):
    """Mann-Whitney AUC = P(pos > neg)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _cosine_score(agent, s, g, a, batch=512):
    out = []
    for i in range(0, len(s), batch):
        _, phi, psi = agent.network.select("critic")(
            jnp.asarray(s[i : i + batch]),
            jnp.asarray(g[i : i + batch]),
            actions=jnp.asarray(a[i : i + batch]),
            info=True,
        )
        sc = np.asarray(jnp.sum(normalize_embeddings(phi) * normalize_embeddings(psi), axis=-1))
        out.append(sc.mean(0) if sc.ndim == 2 else sc)
    return np.concatenate(out).reshape(-1)


def _binned(score, V, n=10):
    o = np.argsort(score)
    Vs = V[o]
    e = np.linspace(0, len(Vs), n + 1).astype(int)
    return float(Vs[e[-2] : e[-1]].mean()), float(Vs[e[0] : e[1]].mean())


def evaluate_two_head_critics(configuration: ConfigMap) -> int:
    """Compare deployed TD and cosine readouts of saved critic checkpoints.

    Uses shared trajectory triples for progress ordering and a fixed seed-1
    negative-goal permutation for cosine retrieval. Writes per-critic metrics
    to ``two_head_analysis_<task>.csv`` without updating any checkpoint.

    Args:
        configuration: Task/split, checkpoint/output paths, critics, discount,
            query count, maximum goal distance, and sampling seed.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = configuration
    task = cfg.task
    evals = Path(cfg.checkpoint_root)
    env, ds = load_crl_env_and_dataset(env_name=task, discount=cfg.discount, split=cfg.split)
    S, A, G, V = sample_trajectory_queries(
        ds, cfg.discount, cfg.n_queries, cfg.max_goal_distance, cfg.seed
    )
    gneg = G[np.random.default_rng(1).permutation(len(G))]

    rows = []
    for crit in cfg.critics:
        d = evals / f"{crit}_{task}"
        if not (d / "params.msgpack").exists():
            raise FileNotFoundError(d / "params.msgpack")
        agent, _ = load_critic(d, critic_rebuild_function(crit))
        qd = critic_score(agent, crit, S, G, A)
        kt_q = float(kendalltau(qd, V).statistic)
        top_q, bot_q = _binned(qd, V)
        row = {
            "task": task,
            "critic": crit,
            "qtd_kendall": kt_q,
            "qtd_top_bin": top_q,
            "qtd_bot_bin": bot_q,
        }
        if crit.startswith("two_head_") or crit == "cosine":
            cos = _cosine_score(agent, S, G, A)
            row["cos_kendall"] = float(kendalltau(cos, V).statistic)
            cos_pos = _cosine_score(agent, S, G, A)
            cos_neg = _cosine_score(agent, S, gneg, A)
            row["retrieval_auc"] = float(_auc(cos_pos, cos_neg))
        rows.append(row)
        print(
            f"[two_head_analysis] {crit:12s} qtd_kendall={kt_q:+.3f} "
            f"top/bot={top_q:.3f}/{bot_q:.3f} "
            f"cos_kendall={row.get('cos_kendall', float('nan')):+.3f} "
            f"retrieval_auc={row.get('retrieval_auc', float('nan')):.3f}",
            flush=True,
        )

    out = Path(cfg.output_dir) / f"two_head_analysis_{task}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "task",
        "critic",
        "qtd_kendall",
        "qtd_top_bin",
        "qtd_bot_bin",
        "cos_kendall",
        "retrieval_auc",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[two_head_analysis] wrote {out} ({len(rows)} rows)")
    return 0
