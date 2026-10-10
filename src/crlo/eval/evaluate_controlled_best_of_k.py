"""Toy candidate-count sweep with closed-form ground truth.

This is the controlled regret result.
The 2D toy has dynamics s' = s + a and reward -||s'-g||, so
Q*(s, a, g) = -||s + a - g|| is exact and dense for *arbitrary* actions --
the regime OGBench cannot provide (sparse reward + single-action washout).

For each critic and each K, we draw K candidate actions from the proposal,
select argmax under the critic, and record the closed-form true return and
the selected embedding norm ||phi(s,a)|| (the failure channel), averaged over
many query/candidate draws. Writes controlled_best_of_k.csv:
    critic, K, mean_true_return, mean_regret, mean_selected_phi_norm,
    mean_selected_score.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import jax.numpy as jnp  # noqa: E402

from crlo.agents.critic_factory import critic_rebuild_function
from crlo.eval.critic_scoring import critic_score  # noqa: E402
from crlo.training.checkpoints import load_critic

CRITICS = ("raw_bilinear", "cosine", "td_q", "hybrid")


def _phi_norm(agent, critic, s, g, a):
    """||phi(s,a)|| for CRL-family critics; NaN for td_q (no factorisation)."""
    if critic == "td_q":
        return np.nan
    s_b = jnp.asarray(s[None].astype(np.float32))
    g_b = jnp.asarray(g[None].astype(np.float32))
    a_b = jnp.asarray(a[None].astype(np.float32))
    _, phi, _ = agent.network.select(
        "critic",
    )(s_b, g_b, actions=a_b, info=True)
    return float(np.linalg.norm(np.asarray(phi).reshape(-1)))


def evaluate_controlled_best_of_k(configuration: ConfigMap) -> int:
    """Measure exact controlled regret over nested candidate counts.

    Loads saved critics and uses the same seeded proposal pools for each critic.
    Writes ``controlled_best_of_k.csv`` under ``output_dir``. Checkpoint metadata
    must declare zero transition noise for the closed-form target to be exact.

    Args:
        configuration: Checkpoint/output directories, candidate counts, query count,
            proposal scale and clipping, and seed from the matching command.
            Its ``output_dir`` field is converted to a ``Path`` in place.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    args = configuration
    args["output_dir"] = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a_dim = 2
    rows = []
    for critic in CRITICS:
        agent, metadata = load_critic(
            Path(args.checkpoint_dir) / critic, critic_rebuild_function(critic)
        )
        if metadata["environment"]["noise_std"] != 0:
            raise ValueError("Exact values require deterministic controlled training")
        # query at fixed (s,g)=(0,0) (matches Fig 2); Q*(a) = -||a||
        s = np.zeros(2, np.float32)
        g = np.zeros(2, np.float32)
        qrng = np.random.default_rng(args.seed + 7)
        maxK = max(args.candidate_counts)
        # pre-draw a pool per query, slice first-K
        pools = [
            np.clip(qrng.normal(scale=args.proposal_std, size=(maxK, a_dim)), -args.clip, args.clip)
            for _ in range(args.n_queries)
        ]
        # score the full candidate pool once per query, then slice first-K.
        # critic_score keeps each deployed readout and its evaluation scale explicit.
        scored = []
        for cand in pools:
            sb = np.broadcast_to(s, (maxK, 2)).astype(np.float32)
            gb = np.broadcast_to(g, (maxK, 2)).astype(np.float32)
            scored.append(critic_score(agent, critic, sb, gb, cand.astype(np.float32)))
        for K in args.candidate_counts:
            rets, regs, norms, selsc = [], [], [], []
            for cand, sc in zip(pools, scored, strict=True):
                idx = int(np.argmax(sc[:K]))
                a_sel = cand[idx]
                true_ret = -np.linalg.norm(s + a_sel - g)
                best = max(-np.linalg.norm(s + cand[j] - g) for j in range(K))
                rets.append(true_ret)
                regs.append(best - true_ret)
                selsc.append(float(sc[idx]))
                norms.append(_phi_norm(agent, critic, s, g, a_sel))
            rows.append(
                {
                    "critic": critic,
                    "K": K,
                    "mean_true_return": float(np.mean(rets)),
                    "mean_regret": float(np.mean(regs)),
                    "mean_selected_phi_norm": float(np.mean(norms))
                    if critic != "td_q"
                    else float("nan"),
                    "mean_selected_score": float(np.mean(selsc)),
                }
            )
        print(
            f"[toy_sweep] {critic}: "
            f"ret K1={rows[-len(args.candidate_counts)]['mean_true_return']:.3f} -> "
            f"K{max(args.candidate_counts)}={rows[-1]['mean_true_return']:.3f}",
            flush=True,
        )

    out = args.output_dir / "controlled_best_of_k.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[toy_sweep] wrote {out} ({len(rows)} rows)")
    return 0
