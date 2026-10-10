"""Evaluate goal compatibility and graded temporal-progress ordering."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, mannwhitneyu

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.envs.ogbench_loader import load_crl_env_and_dataset  # noqa: E402
from crlo.eval.trajectory_queries import sample_trajectory_queries_with_metadata, score_triples
from crlo.training.checkpoints import load_critic  # noqa: E402


def _hard_negative_goals(ds, S, T, K, END, EP, kmax, rng):
    """Build nearby-other-episode and different-offset same-episode goals.

    Returns (G_nearby, G_sametraj, mask_sametraj); ``mask_sametraj`` marks
    anchors with at least two valid temporal distances.
    """
    obs = np.asarray(ds.dataset["observations"])
    # nearby: nearest other-episode anchor's goal, z-scored state space
    Z = (S - S.mean(0)) / (S.std(0) + 1e-8)
    n = len(S)
    nearby_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        d2 = np.sum((Z - Z[i]) ** 2, axis=1)
        d2[EP[i] == EP] = np.inf  # different episode only
        nearby_idx[i] = int(np.argmin(d2))
    G_nearby = obs[T[nearby_idx] + K[nearby_idx]].astype(np.float32)
    # same trajectory, different temporal distance
    max_k = np.minimum(kmax, END - T)
    mask = max_k >= 2
    k_neg = np.empty(n, dtype=np.int64)
    for i in range(n):
        if not mask[i]:
            k_neg[i] = K[i]
            continue
        kp = int(rng.integers(1, max_k[i]))  # uniform over max_k-1 values
        k_neg[i] = kp if kp < K[i] else kp + 1  # skip k itself
    G_sametraj = obs[T + k_neg].astype(np.float32)
    return G_nearby, G_sametraj, mask


def _paired_win_rate(sc_pos, sc_neg):
    return float(np.mean((sc_pos > sc_neg) + 0.5 * (sc_pos == sc_neg)))


def _recall_at_k(agent, critic, S, A, G, rng, n_query, n_pool, ks):
    """Rank g^+ against a pool of random goals; return recall@k for each k."""
    n = len(S)
    q = min(n_query, n)
    qidx = rng.choice(n, size=q, replace=False)
    ranks = np.empty(q, dtype=np.int64)
    for j, i in enumerate(qidx):
        pool_idx = rng.choice(n, size=n_pool - 1, replace=False)
        goals = np.concatenate([G[i][None], G[pool_idx]], axis=0)  # pos first
        s_rep = np.repeat(S[i][None], n_pool, axis=0)
        a_rep = np.repeat(A[i][None], n_pool, axis=0)
        sc = score_triples(agent, critic, s_rep, goals, a_rep)
        # rank of the positive (index 0): #candidates scoring strictly higher
        ranks[j] = int((sc > sc[0]).sum())
    return {k: float((ranks < k).mean()) for k in ks}


def evaluate_goal_retrieval(configuration: ConfigMap) -> int:
    """Measure goal retrieval and temporal-progress ordering on shared triples.

    Loads saved critics and reuses seeded triples, hard negatives, and retrieval
    pools across critics. Writes ``retrieval_<task>.csv`` and
    ``retrieval_curve_<task>.csv`` under ``output_dir``. Retrieval ranks count
    strictly higher scores, so tied goals do not outrank the positive goal.

    Args:
        configuration: Task, critics, discount, triple/query/pool counts, maximum
            temporal offset, seed, and checkpoint/output paths from the command.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = configuration
    task = cfg.task
    critics = list(cfg["critics"])
    gamma = float(cfg["discount"])
    n = int(cfg["n_triples"])
    kmax = int(cfg["kmax"])
    seed = int(cfg["seed"])
    n_query = int(cfg["n_query"])
    n_pool = int(cfg["n_pool"])
    evals_root = Path(cfg["evals_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, ds = load_crl_env_and_dataset(env_name=task, discount=gamma)
    S, A, G, V, T, K, END, EP = sample_trajectory_queries_with_metadata(ds, gamma, n, kmax, seed)
    hard_rng = np.random.default_rng(seed + 1)
    G_nearby, G_sametraj, mask_st = _hard_negative_goals(ds, S, T, K, END, EP, kmax, hard_rng)
    k_dist = np.rint(np.log(V) / np.log(gamma)).astype(int)  # temporal dist
    print(
        f"[retrieval] {task}: {len(S)} triples, k in "
        f"[{k_dist.min()},{k_dist.max()}]; sametraj-eligible "
        f"{int(mask_st.sum())}",
        flush=True,
    )

    rows, curve_rows = [], []
    for critic in critics:
        d = evals_root / f"{critic}_{task}"
        if not (d / "params.msgpack").exists():
            raise FileNotFoundError(d / "params.msgpack")
        agent, _ = load_critic(d, critic_rebuild_function(critic))
        rng = np.random.default_rng(seed)

        sc_pos = score_triples(agent, critic, S, G, A)
        neg = rng.permutation(len(G))
        sc_neg = score_triples(agent, critic, S, G[neg], A)
        u = mannwhitneyu(sc_pos, sc_neg, alternative="greater").statistic
        auc = float(u / (len(sc_pos) * len(sc_neg)))
        sc_nearby = score_triples(agent, critic, S, G_nearby, A)
        auc_nearby = _paired_win_rate(sc_pos, sc_nearby)
        sc_st = score_triples(agent, critic, S, G_sametraj, A)
        auc_sametraj = _paired_win_rate(sc_pos[mask_st], sc_st[mask_st])
        kt = float(kendalltau(sc_pos, V).statistic)
        rec = _recall_at_k(agent, critic, S, A, G, rng, n_query, n_pool, (1, 5))
        print(
            f"[retrieval]   {critic:12s} AUC={auc:.3f}  "
            f"AUCnear={auc_nearby:.3f}  AUCst={auc_sametraj:.3f}  "
            f"R@1={rec[1]:.3f}  R@5={rec[5]:.3f}  gamma_tau={kt:+.3f}",
            flush=True,
        )
        rows.append(
            {
                "task": task,
                "critic": critic,
                "retrieval_auc": auc,
                "auc_nearby": auc_nearby,
                "auc_sametraj": auc_sametraj,
                "n_sametraj": int(mask_st.sum()),
                "recall_at_1": rec[1],
                "recall_at_5": rec[5],
                "gamma_tau": kt,
                "neg_score_mean": float(sc_neg.mean()),
                "pos_score_mean": float(sc_pos.mean()),
                "n_triples": len(S),
                "n_pool": n_pool,
            }
        )
        for k in range(1, kmax + 1):
            m = k_dist == k
            if m.sum() >= 5:
                curve_rows.append(
                    {
                        "task": task,
                        "critic": critic,
                        "k": k,
                        "mean_score": float(sc_pos[m].mean()),
                        "n": int(m.sum()),
                    }
                )

    out = output_dir / f"retrieval_{task}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cout = output_dir / f"retrieval_curve_{task}.csv"
    with cout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)
    print(f"[retrieval] wrote {out} + {cout}", flush=True)
    return 0
