"""Dataset gamma^d ranking quality on an OGBench task.

The OGBench MC-Q* regret signal is structurally degenerate (sparse reward +
single-action washout), so for OGBench we measure a *ranking* signal instead:
do the critic scores order in-distribution ``(s, a, g)`` triples by their
ground-truth goal-conditioned value?

For a goal-conditioned task with discount ``gamma`` and sparse reward, the
offline value of reaching ``g = obs[t+k]`` from ``(obs[t], act[t])`` along a
trajectory is ``gamma^k`` (the discounted hitting value). We sample N such
triples (k = temporal distance within a trajectory), score each with the
critic, and report Kendall tau / Spearman rho between the critic score and
``gamma^k``. Scores use the shared :func:`crlo.eval.critic_scoring.critic_score`, so cosine /
hybrid are evaluated on L2-normalized embeddings (matching how they train and
how the fixed forward now behaves).

Writes ``gamma_ranking_<task>.csv``: task, critic, kendall_tau, spearman_rho,
n_triples, gamma_min, gamma_max.

Usage::

    python experiments/evaluate_progress_ordering.py \
        --task antmaze-large-navigate-v0 \
        --critics raw_bilinear cosine td_q hybrid \
        --output-dir results/progress_ordering
"""

from __future__ import annotations

import csv
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.envs.ogbench_loader import load_crl_env_and_dataset  # noqa: E402
from crlo.eval.trajectory_queries import sample_trajectory_queries, score_triples
from crlo.training.checkpoints import load_critic  # noqa: E402


def evaluate_progress_ordering(configuration: ConfigMap) -> int:
    """Compare critic scores with discounted same-trajectory progress.

    Loads saved critics and scores shared seeded triples. Writes Kendall and
    Spearman correlations to ``gamma_ranking_<task>.csv`` under ``output_dir``.
    The target is ``discount**temporal_offset``, not executed action value.

    Args:
        configuration: Task, critics, discount, triple count, maximum temporal
            offset, seed, and checkpoint/output paths from the command.

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
    evals_root = Path(cfg["evals_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env, ds = load_crl_env_and_dataset(env_name=task, discount=gamma)
    S, A, G, V = sample_trajectory_queries(ds, gamma, n, kmax, seed)
    print(
        f"[gamma_ranking] {task}: {len(S)} triples, gamma^d in [{V.min():.3f}, {V.max():.3f}]",
        flush=True,
    )

    rows = []
    for critic in critics:
        d = evals_root / f"{critic}_{task}"
        if not (d / "params.msgpack").exists():
            raise FileNotFoundError(d / "params.msgpack")
        agent, _ = load_critic(d, critic_rebuild_function(critic))
        sc = score_triples(agent, critic, S, G, A)
        kt = float(kendalltau(sc, V).statistic)
        sr = float(spearmanr(sc, V).statistic)
        print(f"[gamma_ranking]   {critic:12s} kendall={kt:+.3f}  spearman={sr:+.3f}", flush=True)
        rows.append(
            {
                "task": task,
                "critic": critic,
                "kendall_tau": kt,
                "spearman_rho": sr,
                "n_triples": len(S),
                "gamma_min": float(V.min()),
                "gamma_max": float(V.max()),
            }
        )

    out = output_dir / f"gamma_ranking_{task}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[gamma_ranking] wrote {out} ({len(rows)} critics)", flush=True)
    return 0
