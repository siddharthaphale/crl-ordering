"""Generate score-decile calibration inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from crlo.agents.critic_factory import critic_rebuild_function
from crlo.envs.ogbench_loader import load_crl_env_and_dataset
from crlo.eval.critic_scoring import critic_score
from crlo.eval.trajectory_queries import sample_trajectory_queries
from crlo.training.checkpoints import load_critic
from crlo.utils.config_loader import ConfigMap


def _score_all(agent, critic, S, G, A, batch=512):
    out = []
    for i in range(0, len(S), batch):
        out.append(
            critic_score(agent, critic, S[i : i + batch], G[i : i + batch], A[i : i + batch])
        )
    return np.concatenate(out).reshape(-1)


def _binned(scores, V, n_bins=10):
    """Mean gamma^d per score-decile; x = bin-center percentile (5..95)."""
    order = np.argsort(scores)
    Vs = V[order]
    edges = np.linspace(0, len(Vs), n_bins + 1).astype(int)
    xs, ys = [], []
    for b in range(n_bins):
        seg = Vs[edges[b] : edges[b + 1]]
        if len(seg) == 0:
            continue
        xs.append((b + 0.5) / n_bins * 100.0)
        ys.append(float(seg.mean()))
    return np.array(xs), np.array(ys)


def evaluate_score_calibration(configuration: ConfigMap) -> int:
    """Write mean temporal progress by deployed-score percentile bin.

    Scores the same seeded dataset triples with each saved critic and bins
    ``discount**temporal_offset`` by score rank. Writes ``calibration.csv``;
    these are progress diagnostics, not calibrated likelihoods or returns.

    Args:
        configuration: Dataset/split, checkpoint/output paths, critics, query
            sampling settings, inference batch size, bin count, and seed.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = configuration
    env, ds = load_crl_env_and_dataset(
        cfg.task, cfg.discount, split=cfg.split, dataset_dir=cfg.dataset_dir
    )
    env.close()
    states, actions, goals, values = sample_trajectory_queries(
        ds, cfg.discount, cfg.n_queries, cfg.max_goal_distance, cfg.seed
    )
    rows = []
    for critic in cfg.critics:
        agent, _ = load_critic(
            Path(cfg.checkpoint_root) / f"{critic}_{cfg.task}", critic_rebuild_function(critic)
        )
        scores = _score_all(agent, critic, states, goals, actions, cfg.batch_size)
        percentiles, means = _binned(scores, values, cfg.n_bins)
        rows.extend(
            {"task": cfg.task, "critic": critic, "score_pctile": x, "mean_gamma_d": y}
            for x, y in zip(percentiles, means, strict=True)
        )
    output = Path(cfg.output_dir) / "calibration.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["task", "critic", "score_pctile", "mean_gamma_d"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0
