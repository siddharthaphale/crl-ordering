"""Training/readout decomposition recovered from Git commit 0d8f106."""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.stats import kendalltau

from crlo.agents.critic_factory import critic_rebuild_function
from crlo.envs.ogbench_loader import load_crl_env_and_dataset
from crlo.eval.trajectory_queries import sample_trajectory_queries
from crlo.training.checkpoints import load_critic
from crlo.utils.config_loader import ConfigMap


def normalize_embeddings(x):
    """Apply the historical readout normalization along the last axis.

    Args:
        x: Embeddings with features on the final axis.

    Returns:
        Embeddings divided by ``norm(x) + 1e-8`` with the same shape. This
        epsilon placement is retained for parity with frozen readout results.
    """
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _readout_scores(agent, S, G, A, batch=512):
    """{readout: score (N,)} for one encoder, ensemble-averaged."""
    raw, cos, nrm = [], [], []

    def _red(x):
        x = np.asarray(x)
        return (x.mean(0) if x.ndim == 2 else x).reshape(-1)

    for i in range(0, len(S), batch):
        sj = jnp.asarray(S[i : i + batch])
        gj = jnp.asarray(G[i : i + batch])
        aj = jnp.asarray(A[i : i + batch])
        # info=True returns the RAW (unnormalized) phi/psi for every critic; the
        # scalar v is the deployed value (cosine score for the cosine module), so
        # the raw readout is the inner product of phi/psi directly, not v.
        _v, phi, psi = agent.network.select("critic")(sj, gj, actions=aj, info=True)
        raw.append(_red(jnp.sum(phi * psi, axis=-1)))
        cos.append(_red(jnp.sum(normalize_embeddings(phi) * normalize_embeddings(psi), axis=-1)))
        nrm.append(_red(jnp.linalg.norm(phi, axis=-1)))
    return {
        "raw": np.concatenate(raw),
        "cosine": np.concatenate(cos),
        "phi_norm": np.concatenate(nrm),
    }


def evaluate_readout_ordering(configuration: ConfigMap) -> int:
    """Compare raw, cosine, and norm readouts on shared trajectory triples.

    Reuses each loaded encoder across the requested readouts. Preserves the
    historical raw inner product and epsilon-normalized cosine formulas.
    Writes Kendall correlations with temporal progress to ``readout.csv``.

    Args:
        configuration: Dataset/split, checkpoint/output paths, readout names,
            query sampling settings, inference batch size, and seed.

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
    cache, rows = {}, []
    readouts = {
        "raw_raw": {"name": "raw_raw", "encoder": "raw_bilinear", "readout": "raw"},
        "raw_cosine": {"name": "raw_cosine", "encoder": "raw_bilinear", "readout": "cosine"},
        "raw_phinorm": {"name": "raw_phinorm", "encoder": "raw_bilinear", "readout": "phi_norm"},
        "cosine_raw": {"name": "cosine_raw", "encoder": "cosine", "readout": "raw"},
        "cosine_cosine": {"name": "cosine_cosine", "encoder": "cosine", "readout": "cosine"},
    }
    for name in cfg.readouts:
        row = readouts[name]
        encoder = row["encoder"]
        if encoder not in cache:
            agent, _ = load_critic(
                Path(cfg.checkpoint_root) / f"{encoder}_{cfg.task}",
                critic_rebuild_function(encoder),
            )
            cache[encoder] = _readout_scores(agent, states, goals, actions, cfg.batch_size)
        rows.append(
            {
                "task": cfg.task,
                "row": row["name"],
                "kendall_tau": float(kendalltau(cache[encoder][row["readout"]], values).statistic),
            }
        )
    output = Path(cfg.output_dir) / "readout.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "row", "kendall_tau"])
        writer.writeheader()
        writer.writerows(rows)
    return 0
