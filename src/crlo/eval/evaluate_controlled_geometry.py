"""Evaluate saved controlled critics on an action grid and candidate actions.

Writes score grids, gradient norms, and scored candidates consumed by
the companion paper repository's controlled-geometry figure builder. This evaluator never trains.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function
from crlo.eval.config_schemas import ControlledGeometryConfig
from crlo.training.checkpoints import load_critic

CRITICS = ("raw_bilinear", "cosine", "td_q", "hybrid")


def _score_on_grid(
    agent,
    critic: str,
    A0: np.ndarray,
    A1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained critic over the fixed-(s, g) action grid."""
    grid_size = A0.shape[0]
    actions = np.stack([A0.ravel(), A1.ravel()], axis=-1).astype(np.float32)
    n = actions.shape[0]
    s = np.zeros((n, 2), dtype=np.float32)
    g = np.zeros((n, 2), dtype=np.float32)
    if critic == "td_q":
        q1, q2 = agent.network.select("critic")(
            jnp.asarray(s),
            jnp.asarray(g),
            jnp.asarray(actions),
        )
        z = ((np.asarray(q1) + np.asarray(q2)) / 2.0).reshape(
            grid_size,
            grid_size,
        )
    else:
        v, _, _ = agent.network.select("critic")(
            jnp.asarray(s),
            jnp.asarray(g),
            actions=jnp.asarray(actions),
            info=True,
        )
        v_np = np.asarray(v)
        if v_np.ndim == 2:
            v_np = v_np.mean(axis=0)
        z = v_np.reshape(grid_size, grid_size)
    a0_axis, a1_axis = A0[0, :], A1[:, 0]
    gy, gx = np.gradient(z, a1_axis, a0_axis)
    return z, np.sqrt(gx**2 + gy**2)


def evaluate_controlled_geometry(configuration: ConfigMap) -> int:
    """Write action-grid scores and candidate selections for saved critics.

    Uses the geometry seed protocol to generate a shared clipped Gaussian pool.
    Writes per-critic score/gradient NPY grids and candidate CSVs for the paper
    figure builder, without training. Candidate scores index the sampled grid.

    Args:
        configuration: Fields accepted by ``ControlledGeometryConfig``, including
            checkpoint/output directories, grid size, and proposal settings.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = ControlledGeometryConfig.from_mapping(
        configuration,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    a_axis = np.linspace(-1.0, 1.0, cfg.grid_size)
    A0, A1 = np.meshgrid(a_axis, a_axis)

    rng = np.random.default_rng(cfg.seed)
    proposal_std = float(cfg.proposal_std)
    proposal_clip = float(cfg.proposal_clip)
    candidates = np.clip(
        rng.normal(scale=proposal_std, size=(cfg.n_candidates, 2)),
        -proposal_clip,
        proposal_clip,
    )

    for critic in CRITICS:
        agent, _ = load_critic(cfg.checkpoint_dir / critic, critic_rebuild_function(critic))

        score_grid, grad_grid = _score_on_grid(agent, critic, A0, A1)
        np.save(
            cfg.output_dir / f"{critic}_grid.npy",
            score_grid.astype(np.float32),
        )
        np.save(
            cfg.output_dir / f"{critic}_gradnorm.npy",
            grad_grid.astype(np.float32),
        )

        ix = np.clip(
            ((candidates[:, 0] + 1.0) / 2.0 * (cfg.grid_size - 1)).astype(int),
            0,
            cfg.grid_size - 1,
        )
        iy = np.clip(
            ((candidates[:, 1] + 1.0) / 2.0 * (cfg.grid_size - 1)).astype(int),
            0,
            cfg.grid_size - 1,
        )
        cand_scores = score_grid[iy, ix]
        selected = int(np.argmax(cand_scores))
        out_csv = cfg.output_dir / f"{critic}_candidates.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["candidate_id", "a0", "a1", "critic_score", "selected_flag"],
            )
            for i, (x, y) in enumerate(candidates):
                writer.writerow(
                    [
                        i,
                        float(x),
                        float(y),
                        float(cand_scores[i]),
                        int(i == selected),
                    ]
                )
    print(f"[toy_critic_surfaces] outputs written to {cfg.output_dir}")
    return 0
