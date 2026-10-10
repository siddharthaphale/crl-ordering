"""Candidate-pool scorer: evaluate one checkpoint's deployed scalar on cached candidate pools.

Scores every candidate (and the logged anchor) in ``pools.npz`` with one
model's deployed readout and caches the result, so metrics are computed
later from identical, paired tensors (the paired-pool protocol). Works for
both deployed critics and GCIQL reference models (``critic=td_q``).

Writes ``output_dir/scores_{label}.npz`` with::

    scores        (Q, R, K)   deployed-scalar score per candidate
    scores_logged (Q,)        score of the tagged logged-action anchor
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.eval.critic_scoring import critic_score  # noqa: E402
from crlo.training.checkpoints import load_critic  # noqa: E402


def _score_chunked(agent, critic: str, S, G, A, chunk: int) -> np.ndarray:
    """Score flat (N, dim) triples in chunks; returns (N,)."""
    out = np.empty(len(S), np.float64)
    for i in range(0, len(S), chunk):
        sl = slice(i, i + chunk)
        out[sl] = critic_score(agent, critic, S[sl], G[sl], A[sl])
    return out


def score_candidate_pools(configuration: ConfigMap) -> int:
    """Score every shared candidate or the explicit manifest population."""
    cfg = configuration
    if cfg.population == "manifest":
        from crlo.eval.score_manifest_candidates import score_manifest

        return score_manifest(cfg)
    if cfg.population != "all_pools":
        raise ValueError("population must be all_pools or manifest")
    pools_dir = Path(cfg.get("pools_dir"))
    checkpoint_dir = Path(cfg.get("checkpoint_dir"))
    critic = str(cfg.get("critic"))
    rebuild_critic = str(cfg["rebuild_critic"] or "") or critic
    label = str(cfg.get("label"))
    chunk = int(cfg["chunk"])
    out_override = cfg.get("output_dir", None)
    output_dir = Path(out_override) if out_override else pools_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(pools_dir / "pools.npz")
    S, G, A, AL = z["S"], z["G"], z["A"], z["A_logged"]
    Q, R, K, adim = A.shape

    agent, meta = load_critic(checkpoint_dir, critic_rebuild_function(rebuild_critic))
    t0 = time.time()
    S_flat = np.repeat(S, R * K, axis=0)
    G_flat = np.repeat(G, R * K, axis=0)
    A_flat = A.reshape(Q * R * K, adim)
    scores = _score_chunked(agent, critic, S_flat, G_flat, A_flat, chunk).reshape(Q, R, K)
    scores_logged = _score_chunked(agent, critic, S, G, AL, chunk)
    out = output_dir / f"scores_{label}.npz"
    np.savez_compressed(out, scores=scores, scores_logged=scores_logged)
    info = {
        "label": label,
        "critic": critic,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_seed": int(meta.get("seed", -1)),
        "pools": str(pools_dir / "pools.npz"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (output_dir / f"scores_{label}.json").write_text(json.dumps(info, indent=2))
    print(
        f"[score_candidate_pools] {label}: scored {Q}x{R}x{K} in {info['elapsed_s']}s -> {out}",
        flush=True,
    )
    return 0
