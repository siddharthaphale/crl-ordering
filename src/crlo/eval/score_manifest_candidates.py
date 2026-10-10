"""Frozen executed-value scorer: one explicit checkpoint on the frozen query--pool manifest.

Scores exactly the manifest's (query, pool replicate) candidate vectors
— no candidate is regenerated, removed, re-ordered, or filtered — and
writes a manifest-shaped ``(n, K)`` tensor plus a JSON sidecar with the
complete identity chain: checkpoint hashes, pool hash, manifest digest,
resolved config, scorer git commit (plan §10.1). Used for the seed-42
compatibility rescore and every seed-43/44 tensor (plan §8.2).

Writes ``scores_{label}.npz`` + ``scores_{label}.json`` in
``output_dir``. NPZ: ``scores (n, K)``, ``query_id (n,)``,
``pool_replicate (n,)``.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.eval.config_schemas import CandidateScoringConfig
from crlo.eval.critic_scoring import critic_score  # noqa: E402
from crlo.eval.query_manifests import load_manifest, sha256_file  # noqa: E402
from crlo.training.checkpoints import load_critic  # noqa: E402


def score_manifest(cfg_raw) -> int:
    """Score exactly the manifest rows and record their identity chain."""
    data = dict(cfg_raw)
    for key in ("population", "rebuild_critic"):
        data.pop(key, None)
    cfg = CandidateScoringConfig.from_mapping(data)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pools_npz = cfg.pools_dir / "pools.npz"
    pool_sha16 = sha256_file(pools_npz)[:16]
    if pool_sha16 != cfg.expected_pool_sha16:
        raise ValueError(f"pools sha {pool_sha16} != {cfg.expected_pool_sha16}")
    manifest = load_manifest(cfg.manifest_path)
    if manifest["meta"]["pool_sha16"] != pool_sha16:
        raise ValueError("manifest was built against a different pool")
    rows = manifest["rows"]

    z = np.load(pools_npz)
    S, G, A = z["S"], z["G"], z["A"]
    k = A.shape[2]
    qids = np.asarray([r["query_id"] for r in rows])
    reps = np.asarray([r["pool_replicate"] for r in rows])
    S_m = np.repeat(S[qids], k, axis=0)
    G_m = np.repeat(G[qids], k, axis=0)
    A_m = A[qids, reps].reshape(len(rows) * k, A.shape[-1])

    agent, meta = load_critic(cfg.checkpoint_dir, critic_rebuild_function(cfg.critic))
    ckpt_seed = int(meta.get("seed", -1))
    if ckpt_seed != cfg.training_seed:
        raise ValueError(
            f"checkpoint metadata seed {ckpt_seed} != declared training_seed {cfg.training_seed}"
        )
    scores = np.empty(len(S_m), np.float64)
    for i in range(0, len(S_m), cfg.chunk):
        sl = slice(i, i + cfg.chunk)
        scores[sl] = critic_score(agent, cfg.critic, S_m[sl], G_m[sl], A_m[sl])
    scores = scores.reshape(len(rows), k)
    if not np.isfinite(scores).all():
        raise ValueError(
            "non-finite scores on the frozen manifest — implementation blocker (plan §7.2)"
        )

    out = cfg.output_dir / f"scores_{cfg.label}.npz"
    tmp = out.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, scores=scores, query_id=qids, pool_replicate=reps)
    tmp.replace(out)
    sidecar = {
        "label": cfg.label,
        "critic": cfg.critic,
        "training_seed": cfg.training_seed,
        "task": cfg.task,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "checkpoint_params_sha256": sha256_file(cfg.checkpoint_dir / "params.msgpack"),
        "checkpoint_meta_sha256": sha256_file(cfg.checkpoint_dir / "meta.pkl"),
        "pool_sha16": pool_sha16,
        "manifest": manifest["meta"]["manifest"],
        "manifest_rows_sha256": manifest["rows_sha256"],
        "scores_sha16": sha256_file(out)[:16],
        "n_queries": len(rows),
        "k": k,
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (cfg.output_dir / f"scores_{cfg.label}.json").write_text(json.dumps(sidecar, indent=2))
    print(
        f"[manifest_scoring] {cfg.label}: {scores.shape} in {sidecar['elapsed_s']}s -> {out}",
        flush=True,
    )
    return 0
