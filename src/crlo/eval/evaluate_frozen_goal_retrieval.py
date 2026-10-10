"""Frozen executed-value frozen training-split retrieval gate (plan §7.3).

Reproduces the canonical ``run_retrieval_eval.py`` construction —
discount 0.99, 3,000 same-trajectory future-positive triples, maximum
temporal offset 60, RNG seed 0, and ``retrieval_auc`` against the
seed-0 permutation of those goals — but freezes the triple indices and
the negative permutation in a manifest so seeds 42-44 are scored on the
identical instrument (never rebuilt after an outcome exists).

Modes:

- ``build_manifest`` — sample the triples once via the canonical
  sampler, record the flat dataset indices ``t``, offsets ``k``, the
  negative permutation, dataset identity hashes, and a checksum of the
  positive-goal matrix. Run before the first smoke rollout so the
  manifest joins the analysis freeze.
- ``score`` — load the hash-matched manifest, verify the dataset
  reproduces the recorded triples, score one explicit checkpoint, and
  append a ``retrieval_<task>.csv`` row: ``task, critic, training_seed,
  label, retrieval_auc, n_triples, manifest_sha256``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from scipy.stats import mannwhitneyu  # noqa: E402

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.envs.ogbench_loader import load_crl_env_and_dataset  # noqa: E402
from crlo.eval.config_schemas import FrozenGoalRetrievalConfig
from crlo.eval.query_manifests import sha256_bytes, sha256_file  # noqa: E402
from crlo.eval.trajectory_queries import sample_trajectory_queries_with_metadata
from crlo.training.checkpoints import load_critic  # noqa: E402


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()


def _dataset_triples(cfg: FrozenGoalRetrievalConfig):
    """Canonical training-split triples with per-triple metadata."""
    _, ds = load_crl_env_and_dataset(env_name=cfg.task, discount=cfg.discount)
    return ds, sample_trajectory_queries_with_metadata(
        ds, cfg.discount, cfg.n_triples, cfg.kmax, cfg.seed
    )


def build_manifest(cfg: FrozenGoalRetrievalConfig) -> Path:
    """Freeze the retrieval instrument (triples + negative permutation)."""
    _, (S, A, G, _V, T, K, _END, _EP) = _dataset_triples(cfg)
    # Canonical negative permutation: the FIRST draw from
    # default_rng(seed), exactly as run_retrieval_eval consumes it.
    neg = np.random.default_rng(cfg.seed).permutation(len(G))
    payload_arrays = {
        "t_index": np.asarray(T, np.int64),
        "k_offset": np.asarray(K, np.int64),
        "neg_permutation": np.asarray(neg, np.int64),
    }
    checks = {
        "task": cfg.task,
        "discount": cfg.discount,
        "n_triples": cfg.n_triples,
        "kmax": cfg.kmax,
        "seed": cfg.seed,
        "s_sha16": sha256_bytes(np.ascontiguousarray(S).tobytes())[:16],
        "g_sha16": sha256_bytes(np.ascontiguousarray(G).tobytes())[:16],
        "a_sha16": sha256_bytes(np.ascontiguousarray(A).tobytes())[:16],
        "git": _git(),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.manifest_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload_arrays)
    tmp.replace(cfg.manifest_path)
    checks["manifest_sha256"] = sha256_file(cfg.manifest_path)
    side = cfg.manifest_path.with_suffix(".json")
    side.write_text(json.dumps(checks, indent=2))
    print(
        f"[frozen_retrieval] manifest frozen: {cfg.manifest_path} "
        f"sha256={checks['manifest_sha256'][:16]}…",
        flush=True,
    )
    return cfg.manifest_path


def score(cfg: FrozenGoalRetrievalConfig) -> Path:
    """Score one checkpoint on the frozen retrieval manifest."""
    side = json.loads(cfg.manifest_path.with_suffix(".json").read_text())
    if sha256_file(cfg.manifest_path) != side["manifest_sha256"]:
        raise ValueError("retrieval manifest hash mismatch")
    for key in ("task", "discount", "n_triples", "kmax", "seed"):
        if side[key] != getattr(cfg, key):
            raise ValueError(f"manifest {key}={side[key]!r} != config {getattr(cfg, key)!r}")
    z = np.load(cfg.manifest_path)
    T = z["t_index"]
    K = z["k_offset"]
    neg = z["neg_permutation"]

    ds, (S, A, G, _V, T2, K2, _END, _EP) = _dataset_triples(cfg)
    if not (np.array_equal(T, T2) and np.array_equal(K, K2)):
        raise ValueError(
            "dataset no longer reproduces the frozen triple indices — "
            "identity mismatch, refusing to score"
        )
    for name, arr in (("s_sha16", S), ("g_sha16", G), ("a_sha16", A)):
        got = sha256_bytes(np.ascontiguousarray(arr).tobytes())[:16]
        if got != side[name]:
            raise ValueError(f"retrieval manifest {name} mismatch: {got}")

    from crlo.eval.evaluate_progress_ordering import _score  # noqa: PLC0415

    agent, meta = load_critic(cfg.checkpoint_dir, critic_rebuild_function(cfg.critic))
    ckpt_seed = int(meta.get("seed", -1))
    if ckpt_seed != cfg.training_seed:
        raise ValueError(
            f"checkpoint metadata seed {ckpt_seed} != declared training_seed {cfg.training_seed}"
        )
    sc_pos = _score(agent, cfg.critic, S, G, A)
    sc_neg = _score(agent, cfg.critic, S, G[neg], A)
    u = mannwhitneyu(sc_pos, sc_neg, alternative="greater").statistic
    auc = float(u / (len(sc_pos) * len(sc_neg)))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / f"retrieval_{cfg.task.split('-')[0]}.csv"
    row = {
        "task": cfg.task,
        "critic": cfg.critic,
        "training_seed": cfg.training_seed,
        "label": cfg.label,
        "retrieval_auc": auc,
        "n_triples": len(S),
        "manifest_sha256": side["manifest_sha256"],
        "checkpoint_params_sha256": sha256_file(cfg.checkpoint_dir / "params.msgpack"),
        "git": _git(),
    }
    exists = out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[frozen_retrieval] {cfg.label}: AUC={auc:.4f} -> {out}", flush=True)
    return out


def evaluate_frozen_goal_retrieval(configuration: ConfigMap) -> int:
    """Build a frozen retrieval manifest or score one checkpoint against it.

    In ``build_manifest`` mode, writes triple identities and a negative-goal
    permutation. In ``score`` mode, verifies the manifest and dataset identity
    before appending a checkpoint's retrieval metrics to the output CSV.

    Args:
        configuration: Fields accepted by ``FrozenGoalRetrievalConfig``, including
            mode, dataset identity, manifest path, and checkpoint for scoring.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg_raw = configuration
    cfg = FrozenGoalRetrievalConfig.from_mapping(cfg_raw)
    if cfg.mode == "build_manifest":
        build_manifest(cfg)
    else:
        score(cfg)
    return 0
