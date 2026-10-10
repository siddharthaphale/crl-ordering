"""Candidate-pool builder: frozen fixed-(s,g) queries + cached candidate pools.

Builds the shared evaluation asset for the fixed-query protocol
(the fixed-query protocol): a frozen, stratified query set from the
task's *validation* split and, per query, ``n_replicates`` independent
candidate pools of ``k_max`` actions drawn from the goal-conditioned DDPM
behavior proposal, clipped to the env action bounds. The logged dataset
action at each query is attached as a tagged anchor (never selectable).

Pools are generated and cached BEFORE any critic or reference is scored
(the paired-pool protocol): every later analysis is evaluation-only against
this file. Writes ``output_dir/pools.npz`` and ``pools_meta.json``.

Query construction: queries are sampled from complete val-split trajectories
with future-state goals at temporal distance d, stratified near/medium/far
(d in 1-10 / 11-30 / 31-60), ``n_per_stratum`` each, d uniform within the
stratum. The val split is used so queries are held out from critic training.

npz contents::

    S           (Q, obs_dim)                query states
    G           (Q, obs_dim)                query goals (future states)
    A           (Q, R, K, action_dim)       candidate pools (nested K prefixes)
    A_logged    (Q, action_dim)             logged dataset action anchor
    d           (Q,)                        temporal goal distance
    stratum     (Q,)                        0=near, 1=medium, 2=far
    t_index     (Q,)                        flat val-dataset index of the state
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.training.checkpoints import load_diffusion  # noqa: E402

STRATA = ((1, 10), (11, 30), (31, 60))  # near / medium / far goal distance


def _episode_bounds(terminals: np.ndarray) -> np.ndarray:
    """Return sorted flat indices of episode-final steps."""
    return np.nonzero(np.asarray(terminals) > 0)[0]


def _sample_queries(
    obs: np.ndarray,
    act: np.ndarray,
    term: np.ndarray,
    n_per_stratum: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Stratified fixed-(s,g) queries from complete trajectories."""
    rng = np.random.default_rng(seed)
    S, G, AL, D, STR, TI = [], [], [], [], [], []
    for si, (dlo, dhi) in enumerate(STRATA):
        got = 0
        while got < n_per_stratum:
            t = int(rng.integers(0, len(obs) - dlo - 1))
            idx = np.searchsorted(term, t)
            if idx >= len(term):
                continue
            end = int(term[idx])
            if end - t < dlo:
                continue
            d = int(rng.integers(dlo, min(dhi, end - t) + 1))
            S.append(obs[t])
            G.append(obs[t + d])
            AL.append(act[t])
            D.append(d)
            STR.append(si)
            TI.append(t)
            got += 1
    return (
        np.asarray(S, np.float32),
        np.asarray(G, np.float32),
        np.asarray(AL, np.float32),
        np.asarray(D, np.int32),
        np.asarray(STR, np.int8),
        np.asarray(TI, np.int64),
    )


def build_candidate_pools(configuration: ConfigMap) -> int:
    """Freeze validation queries and shared diffusion candidate pools."""
    cfg = configuration
    task = cfg.task
    n_per_stratum = int(cfg["n_per_stratum"])
    n_replicates = int(cfg["n_replicates"])
    k_max = int(cfg["k_max"])
    seed = int(cfg["seed"])
    diffusion_ckpt = Path(cfg.get("diffusion_ckpt"))
    output_dir = Path(cfg.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from crlo.envs.ogbench_loader import load_raw_env_and_datasets

    env, _train_raw, val_raw = load_raw_env_and_datasets(task, cfg.dataset_dir)
    obs = np.asarray(val_raw["observations"])
    act = np.asarray(val_raw["actions"])
    term = _episode_bounds(val_raw["terminals"])
    lo = np.asarray(env.action_space.low, np.float32)
    hi = np.asarray(env.action_space.high, np.float32)
    print(
        f"[candidate_pools] {task}: val split {len(obs)} steps, "
        f"{len(term)} episodes, action bounds [{lo.min()}, {hi.max()}]",
        flush=True,
    )

    S, G, AL, D, STR, TI = _sample_queries(obs, act, term, n_per_stratum, seed)
    Q = len(S)

    policy = load_diffusion(diffusion_ckpt, seed=seed + 1)
    t0 = time.time()
    A = np.empty((Q, n_replicates, k_max, act.shape[-1]), np.float32)
    for i in range(Q):
        draws = policy.sample(S[i].astype("float32"), G[i].astype("float32"), n_replicates * k_max)
        A[i] = np.clip(draws, lo, hi).reshape(n_replicates, k_max, -1)
        if (i + 1) % 200 == 0:
            print(
                f"[candidate_pools]   sampled {i + 1}/{Q} queries ({time.time() - t0:.0f}s)",
                flush=True,
            )

    out = output_dir / "pools.npz"
    np.savez_compressed(out, S=S, G=G, A=A, A_logged=AL, d=D, stratum=STR, t_index=TI)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()
    meta = {
        "task": task,
        "n_queries": Q,
        "n_per_stratum": n_per_stratum,
        "n_replicates": n_replicates,
        "k_max": k_max,
        "seed": seed,
        "diffusion_ckpt": str(diffusion_ckpt),
        "query_split": "val",
        "strata": [list(s) for s in STRATA],
        "clip_lo": float(lo.min()),
        "clip_hi": float(hi.max()),
        "sha256_16": digest,
        "git": git,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "pools_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[candidate_pools] wrote {out} sha256[:16]={digest} "
        f"({Q} queries x {n_replicates} replicates x {k_max})",
        flush=True,
    )
    return 0
