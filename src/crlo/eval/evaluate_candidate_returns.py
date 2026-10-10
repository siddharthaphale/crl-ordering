"""Full-candidate-pool rollout driver: every candidate, executed (plan §5).

Score-blind production driver: reads only the cached pools, the frozen
query manifest, state-restoration data, and the locked continuation
checkpoint. It refuses score paths at the config layer and never opens
one.

For every manifest query it executes all K candidates x 2 halves x
``rollouts_per_half`` cells: exact state restoration (verified, fails
closed), one bound-clipped candidate action, then the frozen sequential
DDPM continuation until locomaze goal attainment or the horizon.
Common random numbers follow the frozen cell-key mapping
``fold_in(fold_in(fold_in(PRNGKey(rollout_seed), query_id), half_id),
rollout_id)`` — candidate index is deliberately absent (plan §5.4).

Shards are query-disjoint (``manifest row index mod n_shards``), written
atomically, and resumable: a completed shard is immutable and is never
re-run. Shard NPZ arrays: ``query_id, episode_id, stratum, stratum_rank,
pool_replicate, candidate, half, rollout, success, steps_to_goal,
disc_return, completed`` (plan §10.3); the JSON sidecar records every
frozen seed and provenance hash.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.eval.candidate_returns import (  # noqa: E402
    DeterministicDiffusionContinuation,
    StochasticDiffusionContinuation,
    cell_key,
    restore_state,
    run_cell,
)
from crlo.eval.config_schemas import CandidateReturnsConfig
from crlo.eval.query_manifests import load_manifest, sha256_file  # noqa: E402
from crlo.eval.rollout_shards import shard_complete, write_shard  # noqa: E402
from crlo.training.checkpoints import load_diffusion  # noqa: E402


def _detect_restore_mode(env, s0: np.ndarray, atol: float, rtol: float) -> str:
    """Frozen restore-mode probe (mirrors the archived audit's logic)."""
    u = env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    obs_dim = len(s0)
    if obs_dim < nq + nv:
        if obs_dim != nq:
            return "dataset"
        return "obs_qpos_only"
    probe = np.asarray(s0, np.float64)
    u.set_state(probe[:nq], probe[nq : nq + nv])
    ok = np.allclose(np.asarray(u.get_ob()), s0, atol=atol, rtol=rtol)
    return "obs_embedded" if ok else "dataset"


def evaluate_candidate_returns(configuration: ConfigMap) -> int:
    """Execute every frozen candidate with a locked continuation policy.

    Reads pools, a query manifest, restoration data, and a diffusion checkpoint
    without consuming critic scores. Completed shards are reused. Each new shard
    records returns and provenance using candidate-independent random keys.

    Args:
        configuration: Arguments accepted by ``CandidateReturnsConfig``, including
            locked input hashes, rollout settings, and shard assignment.

    Returns:
        Zero after writing or reusing the assigned shard. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg_raw = configuration
    cfg = CandidateReturnsConfig.from_mapping(cfg_raw)
    pools_npz = cfg.pools_dir / "pools.npz"
    pool_sha16 = sha256_file(pools_npz)[:16]
    if pool_sha16 != cfg.expected_pool_sha16:
        raise ValueError(f"pools sha {pool_sha16} != {cfg.expected_pool_sha16}")
    ckpt_sha = sha256_file(cfg.diffusion_ckpt / "params.msgpack")
    if ckpt_sha != cfg.continuation_sha256:
        raise ValueError(
            f"continuation params sha256 {ckpt_sha} != locked {cfg.continuation_sha256}"
        )

    manifest = load_manifest(cfg.manifest_path)
    if manifest["meta"]["pool_sha16"] != pool_sha16:
        raise ValueError("manifest was built against a different pool")
    rows = manifest["rows"][cfg.shard :: cfg.n_shards]
    identity = {
        "task": cfg.task,
        "manifest_rows_sha256": manifest["rows_sha256"],
        "pool_sha16": pool_sha16,
        "continuation_sha256": ckpt_sha,
        "rollout_seed": cfg.rollout_seed,
        "continuation_mode": cfg.continuation_mode,
        "action_persist": cfg.action_persist,
        "horizon": cfg.horizon,
        "discount": cfg.discount,
        "rollouts_per_half": cfg.rollouts_per_half,
        "restore_atol": cfg.restore_atol,
        "restore_rtol": cfg.restore_rtol,
        "n_shards": cfg.n_shards,
        "shard": cfg.shard,
    }
    if shard_complete(cfg.output_dir, cfg.shard, identity):
        print(f"[candidate_returns] verified completed shard {cfg.shard}; skipping", flush=True)
        return 0
    if not rows:
        raise ValueError("Shard contains no queries; reduce n_shards")
    print(
        f"[candidate_returns] shard {cfg.shard}/{cfg.n_shards}: {len(rows)} "
        f"queries x {manifest['meta']['k']} candidates x 2 halves x "
        f"{cfg.rollouts_per_half} rollouts",
        flush=True,
    )

    z = np.load(pools_npz)
    S, G, A = z["S"], z["G"], z["A"]
    k = A.shape[2]

    from crlo.envs.ogbench_loader import load_raw_env_and_datasets

    env, _, _ = load_raw_env_and_datasets(cfg.task, cfg.dataset_dir)
    env.reset(seed=cfg.rollout_seed)  # fixed reset RNG, candidate-blind
    u = env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    act_lo = np.asarray(env.action_space.low, np.float64)
    act_hi = np.asarray(env.action_space.high, np.float64)
    goal_tol = float(u._goal_tol)  # noqa: SLF001
    mode = _detect_restore_mode(env, S[rows[0]["query_id"]], cfg.restore_atol, cfg.restore_rtol)
    qpos_ds = qvel_ds = t_index = None
    if mode == "dataset":
        if cfg.val_data_path is None:
            raise ValueError(
                "dataset restore mode requires val_data_path (raw "
                "qpos/qvel); mandatory for HumanoidMaze (plan §5.5)"
            )
        raw = np.load(cfg.val_data_path)
        qpos_ds = raw["qpos"].astype(np.float64)
        qvel_ds = raw["qvel"].astype(np.float64)
        t_index = z["t_index"]
        q0 = rows[0]["query_id"]
        if not np.allclose(raw["observations"][t_index[q0]], S[q0], atol=1e-6):
            raise ValueError("t_index misaligned with raw val split")
    print(
        f"[candidate_returns] task={cfg.task} restore_mode={mode} goal_tol={goal_tol}", flush=True
    )

    base_policy = load_diffusion(cfg.diffusion_ckpt, seed=cfg.rollout_seed)
    policy = (
        DeterministicDiffusionContinuation(base_policy)
        if cfg.continuation_mode == "deterministic"
        else StochasticDiffusionContinuation(base_policy)
    )

    out: dict[str, list] = {
        f: []
        for f in (
            "query_id",
            "episode_id",
            "stratum",
            "stratum_rank",
            "pool_replicate",
            "candidate",
            "half",
            "rollout",
            "success",
            "steps_to_goal",
            "disc_return",
            "completed",
        )
    }
    t0 = time.time()
    for i, row in enumerate(rows):
        q, rep = row["query_id"], row["pool_replicate"]
        goal = G[q]
        goal_xy = goal[:2].astype(np.float64)
        s0 = S[q].astype(np.float64)
        if mode == "dataset":
            qpos, qvel = qpos_ds[int(t_index[q])], qvel_ds[int(t_index[q])]
        elif mode == "obs_qpos_only":
            qpos, qvel = s0[:nq], np.zeros(nv)
        else:
            qpos, qvel = s0[:nq], s0[nq : nq + nv]
        for half in (0, 1):
            for r in range(cfg.rollouts_per_half):
                for c in range(k):
                    policy.rng = cell_key(cfg.rollout_seed, q, half, r)
                    restore_state(env, qpos, qvel, S[q], cfg.restore_atol, cfg.restore_rtol)
                    success, steps, y = run_cell(
                        env,
                        policy,
                        s0,
                        goal,
                        goal_xy,
                        goal_tol,
                        A[q, rep, c],
                        act_lo,
                        act_hi,
                        cfg.horizon,
                        cfg.discount,
                        persist=cfg.action_persist,
                    )
                    out["query_id"].append(q)
                    out["episode_id"].append(row["episode_id"])
                    out["stratum"].append(row["stratum"])
                    out["stratum_rank"].append(row["stratum_rank"])
                    out["pool_replicate"].append(rep)
                    out["candidate"].append(c)
                    out["half"].append(half)
                    out["rollout"].append(r)
                    out["success"].append(success)
                    out["steps_to_goal"].append(steps)
                    out["disc_return"].append(y)
                    out["completed"].append(1)
        done = (i + 1) * 2 * cfg.rollouts_per_half * k
        rate = done / max(time.time() - t0, 1e-9)
        print(
            f"[candidate_returns] shard {cfg.shard}: query {i + 1}/{len(rows)} "
            f"({rate:.2f} cells/s, {time.time() - t0:.0f}s)",
            flush=True,
        )

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()
    sidecar = {
        "task": cfg.task,
        "manifest": manifest["meta"]["manifest"],
        "manifest_rows_sha256": manifest["rows_sha256"],
        "pool_sha16": pool_sha16,
        "val_data_sha16": manifest["meta"]["val_data_sha16"],
        "exclusion_manifest_sha256": manifest["meta"]["exclusion_manifest_sha256"],
        "continuation_sha256": ckpt_sha,
        "rollout_seed": cfg.rollout_seed,
        "tie_seed": manifest["meta"]["tie_seed"],
        "continuation_mode": cfg.continuation_mode,
        "action_persist": cfg.action_persist,
        "horizon": cfg.horizon,
        "discount": cfg.discount,
        "rollouts_per_half": cfg.rollouts_per_half,
        "restore_mode": mode,
        "goal_tol": goal_tol,
        "restore_atol": cfg.restore_atol,
        "restore_rtol": cfg.restore_rtol,
        "n_shards": cfg.n_shards,
        "shard": cfg.shard,
        "git": git,
        "elapsed_s": round(time.time() - t0, 1),
    }
    cells = {f: np.asarray(v) for f, v in out.items()}
    write_shard(cfg.output_dir, cfg.shard, cells, sidecar)
    print(
        f"[candidate_returns] shard {cfg.shard} complete: "
        f"{len(cells['query_id'])} cells in "
        f"{sidecar['elapsed_s']}s",
        flush=True,
    )
    return 0
