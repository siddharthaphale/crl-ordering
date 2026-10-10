"""Critic-blind acceptance check for the frozen continuation policy.

Runs the pure closed-loop DDPM continuation (no candidate injection)
once per medium-stratum query at frozen stratum ranks 460-489 — a range
disjoint from the primary/pilot/smoke manifests and the legacy
exclusions — with horizon 300 and the amendment's dedicated CRN
namespace ``fold_in(fold_in(fold_in(PRNGKey(cfg.rollout_seed), q), 9), 0)``.

Acceptance floor: goal attainment on at least 15/30 queries. Writes
``continuation_acceptance.json`` next to the checkpoint. Reads pools and
state data only; never touches a score file.

Usage::

    python experiments/check_continuation_policy.py \
        <pools_dir> <diffusion_ckpt> <output_json> [deterministic]

With the optional ``deterministic`` argument the Arm-B mode-path
continuation is used (plan §17.2): every acceptance rollout is executed
twice and the pair must be bit-identical (exact-determinism check) in
addition to the goal floor.
"""

from __future__ import annotations

import json
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
)
from crlo.eval.query_manifests import sha256_file, stratum_ranks  # noqa: E402
from crlo.training.checkpoints import load_diffusion  # noqa: E402


def check_continuation_policy(configuration: ConfigMap) -> int:
    """Check continuation success on the declared disjoint query range."""
    cfg = configuration
    pools_dir, ckpt_dir, out_json = map(Path, (cfg.pools_dir, cfg.checkpoint_dir, cfg.output_json))
    deterministic = cfg.continuation_mode == "deterministic"
    if cfg.continuation_mode not in ("stochastic", "deterministic"):
        raise ValueError("Unknown continuation mode")
    z = np.load(pools_dir / "pools.npz")
    S, G, strat = z["S"], z["G"], z["stratum"]
    ranks = stratum_ranks(strat)
    qids = [
        int(q)
        for q in np.flatnonzero(
            (strat == cfg.stratum) & (ranks >= cfg.rank_start) & (ranks <= cfg.rank_end)
        )
    ]
    assert len(qids) == cfg.rank_end - cfg.rank_start + 1, qids

    from crlo.envs.ogbench_loader import load_raw_env_and_datasets

    task = cfg.task
    env, _, _ = load_raw_env_and_datasets(task, cfg.dataset_dir)
    env.reset(seed=cfg.rollout_seed)
    u = env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    act_lo = np.asarray(env.action_space.low, np.float64)
    act_hi = np.asarray(env.action_space.high, np.float64)
    goal_tol = float(u._goal_tol)  # noqa: SLF001
    base = load_diffusion(ckpt_dir, seed=cfg.rollout_seed)
    policy = (
        DeterministicDiffusionContinuation(base)
        if deterministic
        else StochasticDiffusionContinuation(base)
    )

    def one_rollout(q: int) -> tuple[int, int]:
        s0 = S[q].astype(np.float64)
        policy.rng = cell_key(cfg.rollout_seed, q, cfg.namespace_half, cfg.namespace_rollout)
        restore_state(env, s0[:nq], s0[nq : nq + nv], S[q], cfg.restore_atol, cfg.restore_rtol)
        goal_xy = G[q][:2].astype(np.float64)
        ob = s0
        success, steps = 0, 0
        while not success and steps < cfg.horizon:
            a = np.asarray(policy.sample(ob.astype("float32"), G[q].astype("float32"), 1))[0]
            ob, *_ = env.step(np.clip(a, act_lo, act_hi))
            ob = np.asarray(ob)
            steps += 1
            success = int(np.linalg.norm(ob[:2] - goal_xy) <= goal_tol)
        return success, (steps if success else -1)

    t0 = time.time()
    successes, steps_list = [], []
    for q in qids:
        success, steps = one_rollout(q)
        if deterministic:
            dup = one_rollout(q)
            if dup != (success, steps):
                raise RuntimeError(
                    f"exact-determinism check FAILED at q={q}: "
                    f"{(success, steps)} vs {dup} (plan §17.2)"
                )
        successes.append(success)
        steps_list.append(steps)
        print(f"[continuation_acceptance] q={q} success={success} steps={steps}", flush=True)

    n_success = int(sum(successes))
    result = {
        "continuation_mode": ("deterministic" if deterministic else "stochastic"),
        "determinism_check": ("all duplicates bit-identical" if deterministic else None),
        "task": task,
        "checkpoint": str(ckpt_dir),
        "checkpoint_params_sha256": sha256_file(ckpt_dir / "params.msgpack"),
        "pool_sha16": sha256_file(pools_dir / "pools.npz")[:16],
        "rank_range": [cfg.rank_start, cfg.rank_end],
        "stratum": cfg.stratum,
        "horizon": cfg.horizon,
        "namespace": [cfg.namespace_half, cfg.namespace_rollout],
        "query_ids": qids,
        "successes": successes,
        "steps_to_goal": steps_list,
        "n_success": n_success,
        "floor": cfg.minimum_successes,
        "pass": n_success >= cfg.minimum_successes,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(
        f"[continuation_acceptance] {n_success}/30 (floor {cfg.minimum_successes}) -> "
        f"{'PASS' if result['pass'] else 'FAIL'} ({out_json})",
        flush=True,
    )
    return 0 if result["pass"] else 1
