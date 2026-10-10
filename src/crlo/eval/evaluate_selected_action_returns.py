"""Selected-action rollout audit: simulator ground-truth value for selected candidates.

Campaign-3 S16.3 Case-A remedy (also plan S9.2): for a preregistered subset
of the frozen candidate-pool queries, measure the *realized* outcome of executing a
candidate action and then following a fixed, critic-independent continuation
policy (the goal-conditioned DDPM behavior model). Rollout return is
unbiased and critic-independent, so it validates or replaces the learned
GCIQL reference where the reference ensemble cannot resolve candidate
values.

Design (paired, deterministic):
- Preregistered queries: the first ``n_per_stratum`` queries of each stratum
  in frozen pool-index order, pool replicate 0.
- Audited actions per query: each listed critic's K=64 argmax, the
  reference-ensemble-mean argmax (tag ``oracle_ref``), one fixed random
  candidate (tag ``random``), and the logged dataset action (tag
  ``logged``). Duplicate candidate indices are collapsed (tags joined).
- Exact state restoration via ``env.unwrapped.set_state(qpos, qvel)``
  (obs = [qpos, qvel] on locomaze; verified bit-exact).
- Common random numbers: the continuation policy's RNG is reseeded per
  (query, rollout) identically across candidates, so per-query differences
  are paired.
- Success: xy distance to the query goal <= env ``_goal_tol``; the realized
  value is ``gamma ** steps_to_success`` (0 if the horizon is exhausted),
  matching the gamma^d return-to-go convention.

Writes ``rollout_audit_{task}_shard{i}.csv`` (one row per query x action x
rollout) into ``output_dir``. Shard over queries with ``n_shards`` /
``shard`` for parallel CPU execution.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import jax
import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.training.checkpoints import load_diffusion  # noqa: E402

STRATUM_NAMES = ("near", "medium", "far")


def _select_actions(
    pools_dir: Path,
    critic_labels: list[str],
    ref_labels: list[str],
    q: int,
    rep: int,
    rand_seed: int,
) -> dict[str, int | None]:
    """Audited candidate indices for one query: tag -> pool index."""
    tags: dict[str, int | None] = {}
    for lab in critic_labels:
        sc = np.load(pools_dir / f"scores_{lab}.npz")["scores"][q, rep]
        tags[lab] = int(np.argmax(sc))
    refs = np.stack([np.load(pools_dir / f"scores_{r}.npz")["scores"][q, rep] for r in ref_labels])
    tags["oracle_ref"] = int(np.argmax(refs.mean(0)))
    k = refs.shape[-1]
    tags["random"] = int(np.random.default_rng(rand_seed + q).integers(0, k))
    tags["logged"] = None  # logged anchor, not in the pool
    return tags


def evaluate_selected_action_returns(configuration: ConfigMap) -> int:
    """Execute selected pool actions and anchors from restored query states.

    Selects the declared per-stratum prefix of queries and partitions it by
    shard. Runs critic/reference selections, a random candidate, and the logged
    action with the saved diffusion continuation; writes rollout audit CSVs.

    Args:
        configuration: Pool and diffusion paths, critic/reference labels, query
            and rollout counts, horizon, discount, replicate, shard, and audit seed.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = configuration
    task = cfg.task
    pools_dir = Path(cfg.get("pools_dir"))
    diffusion_ckpt = Path(cfg.get("diffusion_ckpt"))
    critic_labels = list(cfg.get("critics") or [])
    ref_labels = list(cfg.get("refs") or [])
    n_per_stratum = int(cfg["n_per_stratum"])
    n_rollouts = int(cfg["n_rollouts"])
    horizon = int(cfg["horizon"])
    gamma = float(cfg["discount"])
    audit_seed = int(cfg["audit_seed"])
    rep = int(cfg["replicate"])
    n_shards = int(cfg["n_shards"])
    shard = int(cfg["shard"])
    output_dir = Path(cfg.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(pools_dir / "pools.npz")
    S, G, A, AL, strat = z["S"], z["G"], z["A"], z["A_logged"], z["stratum"]
    t_index = z["t_index"]

    # Preregistered subset: first n_per_stratum frozen indices per stratum.
    chosen: list[int] = []
    for s in range(3):
        chosen.extend(np.nonzero(strat == s)[0][:n_per_stratum].tolist())
    chosen = chosen[shard::n_shards]

    from crlo.envs.ogbench_loader import load_raw_env_and_datasets

    env, _, _ = load_raw_env_and_datasets(task, cfg.dataset_dir)
    env.reset()
    u = env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    obs_dim = S.shape[-1]
    # State restoration source. locomaze obs is [qpos, qvel] (ant), qpos alone
    # (point, which exposes no velocity in the observation; restore qvel as
    # zero), or a derived proprio vector that does NOT embed the sim state
    # (humanoid: root z/quaternion absent from the obs). In the derived case,
    # restore exactly from the raw dataset's stored qpos/qvel via t_index.
    if obs_dim < nq + nv:
        assert obs_dim == nq, (obs_dim, nq, nv)
        state_from_obs = True
    else:
        probe = S[chosen[0]].astype(np.float64)
        u.set_state(probe[:nq], probe[nq : nq + nv])
        state_from_obs = bool(np.allclose(np.asarray(u.get_ob()), S[chosen[0]], atol=1e-5))
    qpos_ds = qvel_ds = None
    if not state_from_obs:
        raw_path = Path(cfg.dataset_dir).expanduser() / f"{task}-val.npz"
        raw = np.load(raw_path)
        assert raw["observations"].shape[0] > int(t_index.max()), raw_path
        qpos_ds = raw["qpos"].astype(np.float64)
        qvel_ds = raw["qvel"].astype(np.float64)
        # t_index must address the same flat val split the pools sampled from
        assert np.allclose(raw["observations"][t_index[chosen[0]]], S[chosen[0]], atol=1e-6), (
            "t_index misaligned"
        )
    print(
        f"[selected_action_returns] state restoration: "
        f"{'obs-embedded' if state_from_obs else 'dataset qpos/qvel'}",
        flush=True,
    )
    goal_tol = float(u._goal_tol)
    policy = load_diffusion(diffusion_ckpt, seed=audit_seed)

    rows = []
    t0 = time.time()
    for qi, q in enumerate(chosen):
        tags = _select_actions(pools_dir, critic_labels, ref_labels, q, rep, audit_seed)
        # collapse duplicate pool indices so identical actions roll out once
        by_idx: dict[int | None, list[str]] = {}
        for tag, idx in tags.items():
            by_idx.setdefault(idx, []).append(tag)
        goal_xy = G[q][:2].astype(np.float64)
        for idx, tag_list in by_idx.items():
            a0 = (AL[q] if idx is None else A[q, rep, idx]).astype(np.float64)
            for r in range(n_rollouts):
                # CRN: same continuation-policy stream for every candidate
                policy.rng = jax.random.PRNGKey(audit_seed * 1000003 + int(q) * 101 + r)
                if state_from_obs:
                    o = S[q].astype(np.float64)
                    qvel = o[nq : nq + nv] if obs_dim >= nq + nv else np.zeros(nv)
                    u.set_state(o[:nq], qvel)
                else:
                    ti = int(t_index[q])
                    u.set_state(qpos_ds[ti], qvel_ds[ti])
                ob, *_ = env.step(a0)
                ob = np.asarray(ob)
                steps, success = 1, float(np.linalg.norm(ob[:2] - goal_xy) <= goal_tol)
                while not success and steps < horizon:
                    a = np.asarray(policy.sample(ob.astype("float32"), G[q].astype("float32"), 1))[
                        0
                    ]
                    ob, *_ = env.step(np.clip(a, -1.0, 1.0))
                    ob = np.asarray(ob)
                    steps += 1
                    success = float(np.linalg.norm(ob[:2] - goal_xy) <= goal_tol)
                rows.append(
                    {
                        "query": int(q),
                        "stratum": STRATUM_NAMES[int(strat[q])],
                        "tags": "|".join(tag_list),
                        "pool_index": -1 if idx is None else int(idx),
                        "rollout": r,
                        "success": int(success),
                        "steps_to_goal": steps if success else -1,
                        "disc_return": round(gamma**steps if success else 0.0, 6),
                    }
                )
        if (qi + 1) % 5 == 0:
            print(
                f"[selected_action_returns] shard {shard}: {qi + 1}/{len(chosen)} "
                f"queries ({time.time() - t0:.0f}s)",
                flush=True,
            )

    out = output_dir / f"rollout_audit_{task}_shard{shard}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        f"[selected_action_returns] wrote {out} ({len(rows)} rows, {time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0
