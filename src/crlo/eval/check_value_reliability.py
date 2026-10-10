"""Score-blind executed-value reliability and shard merging.

Reads raw rollout shard artifacts (merging them if the merged tensor
does not exist yet) and the frozen query manifest — nothing else. It
cannot accept critic labels or score paths: the config schema refuses
them and this driver never opens a ``scores_*`` file.

Modes:

- ``pilot``  — the frozen three-part futility rule (plan §5.3): one-sided
  90% episode-cluster-bootstrap upper bounds on resolvable fraction,
  split-half agreement, and cross-fitted empirical-winner headroom, each
  against its confirmatory target. Writes ``futility_<task>.json``.
- ``primary`` — Value reliability (plan §6.3): resolvable fraction >= 0.5,
  equal-stratum mean split-half Spearman >= 0.70, mean cross-fitted
  headroom >= h_min, and a positive one-sided 95% lower bound on that
  headroom. Writes ``reliability_<task>.json`` + the resolvable-query mask.

- ``deterministic_primary`` — the Arm-B deterministic-judge Value-reliability analog (plan
  §17.2): the two halves are duplicate executions of a deterministic
  continuation and must agree exactly (fails closed otherwise); the
  gate requires >= 50% of queries with non-constant candidate values
  and an equal-stratum mean cross-candidate value range >= h_min.

Both stochastic modes write ``query_reliability.csv`` (per-query
spread, paired noise, Spearman, Kendall tau-b diagnostic, flat flags,
headroom) and ``bootstrap_episode_indices.npz`` (the frozen resample
matrix; the primary matrix is reused verbatim by Gates B and C).
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

from crlo.eval.config_schemas import ValueReliabilityConfig
from crlo.eval.query_manifests import load_manifest, sha256_file  # noqa: E402
from crlo.eval.rollout_shards import merge_shards, write_merged  # noqa: E402
from crlo.eval.value_statistics import (  # noqa: E402
    assess_value_reliability,
    bootstrap_equal_stratum,
    crossfit_winner,
    draw_episode_resamples,
    equal_stratum_mean,
    lower_bound,
    pilot_futility,
    query_reliability,
    upper_bound,
)


def _ensure_merged(cfg: ValueReliabilityConfig, manifest: dict) -> Path:
    """Merge shard artifacts into the dense tensor if not already done."""
    if cfg.merged_path.exists():
        return cfg.merged_path
    shard_dir = cfg.merged_path.parent.parent / "rollouts"
    tensors = merge_shards(
        shard_dir,
        manifest["rows"],
        cfg.k,
        cfg.rollouts_per_half,
        expected_sidecar={
            "manifest_rows_sha256": manifest["rows_sha256"],
            "pool_sha16": manifest["meta"]["pool_sha16"],
        },
    )
    meta = {
        "manifest_rows_sha256": manifest["rows_sha256"],
        "pool_sha16": manifest["meta"]["pool_sha16"],
        "n_queries": len(manifest["rows"]),
        "k": cfg.k,
        "rollouts_per_half": cfg.rollouts_per_half,
    }
    return write_merged(cfg.merged_path.parent, tensors, meta)


def _deterministic_primary_gate(
    cfg, manifest, merged, ret, rows, strata, episodes, task_prefix
) -> int:
    """Arm-B Value-reliability analog (plan §17.2): exact duplicates + resolution."""
    v_a, v_b = ret[:, :, 0, 0], ret[:, :, 1, 0]
    if not np.array_equal(v_a, v_b):
        n_bad = int((v_a != v_b).sum())
        raise RuntimeError(
            f"deterministic duplicate verification FAILED: {n_bad} cells "
            f"differ between the two executions — implementation blocker "
            f"(plan §17.2); failing closed"
        )
    v = v_a
    nonconstant = np.asarray([np.ptp(v[i]) > 0.0 for i in range(len(rows))])
    value_range = np.asarray([float(np.ptp(v[i])) for i in range(len(rows))])
    headroom_exact = np.asarray([float(v[i].max() - v[i].mean()) for i in range(len(rows))])
    frac = equal_stratum_mean(nonconstant.astype(np.float64), strata)
    mean_range = equal_stratum_mean(value_range, strata)
    checks = {
        "duplicate_agreement": True,
        "nonconstant_fraction": bool(np.isfinite(frac) and frac >= cfg.frac_target),
        "value_range": bool(np.isfinite(mean_range) and mean_range >= cfg.h_min),
    }
    resamples = draw_episode_resamples(episodes, strata, cfg.n_boot, [cfg.bootstrap_seed, 1])
    boot_path = cfg.output_dir / "bootstrap_episode_indices.npz"
    with boot_path.open("wb") as f:
        np.savez_compressed(
            f,
            resamples=resamples,
            episode_ids_sorted=np.unique(episodes),
            seed_words=np.asarray([cfg.bootstrap_seed, 1]),
        )
    with (cfg.output_dir / "query_reliability.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "stratum",
                "episode_id",
                "nonconstant",
                "value_range",
                "headroom_exact",
            ],
        )
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow(
                {
                    "query_id": row["query_id"],
                    "stratum": row["stratum_name"],
                    "episode_id": row["episode_id"],
                    "nonconstant": int(nonconstant[i]),
                    "value_range": value_range[i],
                    "headroom_exact": headroom_exact[i],
                }
            )
    decision = {
        "gate": "GATE_A_DET",
        "pass": all(checks.values()),
        "checks": checks,
        "nonconstant_fraction": float(frac),
        "n_nonconstant": int(nonconstant.sum()),
        "n_queries": len(rows),
        "mean_value_range": float(mean_range),
        "mean_headroom_exact": float(equal_stratum_mean(headroom_exact, strata)),
        "targets": {"nonconstant_fraction": cfg.frac_target, "value_range_h_min": cfg.h_min},
        "resolvable_mask": nonconstant.astype(int).tolist(),
        "task": manifest["meta"]["task"],
        "manifest": manifest["meta"]["manifest"],
        "manifest_rows_sha256": manifest["rows_sha256"],
        "merged_sha16": sha256_file(merged)[:16],
        "bootstrap_seed_words": [cfg.bootstrap_seed, 1],
        "n_boot": cfg.n_boot,
        "continuation_mode": "deterministic",
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
    }
    out = cfg.output_dir / f"reliability_{task_prefix}.json"
    out.write_text(json.dumps(decision, indent=2))
    print(f"[value_reliability] GATE A (det) pass={decision['pass']} -> {out}", flush=True)
    return 0


def check_value_reliability(configuration: ConfigMap) -> int:
    """Assess return reliability without reading critic scores."""
    cfg_raw = configuration
    cfg = ValueReliabilityConfig.from_mapping(cfg_raw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg.manifest_path)
    task_prefix = manifest["meta"]["task"].split("-")[0]
    merged = _ensure_merged(cfg, manifest)
    merged_meta = json.loads(merged.with_suffix(".json").read_text())
    if merged_meta["manifest_rows_sha256"] != manifest["rows_sha256"]:
        raise ValueError("merged tensor was built from a different manifest")
    ret = np.load(merged)["disc_return"]  # (n, K, 2, R)

    rows = manifest["rows"]
    strata = np.asarray([r["stratum"] for r in rows])
    episodes = np.asarray([r["episode_id"] for r in rows])
    n = len(rows)

    if cfg.mode == "deterministic_primary":
        return _deterministic_primary_gate(
            cfg, manifest, merged, ret, rows, strata, episodes, task_prefix
        )

    t0 = time.time()
    rel_rows, resolvable, rho, headroom = [], [], [], []
    for i, row in enumerate(rows):
        y_a, y_b = ret[i, :, 0, :], ret[i, :, 1, :]
        rel = query_reliability(y_a, y_b)
        prio = np.asarray(row["tie_priority"])
        cf = crossfit_winner(y_a.mean(-1), y_b.mean(-1), prio)
        resolvable.append(rel.resolvable)
        rho.append(rel.rho)
        headroom.append(cf.headroom)
        rel_rows.append(
            {
                "query_id": row["query_id"],
                "stratum": row["stratum_name"],
                "episode_id": row["episode_id"],
                "spread_a": rel.spread_a,
                "spread_b": rel.spread_b,
                "noise_a": rel.noise_a,
                "noise_b": rel.noise_b,
                "rho": rel.rho,
                "tau_b": rel.tau_b,
                "flat_value": int(rel.flat_value),
                "resolvable": int(rel.resolvable),
                "winner_idx_a": cf.idx_a,
                "winner_idx_b": cf.idx_b,
                "w_cf": cf.w_cf,
                "pool_mean": cf.pool_mean,
                "headroom_cf": cf.headroom,
            }
        )
    resolvable = np.asarray(resolvable, bool)
    rho = np.asarray(rho, np.float64)
    headroom = np.asarray(headroom, np.float64)

    csv_path = cfg.output_dir / "query_reliability.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rel_rows[0].keys()))
        w.writeheader()
        w.writerows(rel_rows)

    seed_word = 0 if cfg.mode == "pilot" else 1
    resamples = draw_episode_resamples(
        episodes, strata, cfg.n_boot, [cfg.bootstrap_seed, seed_word]
    )
    boot_path = cfg.output_dir / "bootstrap_episode_indices.npz"
    with boot_path.open("wb") as f:
        np.savez_compressed(
            f,
            resamples=resamples,
            episode_ids_sorted=np.unique(episodes),
            seed_words=np.asarray([cfg.bootstrap_seed, seed_word]),
        )

    frac_point = equal_stratum_mean(resolvable.astype(np.float64), strata)
    agree_point = equal_stratum_mean(rho, strata, valid=resolvable)
    head_point = equal_stratum_mean(headroom, strata)
    boot_frac = bootstrap_equal_stratum(resolvable.astype(np.float64), strata, episodes, resamples)
    boot_agree = bootstrap_equal_stratum(rho, strata, episodes, resamples, valid=resolvable)
    boot_head = bootstrap_equal_stratum(headroom, strata, episodes, resamples)

    common = {
        "task": manifest["meta"]["task"],
        "manifest": manifest["meta"]["manifest"],
        "manifest_rows_sha256": manifest["rows_sha256"],
        "merged_sha16": sha256_file(merged)[:16],
        "bootstrap_seed_words": [cfg.bootstrap_seed, seed_word],
        "n_boot": cfg.n_boot,
        "n_queries": n,
        "point": {
            "resolvable_fraction": frac_point,
            "agreement": agree_point,
            "headroom": head_point,
        },
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
        "elapsed_s": round(time.time() - t0, 1),
    }

    if cfg.mode == "pilot":
        decision = pilot_futility(
            resolvable_ub=upper_bound(boot_frac, cfg.pilot_ub_q),
            agreement_ub=upper_bound(boot_agree, cfg.pilot_ub_q),
            headroom_ub=upper_bound(boot_head, cfg.pilot_ub_q),
            n_resolvable=int(resolvable.sum()),
            n_finite_rho=int(np.isfinite(rho).sum()),
            frac_target=cfg.frac_target,
            agreement_target=cfg.agreement_target,
            h_min=cfg.h_min,
        )
        out = cfg.output_dir / f"futility_{task_prefix}.json"
        out.write_text(json.dumps(common | decision, indent=2))
        print(f"[value_reliability] PILOT decision: {decision['decision']} -> {out}", flush=True)
        return 0

    head_lb = lower_bound(boot_head, cfg.gate_lb_q)
    decision = assess_value_reliability(
        resolvable_fraction=frac_point,
        n_resolvable=int(resolvable.sum()),
        n_queries=n,
        agreement=agree_point,
        headroom=head_point,
        headroom_lb=head_lb,
        frac_target=cfg.frac_target,
        agreement_target=cfg.agreement_target,
        h_min=cfg.h_min,
    )
    decision["resolvable_mask"] = resolvable.astype(int).tolist()
    out = cfg.output_dir / f"reliability_{task_prefix}.json"
    out.write_text(json.dumps(common | decision, indent=2))
    print(f"[value_reliability] GATE A pass={decision['pass']} -> {out}", flush=True)
    return 0
