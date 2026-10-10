"""Score-blind action-persistence calibration.

Consumes one deterministic-judge pilot merged tensor per persistence
scale ``m`` and applies the frozen m* rule: the smallest grid m with
nonconstant fraction >= frac_target, equal-stratum mean value range >=
h_min, and median distance-value Spearman >= smoothness_target. Halves
of every tensor must be bit-identical (duplicate verification fails
closed). Writes ``persistence_calibration_antmaze.json`` + per-(m,
query) CSV. Reads rollout artifacts, pools, and the manifest only.
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

from crlo.eval.config_schemas import ActionPersistenceConfig
from crlo.eval.query_manifests import load_manifest, sha256_file  # noqa: E402
from crlo.eval.value_statistics import (  # noqa: E402
    action_value_smoothness,
    equal_stratum_mean,
)


def action_repeat_statistics(
    v: np.ndarray, actions_by_row: list[np.ndarray], strata: np.ndarray
) -> dict:
    """Nonconstant fraction, mean range, and smoothness for one m."""
    n = v.shape[0]
    nonconstant = np.asarray([np.ptp(v[i]) > 0.0 for i in range(n)])
    value_range = np.asarray([float(np.ptp(v[i])) for i in range(n)])
    rhos, ratios, per_query = [], [], []
    for i in range(n):
        if not nonconstant[i]:
            per_query.append((None, None))
            continue
        rho, ratio = action_value_smoothness(actions_by_row[i], v[i])
        per_query.append((rho, ratio))
        if np.isfinite(rho):
            rhos.append(rho)
        ratios.append(ratio)
    return {
        "nonconstant_fraction": equal_stratum_mean(nonconstant.astype(np.float64), strata),
        "n_nonconstant": int(nonconstant.sum()),
        "mean_value_range": equal_stratum_mean(value_range, strata),
        "smoothness_median_rho": (float(np.median(rhos)) if rhos else float("nan")),
        "near_pair_ratio_median": (float(np.nanmedian(ratios)) if ratios else float("nan")),
        "per_query": per_query,
        "nonconstant": nonconstant,
    }


def analyze_action_persistence(configuration: ConfigMap) -> int:
    """Measure executed-value reliability across action persistence durations."""
    cfg_raw = configuration
    cfg = ActionPersistenceConfig.from_mapping(cfg_raw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest = load_manifest(cfg.manifest_path)
    rows = manifest["rows"]
    strata = np.asarray([r["stratum"] for r in rows])
    A = np.load(cfg.pools_dir / "pools.npz")["A"]
    actions_by_row = [A[r["query_id"], r["pool_replicate"]].astype(np.float64) for r in rows]

    grid, csv_rows = [], []
    m_star = None
    for m in cfg.m_grid:
        merged = cfg.merged_root / f"action_persistence_{m}" / "merged" / "full_pool_rollouts.npz"
        z = np.load(merged)
        ret = z["disc_return"]
        if not np.array_equal(ret[:, :, 0, 0], ret[:, :, 1, 0]):
            raise RuntimeError(
                f"m={m}: duplicate verification FAILED — implementation blocker (plan §17.2/§18.1)"
            )
        s = action_repeat_statistics(ret[:, :, 0, 0], actions_by_row, strata)
        ok = (
            np.isfinite(s["nonconstant_fraction"])
            and s["nonconstant_fraction"] >= cfg.frac_target
            and np.isfinite(s["mean_value_range"])
            and s["mean_value_range"] >= cfg.h_min
            and np.isfinite(s["smoothness_median_rho"])
            and s["smoothness_median_rho"] >= cfg.smoothness_target
        )
        if ok and m_star is None:
            m_star = int(m)
        grid.append(
            {
                "m": int(m),
                "qualifies": bool(ok),
                "merged_sha16": sha256_file(merged)[:16],
                **{
                    k: s[k]
                    for k in (
                        "nonconstant_fraction",
                        "n_nonconstant",
                        "mean_value_range",
                        "smoothness_median_rho",
                        "near_pair_ratio_median",
                    )
                },
            }
        )
        for i, row in enumerate(rows):
            rho, ratio = s["per_query"][i]
            csv_rows.append(
                {
                    "m": int(m),
                    "query_id": row["query_id"],
                    "stratum": row["stratum_name"],
                    "nonconstant": int(s["nonconstant"][i]),
                    "rho_dist_value": rho,
                    "near_pair_ratio": ratio,
                }
            )

    with (cfg.output_dir / "persistence_calibration_queries.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    decision = {
        "protocol": "action_persistence",
        "decision": ("FEASIBLE" if m_star is not None else "INFEASIBLE"),
        "m_star": m_star,
        "grid": grid,
        "targets": {
            "nonconstant_fraction": cfg.frac_target,
            "h_min": cfg.h_min,
            "smoothness_median_rho": cfg.smoothness_target,
        },
        "manifest_rows_sha256": manifest["rows_sha256"],
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = cfg.output_dir / "persistence_calibration_antmaze.json"
    out.write_text(json.dumps(decision, indent=2))
    print(f"[action_persistence] {decision['decision']} m_star={m_star} -> {out}", flush=True)
    return 0
