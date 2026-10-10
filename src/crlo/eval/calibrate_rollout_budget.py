"""Frozen executed-valueb Arm-A instrument calibration (plan §17.1, score-blind).

Consumes the ``pilot_r25`` merged tensor and the original 5+5 pilot
tensor. Verifies the deterministic-replay contract (rollout ids 0-4
must reproduce the original pilot bit-exactly — any mismatch is an
implementation blocker), measures the §6.1 reliability statistics at
the frozen nested budgets, applies the frozen 1/sqrt(r) and
Spearman-Brown extrapolations, and applies the no-discretion R* grid
rule. Writes ``calibration_antmaze.json`` + a per-query CSV.

This driver reads rollout artifacts and the manifest only; it cannot
accept critic labels or score paths.
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

from crlo.eval.config_schemas import RolloutBudgetConfig
from crlo.eval.query_manifests import load_manifest, sha256_file  # noqa: E402
from crlo.eval.value_statistics import (  # noqa: E402
    equal_stratum_mean,
    query_reliability,
    spearman_rho,
)


def spearman_brown(rho_r0: float, r0: int, r: int) -> float:
    """Predicted split-half agreement at r draws/half from r0 draws/half."""
    if not np.isfinite(rho_r0) or rho_r0 <= 0.0:
        return float("nan")
    rho1 = rho_r0 / (r0 - (r0 - 1) * rho_r0)
    return float(r * rho1 / (1 + (r - 1) * rho1))


def measured_stats(ret: np.ndarray, strata: np.ndarray, r: int) -> dict:
    """§6.1 resolvable fraction + conditional agreement at budget ``r``."""
    resolvable, rho = [], []
    for i in range(ret.shape[0]):
        rel = query_reliability(ret[i, :, 0, :r], ret[i, :, 1, :r])
        resolvable.append(rel.resolvable)
        rho.append(rel.rho)
    resolvable = np.asarray(resolvable, bool)
    rho = np.asarray(rho)
    return {
        "r": r,
        "resolvable_fraction": equal_stratum_mean(resolvable.astype(np.float64), strata),
        "agreement": equal_stratum_mean(rho, strata, valid=resolvable),
        "n_resolvable": int(resolvable.sum()),
        "resolvable": resolvable,
        "rho": rho,
    }


def predicted_stats(ret: np.ndarray, strata: np.ndarray, r0: int, r: int, agree_r0: float) -> dict:
    """Frozen §17.1 extrapolation: N ~ 1/sqrt(r), S frozen, SB agreement."""
    from crlo.eval.value_statistics import candidate_spread, paired_noise

    scale = float(np.sqrt(r0 / r))
    resolvable = []
    for i in range(ret.shape[0]):
        y_a, y_b = ret[i, :, 0, :r0], ret[i, :, 1, :r0]
        v_a, v_b = y_a.mean(-1), y_b.mean(-1)
        ok = (
            candidate_spread(v_a) > 2 * paired_noise(y_a) * scale
            and candidate_spread(v_b) > 2 * paired_noise(y_b) * scale
            and np.isfinite(spearman_rho(v_a, v_b))
        )
        resolvable.append(ok)
    resolvable = np.asarray(resolvable, bool)
    return {
        "r": r,
        "resolvable_fraction": equal_stratum_mean(resolvable.astype(np.float64), strata),
        "agreement": spearman_brown(agree_r0, r0, r),
        "n_resolvable": int(resolvable.sum()),
        "predicted": True,
    }


def calibrate_rollout_budget(configuration: ConfigMap) -> int:
    """Estimate the rollout budget required by the frozen reliability rule."""
    cfg_raw = configuration
    cfg = RolloutBudgetConfig.from_mapping(cfg_raw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest = load_manifest(cfg.manifest_path)
    strata = np.asarray([r["stratum"] for r in manifest["rows"]])
    z = np.load(cfg.merged_path)
    ret = z["disc_return"]  # (n, K, 2, 25)
    if ret.shape[-1] != cfg.rollouts_per_half:
        raise ValueError(f"merged R {ret.shape[-1]} != {cfg.rollouts_per_half}")

    base = np.load(cfg.baseline_merged_path)
    r_base = base["disc_return"].shape[-1]
    for field in ("disc_return", "success", "steps_to_goal"):
        if not np.array_equal(np.load(cfg.merged_path)[field][..., :r_base], base[field]):
            raise RuntimeError(
                f"deterministic-replay regression FAILED on {field}: "
                f"rollout ids 0-{r_base - 1} do not reproduce the original "
                f"pilot bit-exactly — implementation blocker (plan §17.1)"
            )

    grid = []
    stats_by_r: dict[int, dict] = {}
    for r in cfg.r_grid:
        s = measured_stats(ret, strata, r)
        stats_by_r[r] = s
        grid.append({k: s[k] for k in ("r", "resolvable_fraction", "agreement", "n_resolvable")})
    r0 = cfg.rollouts_per_half
    agree_r0 = stats_by_r[r0]["agreement"]
    for r in cfg.r_extrap:
        s = predicted_stats(ret, strata, r0, r, agree_r0)
        stats_by_r[r] = s
        grid.append(
            {k: s[k] for k in ("r", "resolvable_fraction", "agreement", "n_resolvable")}
            | {"predicted": True}
        )

    r_star = None
    for r in cfg.r_choices:
        s = stats_by_r[r]
        ok = (
            np.isfinite(s["agreement"])
            and s["agreement"] >= cfg.agreement_target
            and np.isfinite(s["resolvable_fraction"])
            and s["resolvable_fraction"] >= cfg.frac_target
        )
        if ok:
            r_star = int(r)
            break

    s25 = stats_by_r[r0]
    with (cfg.output_dir / "calibration_queries.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "stratum", "resolvable_25", "rho_25"])
        w.writeheader()
        for i, row in enumerate(manifest["rows"]):
            w.writerow(
                {
                    "query_id": row["query_id"],
                    "stratum": row["stratum_name"],
                    "resolvable_25": int(s25["resolvable"][i]),
                    "rho_25": s25["rho"][i],
                }
            )

    decision = {
        "arm": "A1",
        "decision": ("A1_FEASIBLE" if r_star is not None else "A1_INFEASIBLE"),
        "r_star": r_star,
        "grid": grid,
        "targets": {"agreement": cfg.agreement_target, "resolvable_fraction": cfg.frac_target},
        "r_choices": list(cfg.r_choices),
        "replay_regression": f"bit-exact on rollout ids 0-{r_base - 1}",
        "manifest_rows_sha256": manifest["rows_sha256"],
        "merged_sha16": sha256_file(cfg.merged_path)[:16],
        "baseline_sha16": sha256_file(cfg.baseline_merged_path)[:16],
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = cfg.output_dir / "calibration_antmaze.json"
    out.write_text(json.dumps(decision, indent=2))
    print(f"[rollout_budget] {decision['decision']} r_star={r_star} -> {out}", flush=True)
    return 0
