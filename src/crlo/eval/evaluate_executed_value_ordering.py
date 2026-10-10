"""Executed-value ordering, critic comparison, and seed replication.

Refuses to run without a passing, hash-matched Value-reliability JSON (the score
reveal is only licensed after the score-blind gate closes). Loads the
frozen manifest, the merged rollout tensor, critic score tensors, the
canonical retrieval CSV, and the frozen primary episode-resample matrix
(reused verbatim: Seed replication never draws new resamples).

Score tensors per (critic, seed) are looked up in ``scores_dir`` as
``scores_{critic}_s{seed}.npz`` first, then — for the historical cached
seed-42 tensors — ``scores_{critic}.npz``. Cached tensors of shape
(Q, R, K) are indexed by each manifest row's (query, pool replicate);
Frozen executed-value rescored tensors of shape (n_manifest, K) are used as-is.
Any non-finite score in the manifest-selected tensor is an
implementation blocker (plan §7.2): the run fails loudly and nothing is
imputed.

Modes:

- ``critic_comparison`` — seed-42 discovery: per-critic Kendall tau-b vs the
  ten-rollout value, selected value at K in the frozen K-curve,
  cross-fitted winner gaps, top-decile stability, the four-effect
  Holm family, and the frozen BROAD/POINT-LIKE/PARTIAL/NO-GO label.
- ``seed_replication`` — seeds 42-44 confirmation: per-seed metrics, equal-weight
  three-seed effects on the same resamples, per-seed preservation
  checks, and the no-upgrade label rule.

Outputs (plan §10.3): ``full_pool_metrics_<task>.csv``,
``pairwise_contrasts_<task>.csv``, ``k_curves_<task>.csv``,
``critic_comparison_<task>.json`` / ``per_seed_metrics_<task>.csv``,
``seed_replication_<task>.json``.
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

from crlo.eval.config_schemas import ExecutedValueOrderingConfig
from crlo.eval.query_manifests import load_manifest, sha256_file  # noqa: E402
from crlo.eval.value_statistics import (  # noqa: E402
    bootstrap_equal_stratum,
    classify_critic_comparison,
    classify_seed_replication,
    crossfit_winner,
    equal_stratum_mean,
    holm_adjust,
    kendall_tau_b,
    normalized_ratios,
    null_centered_p,
    tie_break_argmax,
    top_set,
)

EFFECTS = ("tau_raw", "tau_cos", "val_raw", "val_cos")


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()


def _load_scores(
    cfg: ExecutedValueOrderingConfig, manifest: dict, critic: str, seed: int
) -> np.ndarray:
    """Manifest-selected (n, K) score tensor for one critic checkpoint."""
    cands = [cfg.scores_dir / f"scores_{critic}_s{seed}.npz"]
    if seed == 42:
        cands.append(cfg.scores_dir / f"scores_{critic}.npz")
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no score tensor for {critic} seed {seed}: tried {cands}")
    z = np.load(path)
    sc = np.asarray(z["scores"], np.float64)
    if sc.ndim == 3:
        rows = manifest["rows"]
        sc = np.stack([sc[r["query_id"], r["pool_replicate"]] for r in rows])
    if sc.shape != (len(manifest["rows"]), cfg.k):
        raise ValueError(f"{path}: score shape {sc.shape} != ({len(manifest['rows'])}, {cfg.k})")
    if not np.isfinite(sc).all():
        raise ValueError(
            f"{path}: non-finite scores on the frozen manifest — "
            f"implementation blocker (plan §7.2); never imputed"
        )
    return sc


def _retrieval_auc(cfg: ExecutedValueOrderingConfig) -> dict[tuple[str, int], float]:
    """(critic, training_seed) -> retrieval_auc from the canonical CSV."""
    out: dict[tuple[str, int], float] = {}
    with cfg.retrieval_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["critic"], int(row["training_seed"]))] = float(row["retrieval_auc"])
    return out


def _critic_query_metrics(
    cfg: ExecutedValueOrderingConfig, manifest: dict, ret: np.ndarray, scores: np.ndarray
) -> dict:
    """Per-query per-critic metric arrays on the frozen manifest."""
    rows = manifest["rows"]
    n = len(rows)
    v10 = ret.mean(axis=(2, 3))  # (n, K)
    v_a = ret[:, :, 0, :].mean(-1)
    v_b = ret[:, :, 1, :].mean(-1)
    tau = np.empty(n)
    flat_value = np.zeros(n, int)
    flat_score = np.zeros(n, int)
    sel = {kk: np.empty(n) for kk in cfg.k_curve}
    sel_idx = np.empty(n, np.int64)
    top_hit = np.empty(n)
    w_cf = np.empty(n)
    pool_mean = np.empty(n)
    for i, row in enumerate(rows):
        prio = np.asarray(row["tie_priority"])
        fv = np.ptp(v10[i]) == 0.0
        fs = np.ptp(scores[i]) == 0.0
        flat_value[i], flat_score[i] = int(fv), int(fs)
        t = kendall_tau_b(scores[i], v10[i])
        tau[i] = 0.0 if not np.isfinite(t) else t  # frozen flat rule
        for kk in cfg.k_curve:
            j = tie_break_argmax(scores[i, :kk], prio[:kk])
            sel[kk][i] = v10[i, j]
        j64 = tie_break_argmax(scores[i], prio)
        sel_idx[i] = j64
        ts_a = top_set(v_a[i], cfg.top_decile_size, prio)
        ts_b = top_set(v_b[i], cfg.top_decile_size, prio)
        top_hit[i] = 0.5 * (float(j64 in ts_a) + float(j64 in ts_b))
        cf = crossfit_winner(v_a[i], v_b[i], prio)
        w_cf[i] = cf.w_cf
        pool_mean[i] = cf.pool_mean
    return {
        "tau": tau,
        "flat_value": flat_value,
        "flat_score": flat_score,
        "selected": sel,
        "selected_idx": sel_idx,
        "top_hit": top_hit,
        "w_cf": w_cf,
        "pool_mean": pool_mean,
        "v10_mean": v10.mean(-1),
    }


def _seed_effects(m: dict[str, dict]) -> dict[str, np.ndarray]:
    """Per-query effect arrays for the frozen four-test family."""
    k64 = max(m["td_q"]["selected"].keys())
    return {
        "tau_raw": m["td_q"]["tau"] - m["raw_bilinear"]["tau"],
        "tau_cos": m["td_q"]["tau"] - m["cosine"]["tau"],
        "val_raw": m["td_q"]["selected"][k64] - m["raw_bilinear"]["selected"][k64],
        "val_cos": m["td_q"]["selected"][k64] - m["cosine"]["selected"][k64],
    }


def evaluate_executed_value_ordering(configuration: ConfigMap) -> int:
    """Compare critic ordering using frozen returns and episode resamples.

    Requires passing reliability evidence matched to the manifest and return
    tensor. Reuses the recorded primary episode-resample matrix for comparison
    and seed replication. Writes per-critic metrics, candidate-count curves,
    contrasts, and the mode-specific decision JSON under ``output_dir``.

    Args:
        configuration: Fields accepted by ``ExecutedValueOrderingConfig``: mode,
            frozen input paths, critics/seeds, and statistical settings.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg_raw = configuration
    cfg = ExecutedValueOrderingConfig.from_mapping(cfg_raw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest = load_manifest(cfg.manifest_path)
    task_prefix = manifest["meta"]["task"].split("-")[0]

    reliability = json.loads(cfg.reliability_path.read_text())
    if not reliability.get("pass"):
        raise RuntimeError(
            "Value reliability did not pass — score reveal is not licensed (plan §6.3)"
        )
    if reliability["manifest_rows_sha256"] != manifest["rows_sha256"]:
        raise ValueError("Value-reliability JSON was computed on a different manifest")
    merged_sha16 = sha256_file(cfg.merged_path)[:16]
    if reliability["merged_sha16"] != merged_sha16:
        raise ValueError("Value-reliability JSON was computed on different rollouts")
    ret = np.load(cfg.merged_path)["disc_return"]

    boot_npz = np.load(cfg.resample_indices_path)
    resamples = np.asarray(boot_npz["resamples"])
    seed_words = boot_npz["seed_words"].tolist()
    if seed_words[1] != 1:
        raise ValueError(
            "Comparison and replication stages must reuse the primary resample matrix "
            f"(seed words {seed_words}); never a new draw (plan §10.3)"
        )

    rows = manifest["rows"]
    strata = np.asarray([r["stratum"] for r in rows])
    episodes = np.asarray([r["episode_id"] for r in rows])
    auc = _retrieval_auc(cfg)

    per_seed: dict[int, dict[str, dict]] = {}
    for seed in cfg.seeds:
        per_seed[seed] = {
            c: _critic_query_metrics(cfg, manifest, ret, _load_scores(cfg, manifest, c, seed))
            for c in cfg.critics
        }

    metrics_rows, kcurve_rows = [], []
    for seed, per_critic in per_seed.items():
        for c, m in per_critic.items():
            k64 = max(cfg.k_curve)
            ratios = normalized_ratios(m["w_cf"], m["selected"][k64], m["v10_mean"])
            metrics_rows.append(
                {
                    "task": manifest["meta"]["task"],
                    "critic": c,
                    "training_seed": seed,
                    "retrieval_auc": auc.get((c, seed), float("nan")),
                    "mean_tau_b": equal_stratum_mean(m["tau"], strata),
                    "selected_v10_k64": equal_stratum_mean(m["selected"][k64], strata),
                    "selected_minus_pool_mean": equal_stratum_mean(
                        m["selected"][k64] - m["v10_mean"], strata
                    ),
                    "winner_minus_critic_gap": equal_stratum_mean(
                        m["w_cf"] - m["selected"][k64], strata
                    ),
                    "top_decile_stability": equal_stratum_mean(m["top_hit"], strata),
                    "n_flat_value": int(m["flat_value"].sum()),
                    "n_flat_score": int(m["flat_score"].sum()),
                    "normalized_winner_gap": ratios["normalized_winner_gap"],
                    "captured": ratios["captured"],
                    "ratio_denominator": ratios["denominator"],
                    "n_nonpositive_headroom": ratios["n_nonpositive_headroom"],
                }
            )
            for kk in cfg.k_curve:
                kcurve_rows.append(
                    {
                        "task": manifest["meta"]["task"],
                        "critic": c,
                        "training_seed": seed,
                        "k": kk,
                        "selected_v10": equal_stratum_mean(m["selected"][kk], strata),
                    }
                )

    fields = list(metrics_rows[0].keys())
    name = "full_pool_metrics" if cfg.mode == "critic_comparison" else "per_seed_metrics"
    with (cfg.output_dir / f"{name}_{task_prefix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)
    with (cfg.output_dir / f"k_curves_{task_prefix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kcurve_rows[0].keys()))
        w.writeheader()
        w.writerows(kcurve_rows)

    # Equal-weight mean effects across seeds. equal_stratum_mean is linear
    # in per-query values and strata are seed-invariant, so averaging the
    # per-query effect arrays across seeds keeps every seed of an episode
    # cluster together in each resample (plan §8.3).
    eff_arrays = {
        key: np.mean([_seed_effects(per_seed[s])[key] for s in cfg.seeds], axis=0)
        for key in EFFECTS
    }
    theta = {k: equal_stratum_mean(v, strata) for k, v in eff_arrays.items()}
    boot = {
        k: bootstrap_equal_stratum(v, strata, episodes, resamples) for k, v in eff_arrays.items()
    }
    p_raw = {k: null_centered_p(theta[k], boot[k]) for k in EFFECTS}
    p_holm = dict(zip(EFFECTS, holm_adjust(np.array([p_raw[k] for k in EFFECTS])), strict=True))
    ci = {
        k: (
            float(np.quantile(boot[k], 0.025, method="linear")),
            float(np.quantile(boot[k], 0.975, method="linear")),
        )
        for k in EFFECTS
    }
    ci_lo = {k: ci[k][0] for k in EFFECTS}

    contrasts = [
        {
            "effect": k,
            "estimate": theta[k],
            "p_raw": p_raw[k],
            "p_holm": p_holm[k],
            "ci_lo_95": ci[k][0],
            "ci_hi_95": ci[k][1],
        }
        for k in EFFECTS
    ]
    with (cfg.output_dir / f"pairwise_contrasts_{task_prefix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys()))
        w.writeheader()
        w.writerows(contrasts)

    def seed_stats(seed: int) -> dict:
        m = per_seed[seed]
        eff = _seed_effects(m)
        return {
            "auc_raw": auc.get(("raw_bilinear", seed), float("nan")),
            "auc_cos": auc.get(("cosine", seed), float("nan")),
            "tau_raw": equal_stratum_mean(m["raw_bilinear"]["tau"], strata),
            "tau_cos": equal_stratum_mean(m["cosine"]["tau"], strata),
            "d_tau_raw": equal_stratum_mean(eff["tau_raw"], strata),
            "d_tau_cos": equal_stratum_mean(eff["tau_cos"], strata),
            "d_val_raw": equal_stratum_mean(eff["val_raw"], strata),
            "d_val_cos": equal_stratum_mean(eff["val_cos"], strata),
        }

    pooled = {
        key: float(np.mean([seed_stats(s)[key] for s in cfg.seeds]))
        for key in seed_stats(cfg.seeds[0])
    }
    label = classify_critic_comparison(
        auc_raw=pooled["auc_raw"],
        auc_cosine=pooled["auc_cos"],
        auc_min=cfg.auc_min,
        tau_raw=pooled["tau_raw"],
        tau_cosine=pooled["tau_cos"],
        tau_cap=cfg.tau_cap,
        d_tau_raw=pooled["d_tau_raw"],
        d_tau_cos=pooled["d_tau_cos"],
        delta_tau=cfg.delta_tau,
        d_val_raw=pooled["d_val_raw"],
        d_val_cos=pooled["d_val_cos"],
        delta_v=cfg.delta_v,
        holm_p=p_holm,
        ci_lo=ci_lo,
    )

    decision = {
        "mode": cfg.mode,
        "task": manifest["meta"]["task"],
        "manifest_rows_sha256": manifest["rows_sha256"],
        "merged_sha16": merged_sha16,
        "resample_seed_words": seed_words,
        "resample_sha16": sha256_file(cfg.resample_indices_path)[:16],
        "seeds": list(cfg.seeds),
        "pooled": pooled,
        "per_seed": {str(s): seed_stats(s) for s in cfg.seeds},
        "contrasts": contrasts,
        "thresholds": {
            "delta_tau": cfg.delta_tau,
            "delta_v": cfg.delta_v,
            "tau_cap": cfg.tau_cap,
            "auc_min": cfg.auc_min,
        },
        "label_detail": label,
        "git": _git(),
        "elapsed_s": round(time.time() - t0, 1),
    }

    if cfg.mode == "critic_comparison":
        decision["label"] = label["label"]
        out = cfg.output_dir / f"critic_comparison_{task_prefix}.json"
        out.write_text(json.dumps(decision, indent=2))
        print(f"[critic_comparison] label={decision['label']} -> {out}", flush=True)
        return 0

    critic_comparison_json = json.loads(cfg.comparison_path.read_text())
    discovery = critic_comparison_json["label"]
    if discovery not in ("POINT_LIKE_GO", "BROAD_GO"):
        raise RuntimeError(
            f"Seed replication is conditional on a Critic-comparison GO; recorded {discovery}"
        )
    capped = classify_seed_replication(discovery, label["label"])
    # Per-seed preservation checks for the capped label (plan §8.3).
    per_seed_ok = {}
    for s in cfg.seeds:
        st = seed_stats(s)
        need_val_raw = cfg.delta_v if capped == "BROAD_GO" else 0.0
        per_seed_ok[str(s)] = bool(
            st["auc_raw"] >= cfg.auc_min
            and st["auc_cos"] >= cfg.auc_min
            and st["tau_raw"] <= cfg.tau_cap
            and st["tau_cos"] <= cfg.tau_cap
            and st["d_tau_raw"] >= cfg.delta_tau
            and st["d_tau_cos"] >= cfg.delta_tau
            and st["d_val_cos"] >= cfg.delta_v
            and st["d_val_raw"] >= need_val_raw
        )
    required = EFFECTS if capped == "BROAD_GO" else ("tau_raw", "tau_cos", "val_cos")
    tests_ok = all(p_holm[k] < 0.05 and ci_lo[k] > 0.0 for k in required)
    confirmed = capped in ("POINT_LIKE_GO", "BROAD_GO") and all(per_seed_ok.values()) and tests_ok
    decision |= {
        "classify_critic_comparison": discovery,
        "label": capped if confirmed else "SEED_REPLICATION_FAILED",
        "per_seed_pass": per_seed_ok,
        "required_tests": list(required),
        "tests_pass": bool(tests_ok),
        "confirmed": bool(confirmed),
    }
    out = cfg.output_dir / f"seed_replication_{task_prefix}.json"
    out.write_text(json.dumps(decision, indent=2))
    print(
        f"[seed_replication] confirmed={confirmed} label={decision['label']} -> {out}", flush=True
    )
    return 0
