"""Reference-ordering metrics: resolvability precheck + fixed-query ordering/selection metrics.

Two modes over the cached candidate pools (frozen reference-ensemble protocol):

``mode=precheck`` (reference-only, run BEFORE any critic is scored):
  computes, per (query, replicate) K=64 pool: spread S_q (P90-P10 of the
  ensemble-mean reference value), noise N_q (median pooled reference standard
  error), the resolvable flag S_q > 2 N_q, ensemble top-1 agreement and mean
  pairwise Spearman, and bad-tail presence vs the logged-action anchor.
  Emits the frozen Gate-1 stop/go verdict (>=50% resolvable and >=100
  resolvable queries, overall or in a predeclared stratum).

``mode=metrics`` (after critics are scored):
  A. conditional ordering at K=64: within-query Kendall tau_b vs the ensemble
     mean, pairwise accuracy (all + reference-confident pairs), top-decile
     hit, top-1 agreement -- macro-averaged over queries;
  B. best-of-K selection: regret / normalized regret / selected value /
     top-decile hit across nested K, with a paired bootstrap CI over queries
     (identical resamples for every critic);
  C. cross-query vs within-query diagnostic: pooled global Kendall, macro
     within-query Kendall, pooled Kendall after per-query mean-centering.

All metrics are reported for all queries and for the reference-resolvable
subset, overall and per goal-distance stratum.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]


STRATUM_NAMES = ("near", "medium", "far")


def _load_refs(pools_dir: Path, ref_labels: list[str]):
    """Stack reference score arrays -> (n_refs, Q, R, K), (n_refs, Q)."""
    scores, logged = [], []
    for lab in ref_labels:
        z = np.load(pools_dir / f"scores_{lab}.npz")
        scores.append(z["scores"])
        logged.append(z["scores_logged"])
    return np.stack(scores), np.stack(logged)


def _ref_stats(refs: np.ndarray, refs_logged: np.ndarray):
    """Ensemble mean/se plus per-pool spread, noise, resolvable, bad tail."""
    n_refs = refs.shape[0]
    qbar = refs.mean(0)  # (Q, R, K)
    se = (
        refs.std(0, ddof=1) / np.sqrt(n_refs)  # (Q, R, K)
        if n_refs > 1
        else np.zeros_like(qbar)
    )
    spread = np.percentile(qbar, 90, axis=2) - np.percentile(qbar, 10, axis=2)  # (Q, R)
    noise = np.median(se, axis=2)  # (Q, R)
    resolvable = spread > 2.0 * noise if n_refs > 1 else np.ones_like(spread, bool)
    anchor = refs_logged.mean(0)  # (Q,)
    bad = qbar < (anchor[:, None, None] - 2.0 * noise[:, :, None])
    bad_tail_frac = bad.mean(2)  # (Q, R)
    return qbar, se, spread, noise, resolvable, bad_tail_frac


def _agreement(refs: np.ndarray) -> tuple[float, float]:
    """Top-1 agreement rate and mean pairwise Spearman across references."""
    n_refs, Q, R, K = refs.shape
    if n_refs < 2:
        return float("nan"), float("nan")
    tops = refs.argmax(3)  # (n_refs, Q, R)
    agree, rhos = [], []
    for i, j in combinations(range(n_refs), 2):
        agree.append((tops[i] == tops[j]).mean())
        flat_i = refs[i].reshape(Q * R, K)
        flat_j = refs[j].reshape(Q * R, K)
        idx = np.random.default_rng(0).choice(Q * R, min(500, Q * R), False)
        rhos.append(np.mean([spearmanr(flat_i[t], flat_j[t]).statistic for t in idx]))
    return float(np.mean(agree)), float(np.mean(rhos))


def _pairwise_acc(sc: np.ndarray, qb: np.ndarray, se: np.ndarray, iu: tuple) -> tuple[float, float]:
    """Pairwise ordering accuracy over one pool: (all pairs, confident)."""
    dv = qb[iu[0]] - qb[iu[1]]
    dsc = sc[iu[0]] - sc[iu[1]]
    se_pair = np.sqrt(se[iu[0]] ** 2 + se[iu[1]] ** 2)
    informative = dv != 0
    correct = (np.sign(dsc) == np.sign(dv)) & informative
    acc = correct.sum() / max(informative.sum(), 1)
    conf = informative & (np.abs(dv) > 2.0 * se_pair)
    acc_conf = (correct & conf).sum() / max(conf.sum(), 1) if conf.any() else float("nan")
    return float(acc), float(acc_conf)


def _query_metrics(sc: np.ndarray, qbar: np.ndarray, se: np.ndarray):
    """Per-(query, replicate) ordering metrics at K = k_max."""
    Q, R, K = sc.shape
    iu = np.triu_indices(K, 1)
    tau = np.empty((Q, R))
    acc = np.empty((Q, R))
    accc = np.empty((Q, R))
    top_dec = np.empty((Q, R))
    top1 = np.empty((Q, R))
    n_dec = max(1, K // 10)
    for q in range(Q):
        for r in range(R):
            tau[q, r] = kendalltau(sc[q, r], qbar[q, r]).statistic
            acc[q, r], accc[q, r] = _pairwise_acc(sc[q, r], qbar[q, r], se[q, r], iu)
            sel = int(np.argmax(sc[q, r]))
            dec_thresh = np.partition(qbar[q, r], K - n_dec)[K - n_dec]
            top_dec[q, r] = float(qbar[q, r, sel] >= dec_thresh)
            top1[q, r] = float(sel == int(np.argmax(qbar[q, r])))
    return {
        "kendall_tau": tau,
        "pairwise_acc": acc,
        "pairwise_acc_confident": accc,
        "top_decile_hit": top_dec,
        "top1_agreement": top1,
    }


def _bestofk(sc: np.ndarray, qbar: np.ndarray, ks: list[int]):
    """Best-of-K selection curves; per-(q, r) values for each K."""
    out = {}
    eps = 1e-9
    for k in ks:
        pre_s, pre_q = sc[:, :, :k], qbar[:, :, :k]
        sel = pre_s.argmax(2)
        q_sel = np.take_along_axis(pre_q, sel[:, :, None], 2)[:, :, 0]
        q_max = pre_q.max(2)
        q_mean = pre_q.mean(2)
        n_dec = max(1, k // 10)
        dec_thresh = np.partition(pre_q, k - n_dec, axis=2)[:, :, k - n_dec]
        out[k] = {
            "regret": q_max - q_sel,
            "norm_regret": (q_max - q_sel) / (q_max - q_mean + eps),
            "selected_value": q_sel,
            "top_decile_hit": (q_sel >= dec_thresh).astype(float),
        }
    return out


def _emit(rows, critic, metric, k, stratum, subset, per_qr, sel, boot_u):
    """Append a row with a paired-bootstrap 95% CI over subset queries.

    ``boot_u`` is a fixed (n_boot, Q) uniform array shared by every critic,
    so resamples are identical across critics (paired comparisons).
    """
    per_q = np.nanmean(per_qr, axis=1)  # replicate-macro
    vals = per_q[sel]
    m = len(vals)
    if m == 0:
        return
    idx = np.floor(boot_u[:, :m] * m).astype(int)
    means = np.nanmean(vals[idx], axis=1)
    rows.append(
        {
            "critic": critic,
            "metric": metric,
            "K": k,
            "stratum": stratum,
            "subset": subset,
            "value": round(float(np.nanmean(vals)), 4),
            "ci_lo": round(float(np.nanpercentile(means, 2.5)), 4),
            "ci_hi": round(float(np.nanpercentile(means, 97.5)), 4),
            "n_queries": int(m),
        }
    )


def evaluate_reference_ordering(configuration: ConfigMap) -> int:
    """Check reference reliability or compare fixed-query candidate ordering.

    The reference-only ``precheck`` mode reports resolvability and its gate.
    The ``metrics`` mode compares cached critic scores with the reference
    ensemble on all and resolvable queries, using shared bootstrap resamples.
    Writes the mode's summaries and metric CSVs under ``output_dir``, or
    ``pools_dir`` when no output override is supplied.

    Args:
        configuration: Mode, cached pool/score paths, reference and critic labels,
            candidate counts, bootstrap settings, and output path from the command.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = configuration
    pools_dir = Path(cfg.get("pools_dir"))
    mode = str(cfg["mode"])
    ref_labels = list(cfg["refs"])
    task = str(cfg["task"])
    out_override = cfg.get("output_dir", None)
    out_dir = Path(out_override) if out_override else pools_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(pools_dir / "pools.npz")
    stratum = z["stratum"]
    refs, refs_logged = _load_refs(pools_dir, ref_labels)
    qbar, se, spread, noise, resolvable, bad_tail = _ref_stats(refs, refs_logged)
    Q, R, K = qbar.shape
    res_q = resolvable.mean(1) >= 0.5  # query-level majority

    if mode == "precheck":
        top1_agree, rho = _agreement(refs)
        rows, verdict = (
            [],
            {"criterion": ">=50% resolvable and >=100 resolvable queries (overall or any stratum)"},
        )
        for name, mask in [("overall", np.ones(Q, bool))] + [
            (STRATUM_NAMES[s], stratum == s) for s in range(3)
        ]:
            frac = float(res_q[mask].mean())
            cnt = int(res_q[mask].sum())
            rows.append(
                {
                    "population": name,
                    "n_queries": int(mask.sum()),
                    "resolvable_frac": round(frac, 4),
                    "resolvable_count": cnt,
                    "spread_median": round(float(np.median(spread[mask])), 5),
                    "noise_median": round(float(np.median(noise[mask])), 5),
                    "bad_tail_frac_mean": round(float(bad_tail[mask].mean()), 4),
                    "pass": int(frac >= 0.5 and cnt >= 100),
                }
            )
            verdict[name] = bool(frac >= 0.5 and cnt >= 100)
        verdict["ensemble_top1_agreement"] = top1_agree
        verdict["ensemble_mean_pairwise_spearman"] = rho
        verdict["n_refs"] = len(ref_labels)
        verdict["go"] = any(r["pass"] for r in rows)
        out = out_dir / f"precheck_{task}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (out_dir / f"precheck_{task}.json").write_text(json.dumps(verdict, indent=2))
        for r in rows:
            print(
                f"[reference_reliability] {r['population']:8s} "
                f"resolvable {r['resolvable_frac']:.1%} "
                f"(n={r['resolvable_count']}) "
                f"spread~{r['spread_median']:.4f} "
                f"noise~{r['noise_median']:.4f} "
                f"bad-tail~{r['bad_tail_frac_mean']:.1%} "
                f"pass={r['pass']}",
                flush=True,
            )
        print(
            f"[reference_reliability] agreement: top1={top1_agree:.3f} "
            f"spearman={rho:.3f}  GO={verdict['go']}",
            flush=True,
        )
        return 0

    # ---- mode=metrics ----
    critic_labels = list(cfg.get("critics") or [])
    ks = list(cfg["k_grid"])
    n_boot = int(cfg["n_boot"])
    boot_seed = int(cfg["boot_seed"])
    rng = np.random.default_rng(boot_seed)
    boot_u = rng.random((n_boot, Q))  # shared across critics

    rows = []
    pops = [("overall", np.ones(Q, bool))] + [(STRATUM_NAMES[s], stratum == s) for s in range(3)]
    for lab in critic_labels:
        sc = np.load(pools_dir / f"scores_{lab}.npz")["scores"]
        qm = _query_metrics(sc, qbar, se)
        bk = _bestofk(sc, qbar, ks)
        # C: cross-query vs within-query diagnostic (overall, all queries)
        flat_sc, flat_qb = sc.reshape(-1), qbar.reshape(-1)
        sub = np.random.default_rng(7).choice(len(flat_sc), min(20000, len(flat_sc)), False)
        global_tau = kendalltau(flat_sc[sub], flat_qb[sub]).statistic
        centered = (sc - sc.mean(2, keepdims=True)).reshape(-1)
        centered_tau = kendalltau(centered[sub], flat_qb[sub]).statistic
        rows.append(
            {
                "critic": lab,
                "metric": "global_kendall_tau",
                "K": K,
                "stratum": "overall",
                "subset": "all",
                "value": round(float(global_tau), 4),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_queries": Q,
            }
        )
        rows.append(
            {
                "critic": lab,
                "metric": "global_kendall_tau_centered",
                "K": K,
                "stratum": "overall",
                "subset": "all",
                "value": round(float(centered_tau), 4),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_queries": Q,
            }
        )
        for pop_name, pop_mask in pops:
            for subset, submask in [("all", pop_mask), ("resolvable", pop_mask & res_q)]:
                sel = np.nonzero(submask)[0]
                for mname, arr in qm.items():
                    _emit(rows, lab, mname, K, pop_name, subset, arr, sel, boot_u)
                for k in ks:
                    for mname, arr in bk[k].items():
                        _emit(rows, lab, mname, k, pop_name, subset, arr, sel, boot_u)
        print(f"[reference_ordering] {lab}: done", flush=True)

    out = out_dir / f"reference_ordering_{task}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[reference_ordering] wrote {out} ({len(rows)} rows)", flush=True)
    if cfg.figure_data_path:
        from crlo.eval.reference_figure_data import export_reference_figure_data

        export_reference_figure_data(
            pools_dir,
            Path(cfg.figure_data_path),
            cfg.expected_resolvable_queries,
            (cfg.representative_query, cfg.representative_replicate),
            ref_labels,
            cfg.figure_critic_labels,
        )
    return 0
