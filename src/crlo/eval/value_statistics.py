"""Frozen executed-value statistical conventions.

Pure-numpy/scipy implementations of every preregistered Frozen executed-value
estimator so unit tests can exercise them without a driver, an
environment, or JAX:

- split-half candidate values, paired Monte Carlo noise, and the
  rollout-resolvability rule (plan §6.1);
- deterministic tie priorities and the cross-fitted empirical-winner
  headroom (plan §6.2);
- the frozen episode-cluster bootstrap, null-centered p-values, and
  Holm adjustment (plan §6.4);
- Value-reliability / pilot-futility decisions (plan §5.3, §6.3) and the
  Comparison/replication outcome-label precedence (plan §7.3, §8.3).

Numerical conventions frozen by the plan: ``numpy.quantile`` with
``method="linear"``, ``ddof=1`` standard deviations, Spearman as
average ranks followed by Pearson correlation of ranks, and Kendall as
the tie-corrected tau-b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

QUANTILE_METHOD = "linear"
STRATA = (0, 1, 2)


# ---------------------------------------------------------------------------
# Rank statistics
# ---------------------------------------------------------------------------


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties assigned the average rank."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Average-rank Spearman: Pearson correlation of tie-averaged ranks.

    Returns ``nan`` when either input is constant (undefined correlation);
    the caller decides how an undefined value propagates (plan §6.1: a
    constant/tied half is a resolvability failure, never a dropped row).
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    rx, ry = average_ranks(x), average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-corrected Kendall tau-b; ``nan`` when either input is constant."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(kendalltau(x, y).statistic)


# ---------------------------------------------------------------------------
# Split-half values and resolvability (plan §6.1)
# ---------------------------------------------------------------------------


def half_mean_se(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate mean and ddof=1 SE over rollouts. ``y``: (K, R)."""
    y = np.asarray(y, np.float64)
    r = y.shape[-1]
    return y.mean(-1), y.std(-1, ddof=1) / np.sqrt(r)


def candidate_spread(vhat: np.ndarray) -> float:
    """P90 - P10 of per-candidate half means (linear quantiles)."""
    lo, hi = np.quantile(np.asarray(vhat, np.float64), [0.10, 0.90], method=QUANTILE_METHOD)
    return float(hi - lo)


def paired_noise(y: np.ndarray) -> float:
    """Median over candidate pairs i<j of sd(Y_i - Y_j, ddof=1)/sqrt(R).

    The paired construction preserves the common-random-numbers design
    (plan §6.1); an independent-candidate SE would describe a different
    experiment. ``y``: (K, R).
    """
    y = np.asarray(y, np.float64)
    k, r = y.shape
    diffs = y[:, None, :] - y[None, :, :]  # (K, K, R)
    iu = np.triu_indices(k, 1)
    sd = diffs[iu].std(-1, ddof=1)  # (K*(K-1)/2,)
    return float(np.median(sd) / np.sqrt(r))


@dataclass(frozen=True)
class QueryReliability:
    """Per-query split-half reliability summary (plan §6.1)."""

    spread_a: float
    spread_b: float
    noise_a: float
    noise_b: float
    rho: float
    tau_b: float
    flat_value: bool
    resolvable: bool


def query_reliability(y_a: np.ndarray, y_b: np.ndarray) -> QueryReliability:
    """Apply the frozen §6.1 resolvability rule to one query's halves.

    ``y_a``, ``y_b``: (K, R) per-half return draws. Resolvable requires
    S_h > 2 N_h in both halves AND a finite split-half Spearman.
    """
    va, _ = half_mean_se(y_a)
    vb, _ = half_mean_se(y_b)
    sa, sb = candidate_spread(va), candidate_spread(vb)
    na, nb = paired_noise(y_a), paired_noise(y_b)
    rho = spearman_rho(va, vb)
    tau = kendall_tau_b(va, vb)
    v10 = 0.5 * (va + vb)
    flat_value = bool(np.ptp(v10) == 0.0)
    ok = bool(sa > 2.0 * na and sb > 2.0 * nb and np.isfinite(rho))
    return QueryReliability(sa, sb, na, nb, rho, tau, flat_value, ok)


# ---------------------------------------------------------------------------
# Tie priorities, selection, cross-fitted winner (plan §6.2)
# ---------------------------------------------------------------------------


def tie_priority(tie_seed: int, query_id: int, pool_replicate: int, k: int) -> np.ndarray:
    """Frozen random priority for one query--pool pair.

    Returns ``prio`` with ``prio[c]`` = preference rank of candidate ``c``
    (lower preferred). Every winner/argmax/top-set operation breaks value
    ties by minimal ``prio``.
    """
    perm = np.random.default_rng(
        np.random.SeedSequence([tie_seed, query_id, pool_replicate])
    ).permutation(k)
    prio = np.empty(k, np.int64)
    prio[perm] = np.arange(k)
    return prio


def tie_break_argmax(values: np.ndarray, prio: np.ndarray) -> int:
    """Index of the max value; ties broken by minimal frozen priority."""
    values = np.asarray(values, np.float64)
    top = np.flatnonzero(values == values.max())
    return int(top[np.argmin(prio[top])])


def top_set(values: np.ndarray, m: int, prio: np.ndarray) -> np.ndarray:
    """Top-``m`` candidate indices by (value desc, frozen priority asc)."""
    order = np.lexsort((prio, -np.asarray(values, np.float64)))
    return np.sort(order[:m])


@dataclass(frozen=True)
class CrossFitWinner:
    """Cross-fitted empirical-winner quantities for one query (plan §6.2)."""

    idx_a: int
    idx_b: int
    w_cf: float
    pool_mean: float
    headroom: float


def crossfit_winner(v_a: np.ndarray, v_b: np.ndarray, prio: np.ndarray) -> CrossFitWinner:
    """Symmetric cross-fit of the five-draw empirical-winner rule.

    The winner is selected in one half and valued in the other, so the
    reported value is held out; the baseline is the enumerated full-pool
    mean (no extra random-action draw).
    """
    ia = tie_break_argmax(v_a, prio)
    ib = tie_break_argmax(v_b, prio)
    w_cf = 0.5 * (float(v_b[ia]) + float(v_a[ib]))
    pool_mean = 0.5 * (float(np.mean(v_a)) + float(np.mean(v_b)))
    return CrossFitWinner(ia, ib, w_cf, pool_mean, w_cf - pool_mean)


# ---------------------------------------------------------------------------
# Equal-stratum aggregation
# ---------------------------------------------------------------------------


def equal_stratum_mean(
    values: np.ndarray, strata: np.ndarray, valid: np.ndarray | None = None
) -> float:
    """Mean of the three per-stratum arithmetic means.

    ``valid`` restricts to a conditional subpopulation (e.g. resolvable
    queries). A required stratum with no valid entry makes the statistic
    undefined (``nan``) rather than silently reweighted (plan §6.1).
    """
    values = np.asarray(values, np.float64)
    strata = np.asarray(strata)
    if valid is None:
        valid = np.ones(len(values), bool)
    out = []
    for s in STRATA:
        m = (strata == s) & valid
        if not m.any():
            return float("nan")
        out.append(values[m].mean())
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# Frozen episode-cluster bootstrap (plan §6.4)
# ---------------------------------------------------------------------------


def draw_episode_resamples(
    episode_of_query: np.ndarray, stratum_of_query: np.ndarray, n_boot: int, seed_words: list[int]
) -> np.ndarray:
    """Draw the frozen (B, n_ep) matrix of episode-position resamples.

    Positions index the sorted unique episode-ID list. Rows are drawn
    sequentially from ``default_rng(SeedSequence(seed_words))``; a row
    whose carried queries miss any stratum is redrawn (continuing the
    stream) before use, per plan §6.4.
    """
    episodes = np.unique(np.asarray(episode_of_query))
    n_ep = len(episodes)
    ep_strata = [np.unique(stratum_of_query[episode_of_query == e]) for e in episodes]
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    out = np.empty((n_boot, n_ep), np.int64)
    for b in range(n_boot):
        while True:
            row = rng.integers(0, n_ep, size=n_ep)
            covered = set()
            for p in row:
                covered.update(ep_strata[p].tolist())
            if covered >= set(STRATA):
                out[b] = row
                break
    return out


def resampled_query_indices(resample_row: np.ndarray, episode_of_query: np.ndarray) -> np.ndarray:
    """Query indices (with multiplicity) carried by one episode resample."""
    episodes = np.unique(episode_of_query)
    picked = episodes[np.asarray(resample_row)]
    parts = [np.flatnonzero(episode_of_query == e) for e in picked]
    return np.concatenate(parts) if parts else np.empty(0, np.int64)


def bootstrap_equal_stratum(
    values: np.ndarray,
    strata: np.ndarray,
    episode_of_query: np.ndarray,
    resamples: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Equal-stratum statistic on every frozen resample.

    Returns a (B,) vector; replicates whose conditional statistic is
    undefined are ``nan`` (mapped by the caller per plan §6.4: ``-inf``
    for futility upper bounds, hard failure for required lower bounds).
    """
    values = np.asarray(values, np.float64)
    out = np.empty(len(resamples), np.float64)
    for b, row in enumerate(resamples):
        qi = resampled_query_indices(row, episode_of_query)
        v = valid[qi] if valid is not None else None
        out[b] = equal_stratum_mean(values[qi], np.asarray(strata)[qi], v)
    return out


def upper_bound(boot: np.ndarray, q: float) -> float:
    """One-sided upper bound: the ``q`` quantile; nan replicates -> -inf."""
    vals = np.where(np.isfinite(boot), boot, -np.inf)
    if not np.isfinite(vals).any():
        return float("-inf")
    out = np.quantile(vals, q, method=QUANTILE_METHOD)
    if np.isnan(out):
        # linear interpolation between two -inf order statistics
        return float("-inf")
    return float(out)


def lower_bound(boot: np.ndarray, q: float) -> float:
    """One-sided lower bound: the ``q`` quantile; nan replicates fail hard."""
    boot = np.asarray(boot, np.float64)
    if np.isnan(boot).any():
        raise ValueError(
            "undefined bootstrap replicate in a required lower-bound "
            "calculation (plan §6.4: never silently discarded)"
        )
    return float(np.quantile(boot, q, method=QUANTILE_METHOD))


def null_centered_p(theta_hat: float, boot: np.ndarray) -> float:
    """Null-centered finite-sample-corrected two-sided bootstrap p-value."""
    boot = np.asarray(boot, np.float64)
    if np.isnan(boot).any() or not np.isfinite(theta_hat):
        raise ValueError("undefined replicate/estimate in a required test")
    n_extreme = int(np.sum(np.abs(boot - theta_hat) >= abs(theta_hat)))
    return float((1 + n_extreme) / (len(boot) + 1))


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Gate decisions (plan §5.3, §6.3, §7.3, §8.3)
# ---------------------------------------------------------------------------


def pilot_futility(
    resolvable_ub: float,
    agreement_ub: float,
    headroom_ub: float,
    n_resolvable: int,
    n_finite_rho: int,
    frac_target: float,
    agreement_target: float,
    h_min: float,
) -> dict:
    """Frozen three-part futility rule (plan §5.3). Returns the decision."""
    auto_futile = n_resolvable == 0 or n_finite_rho < 2
    checks = {
        "resolvable_fraction_ub": (resolvable_ub, frac_target),
        "agreement_ub": (agreement_ub, agreement_target),
        "headroom_ub": (headroom_ub, h_min),
    }
    failed = [k for k, (v, t) in checks.items() if not np.isfinite(v) or v < t]
    futile = auto_futile or bool(failed)
    return {
        "decision": "FUTILITY_STOP" if futile else "PROCEED",
        "auto_futility": bool(auto_futile),
        "failed_targets": failed,
        "n_resolvable": int(n_resolvable),
        "n_finite_rho": int(n_finite_rho),
        **{k: float(v) for k, (v, _) in checks.items()},
        "targets": {k: float(t) for k, (_, t) in checks.items()},
    }


def assess_value_reliability(
    resolvable_fraction: float,
    n_resolvable: int,
    n_queries: int,
    agreement: float,
    headroom: float,
    headroom_lb: float,
    frac_target: float,
    agreement_target: float,
    h_min: float,
) -> dict:
    """Frozen Value-reliability conjunction (plan §6.3). All four must hold."""
    checks = {
        "resolvable_fraction": bool(
            np.isfinite(resolvable_fraction) and resolvable_fraction >= frac_target
        ),
        "agreement": bool(np.isfinite(agreement) and agreement >= agreement_target),
        "headroom": bool(np.isfinite(headroom) and headroom >= h_min),
        "headroom_lb_positive": bool(np.isfinite(headroom_lb) and headroom_lb > 0.0),
    }
    return {
        "gate": "GATE_A",
        "pass": all(checks.values()),
        "checks": checks,
        "resolvable_fraction": float(resolvable_fraction),
        "n_resolvable": int(n_resolvable),
        "n_queries": int(n_queries),
        "agreement": float(agreement),
        "headroom": float(headroom),
        "headroom_lb": float(headroom_lb),
        "targets": {
            "resolvable_fraction": frac_target,
            "agreement": agreement_target,
            "h_min": h_min,
        },
    }


def classify_critic_comparison(
    auc_raw: float,
    auc_cosine: float,
    auc_min: float,
    tau_raw: float,
    tau_cosine: float,
    tau_cap: float,
    d_tau_raw: float,
    d_tau_cos: float,
    delta_tau: float,
    d_val_raw: float,
    d_val_cos: float,
    delta_v: float,
    holm_p: dict[str, float],
    ci_lo: dict[str, float],
) -> dict:
    """Frozen Critic-comparison outcome-label precedence (plan §7.3).

    ``holm_p`` / ``ci_lo`` are keyed by the four frozen effects:
    ``tau_raw``, ``tau_cos``, ``val_raw``, ``val_cos``. The tau cap
    applies to raw and cosine only (no cap on TD-Q or hybrid).
    """
    retrieval_ok = auc_raw >= auc_min and auc_cosine >= auc_min
    caps_ok = tau_raw <= tau_cap and tau_cosine <= tau_cap

    def sig(key: str) -> bool:
        return holm_p[key] < 0.05 and ci_lo[key] > 0.0

    broad = (
        retrieval_ok
        and caps_ok
        and d_tau_raw >= delta_tau
        and d_tau_cos >= delta_tau
        and d_val_raw >= delta_v
        and d_val_cos >= delta_v
        and all(sig(k) for k in ("tau_raw", "tau_cos", "val_raw", "val_cos"))
    )
    point_like = (
        not broad
        and retrieval_ok
        and caps_ok
        and d_tau_raw >= delta_tau
        and d_tau_cos >= delta_tau
        and d_val_cos >= delta_v
        and d_val_raw >= 0.0
        and all(sig(k) for k in ("tau_raw", "tau_cos", "val_cos"))
    )
    directional = {
        "tau_raw": d_tau_raw >= delta_tau and sig("tau_raw"),
        "tau_cos": d_tau_cos >= delta_tau and sig("tau_cos"),
        "val_raw": d_val_raw >= delta_v and sig("val_raw"),
        "val_cos": d_val_cos >= delta_v and sig("val_cos"),
    }
    partial = not broad and not point_like and retrieval_ok and any(directional.values())
    if broad:
        label = "BROAD_GO"
    elif point_like:
        label = "POINT_LIKE_GO"
    elif partial:
        label = "PARTIAL"
    else:
        label = "NO_GO"
    return {
        "label": label,
        "retrieval_ok": bool(retrieval_ok),
        "caps_ok": bool(caps_ok),
        "directional": {k: bool(v) for k, v in directional.items()},
        "effects": {
            "d_tau_raw": d_tau_raw,
            "d_tau_cos": d_tau_cos,
            "d_val_raw": d_val_raw,
            "d_val_cos": d_val_cos,
        },
        "holm_p": dict(holm_p),
        "ci_lo": dict(ci_lo),
    }


def classify_seed_replication(critic_comparison: str, candidate: str) -> str:
    """Never upgrade beyond the Critic-comparison discovery label (plan §8.3)."""
    rank = {"NO_GO": 0, "PARTIAL": 1, "POINT_LIKE_GO": 2, "BROAD_GO": 3}
    if rank[candidate] > rank[critic_comparison]:
        return critic_comparison
    return candidate


# ---------------------------------------------------------------------------
# Action-space smoothness (plan §18 calibration; exploratory in 7b)
# ---------------------------------------------------------------------------


def action_value_smoothness(actions: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Distance-value Spearman + near-pair |dV| ratio for one query.

    Smooth (learnable) value structure gives a clearly positive Spearman
    and a near-pair ratio well below 1; chaotic structure gives ~0 / ~1.
    """
    k = len(values)
    iu = np.triu_indices(k, 1)
    dv = np.abs(np.subtract.outer(values, values))[iu]
    da = np.linalg.norm(actions[:, None, :] - actions[None, :, :], axis=-1)[iu]
    rho = spearman_rho(da, dv)
    q25 = np.quantile(da, 0.25, method=QUANTILE_METHOD)
    all_mean = float(dv.mean())
    near = dv[da <= q25]
    ratio = float(near.mean()) / all_mean if all_mean > 0 and len(near) else float("nan")
    return rho, ratio


# ---------------------------------------------------------------------------
# Normalized reporting ratios (plan §7.2)
# ---------------------------------------------------------------------------


def normalized_ratios(v_winner_cf: np.ndarray, v_selected: np.ndarray, v_rand: np.ndarray) -> dict:
    """Ratio-of-aggregate-means winner gap and captured headroom.

    Never averages per-query ratios; ``NA`` (None) when the population
    denominator is nonpositive, with diagnostics (plan §7.2).
    """
    wm = float(np.mean(v_winner_cf))
    fm = float(np.mean(v_selected))
    rm = float(np.mean(v_rand))
    denom = wm - rm
    n_nonpos = int(np.sum((np.asarray(v_winner_cf) - np.asarray(v_rand)) <= 0.0))
    if denom <= 0.0:
        return {
            "normalized_winner_gap": None,
            "captured": None,
            "denominator": denom,
            "n_nonpositive_headroom": n_nonpos,
        }
    return {
        "normalized_winner_gap": (wm - fm) / denom,
        "captured": (fm - rm) / denom,
        "denominator": denom,
        "n_nonpositive_headroom": n_nonpos,
    }
