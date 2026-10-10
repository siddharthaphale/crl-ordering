"""Frozen executed-value frozen statistical conventions (plan §10.2)."""

from __future__ import annotations

import numpy as np
import pytest

from crlo.eval.value_statistics import (
    assess_value_reliability,
    average_ranks,
    bootstrap_equal_stratum,
    candidate_spread,
    classify_critic_comparison,
    classify_seed_replication,
    crossfit_winner,
    draw_episode_resamples,
    equal_stratum_mean,
    half_mean_se,
    holm_adjust,
    kendall_tau_b,
    lower_bound,
    normalized_ratios,
    null_centered_p,
    paired_noise,
    pilot_futility,
    query_reliability,
    resampled_query_indices,
    spearman_rho,
    tie_break_argmax,
    tie_priority,
    top_set,
    upper_bound,
)

RNG = np.random.default_rng(0)


# --- rank statistics -------------------------------------------------------


def test_average_ranks_ties():
    assert np.allclose(average_ranks(np.array([1.0, 2.0, 2.0, 3.0])), [1.0, 2.5, 2.5, 4.0])


def test_spearman_matches_scipy_with_ties():
    from scipy.stats import spearmanr

    x = np.array([1.0, 2.0, 2.0, 3.0, 0.5, 2.0])
    y = np.array([0.1, 0.4, 0.3, 0.9, 0.0, 0.4])
    assert spearman_rho(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)


def test_spearman_constant_is_nan_not_dropped():
    assert np.isnan(spearman_rho(np.ones(5), np.arange(5.0)))
    assert np.isnan(spearman_rho(np.arange(5.0), np.zeros(5)))


def test_kendall_tau_b_constant_nan():
    assert np.isnan(kendall_tau_b(np.ones(4), np.arange(4.0)))
    assert kendall_tau_b(np.arange(4.0), np.arange(4.0)) == pytest.approx(1.0)


# --- split-half machinery (frozen ddof / quantile conventions) -------------


def test_half_mean_se_ddof1():
    y = RNG.normal(size=(8, 5))
    v, se = half_mean_se(y)
    assert np.allclose(v, y.mean(-1))
    assert np.allclose(se, y.std(-1, ddof=1) / np.sqrt(5))


def test_candidate_spread_linear_quantiles():
    v = np.arange(64.0)
    want = np.quantile(v, 0.9, method="linear") - np.quantile(v, 0.1, method="linear")
    assert candidate_spread(v) == pytest.approx(want)


def test_paired_noise_cancels_common_randomness():
    # Common random numbers: shared noise cancels in Y_i - Y_j, so the
    # paired estimate must sit far below the independent per-candidate SE.
    common = RNG.normal(size=(1, 5)) * 1.0
    idio = RNG.normal(size=(16, 5)) * 0.01
    y = common + idio
    _, se = half_mean_se(y)
    assert paired_noise(y) < 0.2 * se.mean()


def test_query_reliability_pass_and_fail_fixtures():
    # Strong signal: candidate means far apart, halves agree.
    base = np.linspace(0.0, 0.9, 64)[:, None]
    y_a = base + RNG.normal(size=(64, 5)) * 1e-3
    y_b = base + RNG.normal(size=(64, 5)) * 1e-3
    rel = query_reliability(y_a, y_b)
    assert rel.resolvable and rel.rho > 0.99
    # Pure noise: spread indistinguishable from paired noise -> fail.
    y_a = RNG.normal(size=(64, 5))
    y_b = RNG.normal(size=(64, 5))
    rel = query_reliability(y_a, y_b)
    assert not rel.resolvable


def test_query_reliability_constant_halves_fail_not_drop():
    y = np.zeros((64, 5))
    rel = query_reliability(y, y)
    assert not rel.resolvable
    assert np.isnan(rel.rho)
    assert rel.flat_value


# --- tie priorities + cross-fit (plan §6.2) --------------------------------


def test_tie_priority_frozen_and_permutation():
    p1 = tie_priority(7002, 34, 4, 64)
    p2 = tie_priority(7002, 34, 4, 64)
    assert np.array_equal(p1, p2)
    assert sorted(p1.tolist()) == list(range(64))
    assert not np.array_equal(p1, tie_priority(7002, 35, 4, 64))
    assert not np.array_equal(p1, tie_priority(7002, 34, 3, 64))


def test_tie_break_argmax_uses_priority_not_index():
    values = np.array([1.0, 2.0, 2.0, 0.0])
    prio = np.array([0, 3, 1, 2])
    assert tie_break_argmax(values, prio) == 2  # not the earlier index 1


def test_top_set_priority_break():
    values = np.array([5.0, 1.0, 1.0, 1.0])
    prio = np.array([0, 3, 1, 2])
    assert top_set(values, 2, prio).tolist() == [0, 2]


def test_crossfit_winner_no_same_sample_evaluation():
    # Candidate 0 has a lucky half-A draw; held-out value is mediocre.
    v_a = np.array([0.9, 0.5, 0.1])
    v_b = np.array([0.1, 0.5, 0.2])
    prio = np.array([0, 1, 2])
    cf = crossfit_winner(v_a, v_b, prio)
    assert cf.idx_a == 0 and cf.idx_b == 1
    # W_CF = (v_b[0] + v_a[1]) / 2, never v_a[idx_a].
    assert cf.w_cf == pytest.approx(0.5 * (0.1 + 0.5))
    plug_in_max = 0.5 * (v_a.max() + v_b.max())
    assert cf.w_cf < plug_in_max


def test_crossfit_negative_winner_minus_critic_gap_permitted():
    v_a = np.array([0.0, 0.4])
    v_b = np.array([0.4, 0.0])
    prio = np.array([0, 1])
    cf = crossfit_winner(v_a, v_b, prio)
    critic_selected = 0.4  # critic happens to pick well
    assert cf.w_cf - critic_selected < 0.0  # gap may be negative


# --- equal-stratum weighting ------------------------------------------------


def test_equal_stratum_mean_weighting():
    values = np.array([1.0, 1.0, 0.0, 2.0])
    strata = np.array([0, 0, 1, 2])
    # (mean 1.0 + mean 0.0 + mean 2.0) / 3, not the pooled mean.
    assert equal_stratum_mean(values, strata) == pytest.approx(1.0)


def test_equal_stratum_mean_empty_stratum_nan():
    values = np.array([1.0, 1.0])
    strata = np.array([0, 1])
    assert np.isnan(equal_stratum_mean(values, strata))
    valid = np.array([True, False])
    strata3 = np.array([0, 1])
    assert np.isnan(equal_stratum_mean(values, strata3, valid))


# --- frozen episode-cluster bootstrap ----------------------------------------


def _cluster_fixture():
    # 6 episodes, 12 queries; episodes span strata; some contribute 3 queries.
    episode = np.array([0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5])
    strata = np.array([0, 0, 1, 0, 1, 1, 2, 2, 2, 0, 1, 2])
    return episode, strata


def test_resamples_reproducible_and_strata_covered():
    episode, strata = _cluster_fixture()
    r1 = draw_episode_resamples(episode, strata, 50, [7003, 1])
    r2 = draw_episode_resamples(episode, strata, 50, [7003, 1])
    assert np.array_equal(r1, r2)
    assert not np.array_equal(r1, draw_episode_resamples(episode, strata, 50, [7003, 0]))
    ep_sorted = np.unique(episode)
    for row in r1:
        qi = resampled_query_indices(row, episode)
        assert set(strata[qi].tolist()) == {0, 1, 2}
        assert set(ep_sorted[row]).issubset(set(ep_sorted))


def test_resample_carries_whole_episodes():
    # The dependence fixture: an episode with 3 queries must always be
    # carried whole (query-level resampling would split it).
    episode, strata = _cluster_fixture()
    r = draw_episode_resamples(episode, strata, 20, [7003, 1])
    for row in r:
        qi = resampled_query_indices(row, episode)
        counts: dict[int, int] = {}
        for q in qi:
            counts[q] = counts.get(q, 0) + 1
        # every query of episode 0 appears the same number of times
        ep0 = np.flatnonzero(episode == 0)
        appearances = {counts.get(int(q), 0) for q in ep0}
        assert len(appearances) == 1


def test_bootstrap_conditional_nan_handling():
    episode, strata = _cluster_fixture()
    values = np.linspace(0, 1, len(episode))
    valid = strata != 2  # stratum 2 never valid
    r = draw_episode_resamples(episode, strata, 10, [7003, 0])
    boot = bootstrap_equal_stratum(values, strata, episode, r, valid=valid)
    assert np.isnan(boot).all()
    assert upper_bound(boot, 0.9) == -np.inf
    with pytest.raises(ValueError, match="never silently discarded"):
        lower_bound(boot, 0.05)


def test_null_centered_p_and_holm():
    boot = np.array([0.5, 1.5, 2.5, -0.5, 1.0])
    # |b - 1| >= 1 for b in {2.5, -0.5} -> (1+2)/(5+1)
    assert null_centered_p(1.0, boot) == pytest.approx(3 / 6)
    p = holm_adjust(np.array([0.01, 0.04, 0.03, 0.005]))
    assert np.allclose(p, [0.03, 0.06, 0.06, 0.02])


# --- gate decisions -----------------------------------------------------------


def test_pilot_futility_fixtures():
    ok = pilot_futility(0.6, 0.8, 0.02, 10, 10, 0.5, 0.7, 0.012)
    assert ok["decision"] == "PROCEED"
    bad = pilot_futility(0.4, 0.8, 0.02, 10, 10, 0.5, 0.7, 0.012)
    assert bad["decision"] == "FUTILITY_STOP"
    assert bad["failed_targets"] == ["resolvable_fraction_ub"]
    auto = pilot_futility(0.6, 0.8, 0.02, 0, 10, 0.5, 0.7, 0.012)
    assert auto["decision"] == "FUTILITY_STOP" and auto["auto_futility"]
    few = pilot_futility(0.6, 0.8, 0.02, 5, 1, 0.5, 0.7, 0.012)
    assert few["decision"] == "FUTILITY_STOP"
    undef = pilot_futility(0.6, float("nan"), 0.02, 5, 5, 0.5, 0.7, 0.012)
    assert undef["decision"] == "FUTILITY_STOP"


def test_reliability_fixtures():
    good = assess_value_reliability(0.6, 180, 300, 0.8, 0.02, 0.005, 0.5, 0.7, 0.012)
    assert good["pass"]
    for kwargs in (
        {"resolvable_fraction": 0.4},
        {"agreement": 0.6},
        {"headroom": 0.005},
        {"headroom_lb": -0.001},
        {"agreement": float("nan")},
    ):
        base = {
            "resolvable_fraction": 0.6,
            "n_resolvable": 180,
            "n_queries": 300,
            "agreement": 0.8,
            "headroom": 0.02,
            "headroom_lb": 0.005,
            "frac_target": 0.5,
            "agreement_target": 0.7,
            "h_min": 0.012,
        }
        assert not assess_value_reliability(**(base | kwargs))["pass"]


def _label(**over):
    base = {
        "auc_raw": 0.99,
        "auc_cosine": 0.99,
        "auc_min": 0.95,
        "tau_raw": 0.1,
        "tau_cosine": 0.1,
        "tau_cap": 0.5,
        "d_tau_raw": 0.3,
        "d_tau_cos": 0.3,
        "delta_tau": 0.10,
        "d_val_raw": 0.01,
        "d_val_cos": 0.01,
        "delta_v": 0.006,
        "holm_p": {k: 0.001 for k in ("tau_raw", "tau_cos", "val_raw", "val_cos")},
        "ci_lo": {k: 0.001 for k in ("tau_raw", "tau_cos", "val_raw", "val_cos")},
    }
    return classify_critic_comparison(**(base | over))


def test_critic_comparison_broad_go():
    assert _label()["label"] == "BROAD_GO"


def test_critic_comparison_point_like_when_raw_value_tie():
    out = _label(
        d_val_raw=0.0,
        holm_p={"tau_raw": 0.001, "tau_cos": 0.001, "val_raw": 0.9, "val_cos": 0.001},
        ci_lo={"tau_raw": 0.01, "tau_cos": 0.01, "val_raw": -0.01, "val_cos": 0.01},
    )
    assert out["label"] == "POINT_LIKE_GO"


def test_critic_comparison_partial_and_no_go():
    out = _label(
        d_tau_cos=0.05,
        holm_p={"tau_raw": 0.001, "tau_cos": 0.9, "val_raw": 0.9, "val_cos": 0.9},
        ci_lo={"tau_raw": 0.01, "tau_cos": -0.1, "val_raw": -0.1, "val_cos": -0.1},
        d_val_raw=0.0,
        d_val_cos=0.0,
    )
    assert out["label"] == "PARTIAL"
    out = _label(auc_raw=0.90)
    assert out["label"] == "NO_GO"  # retrieval gate failure
    out = _label(d_tau_raw=0.05, d_tau_cos=0.05, d_val_raw=0.001, d_val_cos=0.001)
    assert out["label"] == "NO_GO"


def test_critic_comparison_no_tau_cap_on_tdq():
    # TD-Q ordering extremely well must not disqualify the GO: the cap
    # applies to raw and cosine only (plan §7.3 as amended).
    out = _label()  # no TD-Q tau argument exists at all
    assert out["label"] == "BROAD_GO"
    out = _label(tau_raw=0.6)
    assert out["label"] in ("PARTIAL", "NO_GO")


def test_seed_replication_never_upgrades():
    assert classify_seed_replication("POINT_LIKE_GO", "BROAD_GO") == "POINT_LIKE_GO"
    assert classify_seed_replication("BROAD_GO", "POINT_LIKE_GO") == "POINT_LIKE_GO"
    assert classify_seed_replication("BROAD_GO", "BROAD_GO") == "BROAD_GO"


def test_normalized_ratios_aggregate_not_per_query():
    w = np.array([0.5, 0.1])
    f = np.array([0.4, 0.05])
    r = np.array([0.1, 0.05])
    out = normalized_ratios(w, f, r)
    wm, fm, rm = w.mean(), f.mean(), r.mean()
    assert out["captured"] == pytest.approx((fm - rm) / (wm - rm))
    flat = normalized_ratios(np.array([0.1]), np.array([0.1]), np.array([0.2]))
    assert flat["normalized_winner_gap"] is None
    assert flat["n_nonpositive_headroom"] == 1
