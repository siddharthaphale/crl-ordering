"""Executed-value critic comparison: numerical fixtures and scientific guards."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from crlo.eval.query_manifests import sha256_file
from crlo.eval.value_statistics import tie_priority

from .arguments import experiment_args
from .test_reliability_score_independence import (
    TASK,
    K,
    R,
    _manifest_rows,
    _strong_returns,
    _write_world,
)

CRITICS = ("td_q", "raw_bilinear", "cosine", "hybrid")


def _run_precheck(tmp_path, man_path):
    arguments = experiment_args(
        **{
            "mode": "primary",
            "manifest_path": f"{man_path}",
            "merged_path": f"{tmp_path / 'merged' / 'full_pool_rollouts.npz'}",
            "output_dir": f"{tmp_path / 'precheck'}",
            "k": f"{K}",
            "rollouts_per_half": f"{R}",
            "bootstrap_seed": 7003,
            "n_boot": 99,
            "frac_target": 0.5,
            "agreement_target": 0.7,
            "h_min": 0.012,
            "pilot_ub_q": 0.9,
            "gate_lb_q": 0.05,
        }
    )
    from experiments.check_value_reliability import main as pre_main

    assert pre_main(arguments) == 0


def _write_scores(tmp_path, rows, v10, seeds=(42,), cached_shape=False, flat_cosine=False):
    """td_q orders perfectly; raw/cosine anti-order (or cosine constant)."""
    d = tmp_path / "scores"
    d.mkdir(exist_ok=True)
    for seed in seeds:
        tensors = {
            "td_q": v10.copy(),
            "raw_bilinear": -v10,
            "cosine": (np.zeros_like(v10) if flat_cosine else -2.0 * v10),
            "hybrid": v10 * 0.5,
        }
        for c, sc in tensors.items():
            name = (
                f"scores_{c}_s{seed}.npz" if (seed != 42 or not cached_shape) else f"scores_{c}.npz"
            )
            if cached_shape:
                # historical (Q, R, K) tensor: scatter manifest rows into a
                # bigger cube so extraction-by-(query, replicate) is exercised
                big = np.full((max(r["query_id"] for r in rows) + 1, 5, K), -9.0)
                for i, row in enumerate(rows):
                    big[row["query_id"], row["pool_replicate"]] = sc[i]
                payload = big
            else:
                payload = sc
            with (d / name).open("wb") as f:
                np.savez(
                    f,
                    scores=payload,
                    query_id=np.array([r["query_id"] for r in rows]),
                    pool_replicate=np.array([r["pool_replicate"] for r in rows]),
                )
    return d


def _write_retrieval_csv(tmp_path, seeds=(42,), auc=0.99):
    p = tmp_path / "retrieval_antmaze.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["task", "critic", "training_seed", "label", "retrieval_auc"]
        )
        w.writeheader()
        for s in seeds:
            for c in CRITICS:
                w.writerow(
                    {
                        "task": TASK,
                        "critic": c,
                        "training_seed": s,
                        "label": f"{c}_s{s}",
                        "retrieval_auc": auc,
                    }
                )
    return p


def _metrics_args(tmp_path, man_path, mode="critic_comparison", seeds=(42,), comparison_path=None):
    return experiment_args(
        **{
            "mode": f"{mode}",
            "manifest_path": f"{man_path}",
            "merged_path": f"{tmp_path / 'merged' / 'full_pool_rollouts.npz'}",
            "reliability_path": f"{tmp_path / 'precheck' / 'reliability_antmaze.json'}",
            "scores_dir": f"{tmp_path / 'scores'}",
            "retrieval_csv": f"{tmp_path / 'retrieval_antmaze.csv'}",
            "resample_indices_path": f"{tmp_path / 'precheck' / 'bootstrap_episode_indices.npz'}",
            "output_dir": f"{tmp_path / 'metrics'}",
            "comparison_path": comparison_path,
            "critics": ["td_q", "raw_bilinear", "cosine", "hybrid"],
            "seeds": seeds,
            "k": f"{K}",
            "rollouts_per_half": f"{R}",
            "k_curve": [1, 2, 4, K],
            "delta_tau": 0.1,
            "delta_v": 0.006,
            "tau_cap": 0.5,
            "auc_min": 0.95,
            "top_decile_size": 2,
        }
    )


@pytest.fixture()
def world(tmp_path):
    rows = _manifest_rows()
    rng = np.random.default_rng(7)
    returns = _strong_returns(rows, rng)
    man_path, payload = _write_world(tmp_path, rows, returns)
    _run_precheck(tmp_path, man_path)
    v10 = returns.mean(axis=(2, 3))
    return tmp_path, rows, man_path, payload, v10


def test_critic_comparison_broad_go_end_to_end(world):
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10, cached_shape=True)
    _write_retrieval_csv(tmp_path)
    from experiments.evaluate_executed_value_ordering import main

    assert main(_metrics_args(tmp_path, man_path)) == 0
    gate = json.loads((tmp_path / "metrics" / "critic_comparison_antmaze.json").read_text())
    assert gate["label"] == "BROAD_GO"
    assert set(gate["pooled"]) >= {"d_tau_raw", "d_val_cos"}
    with (tmp_path / "metrics" / "full_pool_metrics_antmaze.csv").open() as f:
        rows_csv = list(csv.DictReader(f))
    assert {r["critic"] for r in rows_csv} == set(CRITICS)
    td = next(r for r in rows_csv if r["critic"] == "td_q")
    assert float(td["mean_tau_b"]) > 0.99
    assert (tmp_path / "metrics" / "k_curves_antmaze.csv").exists()
    assert (tmp_path / "metrics" / "pairwise_contrasts_antmaze.csv").exists()


def test_critic_comparison_flat_score_rules(world):
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10, flat_cosine=True)
    _write_retrieval_csv(tmp_path)
    from experiments.evaluate_executed_value_ordering import main

    assert main(_metrics_args(tmp_path, man_path)) == 0
    with (tmp_path / "metrics" / "full_pool_metrics_antmaze.csv").open() as f:
        rows_csv = list(csv.DictReader(f))
    cos = next(r for r in rows_csv if r["critic"] == "cosine")
    assert int(cos["n_flat_score"]) == len(rows)  # flagged, not dropped
    assert float(cos["mean_tau_b"]) == 0.0  # frozen tau=0 rule


def test_critic_comparison_requires_passing_reliability(world):
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10)
    _write_retrieval_csv(tmp_path)
    reliability = tmp_path / "precheck" / "reliability_antmaze.json"
    j = json.loads(reliability.read_text())
    j["pass"] = False
    reliability.write_text(json.dumps(j))
    from experiments.evaluate_executed_value_ordering import main

    with pytest.raises(RuntimeError, match="not licensed"):
        main(_metrics_args(tmp_path, man_path))


def test_critic_comparison_rejects_foreign_resamples(world, tmp_path):
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10)
    _write_retrieval_csv(tmp_path)
    boot = tmp_path / "precheck" / "bootstrap_episode_indices.npz"
    with np.load(boot) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["seed_words"] = np.array([7003, 0])  # pilot words
    with boot.open("wb") as f:
        np.savez_compressed(f, **arrays)
    from experiments.evaluate_executed_value_ordering import main

    with pytest.raises(ValueError, match="primary resample"):
        main(_metrics_args(tmp_path, man_path))


def test_critic_comparison_nonfinite_scores_block(world):
    tmp_path, rows, man_path, _, v10 = world
    d = _write_scores(tmp_path, rows, v10)
    with np.load(d / "scores_td_q_s42.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["scores"][0, 0] = np.nan
    with (d / "scores_td_q_s42.npz").open("wb") as f:
        np.savez(f, **arrays)
    _write_retrieval_csv(tmp_path)
    from experiments.evaluate_executed_value_ordering import main

    with pytest.raises(ValueError, match="non-finite scores"):
        main(_metrics_args(tmp_path, man_path))


def test_seed_replication_confirms_and_never_upgrades(world):
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10, seeds=(42, 43, 44))
    _write_retrieval_csv(tmp_path, seeds=(42, 43, 44))
    comparison_path = tmp_path / "critic_comparison_antmaze.json"
    comparison_path.write_text(json.dumps({"label": "POINT_LIKE_GO"}))
    from experiments.evaluate_executed_value_ordering import main

    arguments = _metrics_args(
        tmp_path,
        man_path,
        mode="seed_replication",
        seeds=(42, 43, 44),
        comparison_path=comparison_path,
    )
    assert main(arguments) == 0
    gate = json.loads((tmp_path / "metrics" / "seed_replication_antmaze.json").read_text())
    assert gate["confirmed"]
    # BROAD-quality stats, but the discovery was POINT-LIKE: no upgrade.
    assert gate["label"] == "POINT_LIKE_GO"
    assert set(gate["per_seed_pass"]) == {"42", "43", "44"}
    assert (tmp_path / "metrics" / "per_seed_metrics_antmaze.csv").exists()


def test_seed_replication_blocked_without_critic_comparison_go(world):
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10, seeds=(42, 43, 44))
    _write_retrieval_csv(tmp_path, seeds=(42, 43, 44))
    comparison_path = tmp_path / "critic_comparison_antmaze.json"
    comparison_path.write_text(json.dumps({"label": "PARTIAL"}))
    from experiments.evaluate_executed_value_ordering import main

    arguments = _metrics_args(
        tmp_path,
        man_path,
        mode="seed_replication",
        seeds=(42, 43, 44),
        comparison_path=comparison_path,
    )
    with pytest.raises(RuntimeError, match="conditional on a Critic-comparison GO"):
        main(arguments)


def test_reliability_hash_binding(world):
    # Value-reliability JSON recorded against different rollouts must be rejected.
    tmp_path, rows, man_path, _, v10 = world
    _write_scores(tmp_path, rows, v10)
    _write_retrieval_csv(tmp_path)
    merged = tmp_path / "merged" / "full_pool_rollouts.npz"
    with np.load(merged) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["disc_return"] = arrays["disc_return"] * 0.5
    with merged.open("wb") as f:
        np.savez_compressed(f, **arrays)
    assert (
        sha256_file(merged)[:16]
        != json.loads((tmp_path / "precheck" / "reliability_antmaze.json").read_text())[
            "merged_sha16"
        ]
    )
    from experiments.evaluate_executed_value_ordering import main

    with pytest.raises(ValueError, match="different rollouts"):
        main(_metrics_args(tmp_path, man_path))


def test_manifest_tie_priority_reused(world):
    # The manifest's frozen priorities are what the driver consumes;
    # regenerate and compare to guard the SeedSequence contract.
    _, rows, _, _, _ = world
    for row in rows:
        want = tie_priority(7002, row["query_id"], row["pool_replicate"], K)
        assert row["tie_priority"] == want.tolist()
