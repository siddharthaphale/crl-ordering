"""Frozen executed-value precheck driver: end-to-end synthetic Value reliability + score-blindness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from crlo.eval.config_schemas import ValueReliabilityConfig
from crlo.eval.query_manifests import manifest_payload
from crlo.eval.rollout_shards import write_shard
from crlo.eval.value_statistics import tie_priority

from .arguments import experiment_args

K, R = 8, 5
TASK = "antmaze-large-navigate-v0"


def _manifest_rows(n_per_stratum=2):
    rows = []
    q = 0
    for s in range(3):
        for j in range(n_per_stratum):
            rows.append(
                {
                    "query_id": q,
                    "stratum": s,
                    "stratum_name": ["near", "medium", "far"][s],
                    "stratum_rank": 34 + j,
                    "pool_replicate": (34 + j) % 5,
                    # episodes span strata; episode 0 contributes 3 queries
                    "episode_id": 0 if q < 3 else q,
                    "tie_priority": tie_priority(7002, q, (34 + j) % 5, K).tolist(),
                }
            )
            q += 1
    return rows


def _write_world(tmp_path, rows, returns):
    """Manifest + one shard holding all cells with the given returns."""
    meta = {
        "task": TASK,
        "manifest": "pilot",
        "pool_sha16": "p" * 16,
        "val_data_sha16": "v" * 16,
        "exclusion_manifest_sha256": "e" * 64,
        "n_replicates": 5,
        "k": K,
        "tie_seed": 7002,
    }
    payload = manifest_payload(rows, meta)
    man_path = tmp_path / "pilot_antmaze.json"
    man_path.write_text(json.dumps(payload))
    cells = {
        f: []
        for f in (
            "query_id",
            "candidate",
            "half",
            "rollout",
            "success",
            "steps_to_goal",
            "disc_return",
        )
    }
    for i, row in enumerate(rows):
        for c in range(K):
            for h in range(2):
                for r in range(R):
                    y = returns[i, c, h, r]
                    cells["query_id"].append(row["query_id"])
                    cells["candidate"].append(c)
                    cells["half"].append(h)
                    cells["rollout"].append(r)
                    cells["success"].append(int(y > 0))
                    cells["steps_to_goal"].append(5 if y > 0 else -1)
                    cells["disc_return"].append(y)
    write_shard(
        tmp_path / "rollouts",
        0,
        {k: np.asarray(v) for k, v in cells.items()},
        {"manifest_rows_sha256": payload["rows_sha256"], "pool_sha16": meta["pool_sha16"]},
    )
    return man_path, payload


def _run_precheck(tmp_path, man_path, mode):
    from experiments.check_value_reliability import main

    arguments = experiment_args(
        **{
            "mode": f"{mode}",
            "manifest_path": f"{man_path}",
            "merged_path": f"{tmp_path / 'merged' / 'full_pool_rollouts.npz'}",
            "output_dir": f"{tmp_path / 'precheck'}",
            "k": f"{K}",
            "rollouts_per_half": f"{R}",
            "bootstrap_seed": 7003,
            "n_boot": 40,
            "frac_target": 0.5,
            "agreement_target": 0.7,
            "h_min": 0.012,
            "pilot_ub_q": 0.9,
            "gate_lb_q": 0.05,
        }
    )
    assert main(arguments) == 0
    return tmp_path / "precheck"


def _strong_returns(rows, rng):
    """Rollout-resolvable world: wide candidate spread, agreeing halves."""
    n = len(rows)
    base = np.linspace(0.0, 0.9, K)
    y = np.broadcast_to(base[None, :, None, None], (n, K, 2, R)).copy()
    y += rng.normal(size=y.shape) * 1e-3
    return np.clip(y, 0.0, 1.0)


def test_reliability_pass_fixture(tmp_path):
    rows = _manifest_rows()
    rng = np.random.default_rng(1)
    man_path, _ = _write_world(tmp_path, rows, _strong_returns(rows, rng))
    out = _run_precheck(tmp_path, man_path, "primary")
    gate = json.loads((out / "reliability_antmaze.json").read_text())
    assert gate["pass"] and gate["gate"] == "GATE_A"
    assert gate["n_resolvable"] == len(rows)
    assert len(gate["resolvable_mask"]) == len(rows)
    assert (out / "query_reliability.csv").exists()
    boot = np.load(out / "bootstrap_episode_indices.npz")
    assert boot["seed_words"].tolist() == [7003, 1]
    # merged artifact was produced and matches the frozen shape
    merged = np.load(tmp_path / "merged" / "full_pool_rollouts.npz")
    assert merged["disc_return"].shape == (len(rows), K, 2, R)


def test_reliability_fail_fixture_flat_values(tmp_path):
    rows = _manifest_rows()
    man_path, _ = _write_world(tmp_path, rows, np.zeros((len(rows), K, 2, R)))
    out = _run_precheck(tmp_path, man_path, "primary")
    gate = json.loads((out / "reliability_antmaze.json").read_text())
    assert not gate["pass"]
    assert gate["n_resolvable"] == 0


def test_pilot_futility_modes(tmp_path):
    rows = _manifest_rows()
    rng = np.random.default_rng(2)
    man_path, _ = _write_world(tmp_path, rows, _strong_returns(rows, rng))
    out = _run_precheck(tmp_path, man_path, "pilot")
    fut = json.loads((out / "futility_antmaze.json").read_text())
    assert fut["decision"] == "PROCEED"
    boot = np.load(out / "bootstrap_episode_indices.npz")
    assert boot["seed_words"].tolist() == [7003, 0]  # pilot seed word


def test_pilot_flat_world_is_futile(tmp_path):
    rows = _manifest_rows()
    man_path, _ = _write_world(tmp_path, rows, np.zeros((len(rows), K, 2, R)))
    out = _run_precheck(tmp_path, man_path, "pilot")
    fut = json.loads((out / "futility_antmaze.json").read_text())
    assert fut["decision"] == "FUTILITY_STOP" and fut["auto_futility"]


def test_precheck_never_opens_a_score_file(tmp_path, monkeypatch):
    """Hard score-blindness: canary score files + np.load tripwire."""
    rows = _manifest_rows()
    rng = np.random.default_rng(3)
    man_path, _ = _write_world(tmp_path, rows, _strong_returns(rows, rng))
    # canary score files scattered through every input directory
    for d in (tmp_path, tmp_path / "rollouts"):
        with (d / "scores_td_q.npz").open("wb") as f:
            np.savez(f, scores=np.zeros((3, K)))
    real_load = np.load

    def tripwire(path, *a, **kw):
        if "scores_" in str(path):
            raise AssertionError(f"score-blind stage opened {path}")
        return real_load(path, *a, **kw)

    monkeypatch.setattr(np, "load", tripwire)
    out = _run_precheck(tmp_path, man_path, "primary")
    assert json.loads((out / "reliability_antmaze.json").read_text())["pass"]


def test_precheck_config_refuses_score_paths(tmp_path):
    with pytest.raises(ValueError, match="score-blind"):
        ValueReliabilityConfig(
            mode="primary",
            manifest_path=tmp_path / "scores_x.npz",
            merged_path=tmp_path / "m.npz",
            output_dir=tmp_path,
        )


def _det_world(tmp_path, rows, values, mismatch=False):
    """Deterministic-judge world: halves are duplicate executions."""
    import json as _json

    from crlo.eval.query_manifests import manifest_payload as _mp

    meta = {
        "task": TASK,
        "manifest": "primary",
        "pool_sha16": "p" * 16,
        "val_data_sha16": "v" * 16,
        "exclusion_manifest_sha256": "e" * 64,
        "n_replicates": 5,
        "k": K,
        "tie_seed": 7002,
    }
    payload = _mp(rows, meta)
    man_path = tmp_path / "primary_antmaze.json"
    man_path.write_text(_json.dumps(payload))
    cells = {
        f: []
        for f in (
            "query_id",
            "candidate",
            "half",
            "rollout",
            "success",
            "steps_to_goal",
            "disc_return",
        )
    }
    for i, row in enumerate(rows):
        for c in range(K):
            for h in range(2):
                y = values[i, c]
                if mismatch and h == 1 and i == 0 and c == 0:
                    y = y + 0.5
                cells["query_id"].append(row["query_id"])
                cells["candidate"].append(c)
                cells["half"].append(h)
                cells["rollout"].append(0)
                cells["success"].append(int(y > 0))
                cells["steps_to_goal"].append(5 if y > 0 else -1)
                cells["disc_return"].append(y)
    write_shard(
        tmp_path / "rollouts",
        0,
        {k: np.asarray(v) for k, v in cells.items()},
        {"manifest_rows_sha256": payload["rows_sha256"], "pool_sha16": meta["pool_sha16"]},
    )
    return man_path


def _run_det_precheck(tmp_path, man_path):
    from experiments.check_value_reliability import main

    arguments = experiment_args(
        **{
            "mode": "deterministic_primary",
            "manifest_path": f"{man_path}",
            "merged_path": f"{tmp_path / 'merged' / 'full_pool_rollouts.npz'}",
            "output_dir": f"{tmp_path / 'precheck'}",
            "k": f"{K}",
            "rollouts_per_half": 1,
            "bootstrap_seed": 7003,
            "n_boot": 25,
            "frac_target": 0.5,
            "agreement_target": 0.7,
            "h_min": 0.012,
            "pilot_ub_q": 0.9,
            "gate_lb_q": 0.05,
        }
    )
    return main(arguments)


def test_det_gate_pass_fixture(tmp_path):
    rows = _manifest_rows()
    values = np.tile(np.linspace(0.0, 0.9, K), (len(rows), 1))
    man_path = _det_world(tmp_path, rows, values)
    assert _run_det_precheck(tmp_path, man_path) == 0
    gate = json.loads((tmp_path / "precheck" / "reliability_antmaze.json").read_text())
    assert gate["pass"] and gate["gate"] == "GATE_A_DET"
    assert gate["nonconstant_fraction"] == 1.0
    boot = np.load(tmp_path / "precheck" / "bootstrap_episode_indices.npz")
    assert boot["seed_words"].tolist() == [7003, 1]  # Critic-comparison compatible


def test_det_gate_fails_on_flat_values(tmp_path):
    rows = _manifest_rows()
    man_path = _det_world(tmp_path, rows, np.zeros((len(rows), K)))
    assert _run_det_precheck(tmp_path, man_path) == 0
    gate = json.loads((tmp_path / "precheck" / "reliability_antmaze.json").read_text())
    assert not gate["pass"]


def test_det_critic_comparisonlocks_on_duplicate_mismatch(tmp_path):
    rows = _manifest_rows()
    values = np.tile(np.linspace(0.0, 0.9, K), (len(rows), 1))
    man_path = _det_world(tmp_path, rows, values, mismatch=True)
    with pytest.raises(RuntimeError, match="duplicate verification FAILED"):
        _run_det_precheck(tmp_path, man_path)
