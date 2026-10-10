"""Frozen executed-valueb Arm-A calibration: SB formula, R* rule, replay regression."""

from __future__ import annotations

import json

import numpy as np
import pytest

from crlo.eval.calibrate_rollout_budget import spearman_brown
from crlo.eval.config_schemas import RolloutBudgetConfig
from crlo.eval.query_manifests import manifest_payload
from crlo.eval.rollout_shards import write_merged
from crlo.eval.value_statistics import tie_priority

from .arguments import experiment_args

K, R25 = 8, 25


def test_spearman_brown_inversion():
    # rho1 = 0.1 -> rho(25) = 25*0.1/(1+24*0.1) = 0.7353
    rho25 = 25 * 0.1 / (1 + 24 * 0.1)
    assert spearman_brown(rho25, 25, 25) == pytest.approx(rho25)
    assert spearman_brown(rho25, 25, 50) == pytest.approx(50 * 0.1 / (1 + 49 * 0.1))
    assert np.isnan(spearman_brown(float("nan"), 25, 50))
    assert np.isnan(spearman_brown(-0.2, 25, 50))


def _world(tmp_path, noise_sd, seed=0):
    """Manifest + R=25 merged + R=5 baseline with known signal/noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(6):
        s = q // 2
        rows.append(
            {
                "query_id": q,
                "stratum": s,
                "stratum_name": ["near", "medium", "far"][s],
                "stratum_rank": 34 + q % 2,
                "pool_replicate": (34 + q % 2) % 5,
                "episode_id": q,
                "tie_priority": tie_priority(7002, q, 0, K).tolist(),
            }
        )
    man = manifest_payload(
        rows,
        {
            "task": "antmaze-large-navigate-v0",
            "manifest": "pilot",
            "pool_sha16": "p" * 16,
            "val_data_sha16": "v",
            "exclusion_manifest_sha256": "e",
            "n_replicates": 5,
            "k": K,
            "tie_seed": 7002,
        },
    )
    man_path = tmp_path / "pilot.json"
    man_path.write_text(json.dumps(man))
    signal = np.linspace(0, 0.8, K)[None, :, None, None]
    y = np.clip(signal + rng.normal(size=(6, K, 2, R25)) * noise_sd, 0, 1)
    tensors = {
        "disc_return": y,
        "success": (y > 0).astype(np.int64),
        "steps_to_goal": np.full(y.shape, 7, np.int64),
    }
    write_merged(tmp_path / "m25", tensors, {"n": 6})
    base = {k: v[..., :5] for k, v in tensors.items()}
    write_merged(tmp_path / "m5", base, {"n": 6})
    return man_path


def _run(tmp_path, man_path):
    from experiments.calibrate_rollout_budget import main

    arguments = experiment_args(
        **{
            "manifest_path": f"{man_path}",
            "merged_path": f"{tmp_path / 'm25' / 'full_pool_rollouts.npz'}",
            "baseline_merged_path": f"{tmp_path / 'm5' / 'full_pool_rollouts.npz'}",
            "output_dir": f"{tmp_path / 'cal'}",
            "k": f"{K}",
            "rollouts_per_half": 25,
            "r_grid": [5, 10, 15, 20, 25],
            "r_extrap": [35, 50],
            "r_choices": [25, 35, 50],
            "agreement_target": 0.7,
            "frac_target": 0.5,
        }
    )
    assert main(arguments) == 0
    return json.loads((tmp_path / "cal" / "calibration_antmaze.json").read_text())


def test_low_noise_world_is_feasible_at_25(tmp_path):
    man = _world(tmp_path, noise_sd=0.05)
    out = _run(tmp_path, man)
    assert out["decision"] == "A1_FEASIBLE" and out["r_star"] == 25
    r25 = next(g for g in out["grid"] if g["r"] == 25)
    assert r25["agreement"] > 0.9


def test_hopeless_noise_world_is_infeasible(tmp_path):
    man = _world(tmp_path, noise_sd=50.0)
    out = _run(tmp_path, man)
    assert out["decision"] == "A1_INFEASIBLE" and out["r_star"] is None


def test_replay_regression_blocks_on_mismatch(tmp_path):
    man = _world(tmp_path, noise_sd=0.05)
    base = tmp_path / "m5" / "full_pool_rollouts.npz"
    with np.load(base) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["disc_return"][0, 0, 0, 0] += 1e-9
    with base.open("wb") as f:
        np.savez_compressed(f, **arrays)
    with pytest.raises(RuntimeError, match="bit-exactly"):
        _run(tmp_path, man)


def test_calibration_config_guards(tmp_path):
    base = {
        "manifest_path": tmp_path,
        "merged_path": tmp_path,
        "baseline_merged_path": tmp_path,
        "output_dir": tmp_path,
    }
    cfg = RolloutBudgetConfig(**base)
    assert cfg.r_choices == (25, 35, 50)
    with pytest.raises(ValueError, match="end at rollouts_per_half"):
        RolloutBudgetConfig(**base, r_grid=(5, 10))
    with pytest.raises(ValueError, match="exceed rollouts_per_half"):
        RolloutBudgetConfig(**base, r_extrap=(20,))
    with pytest.raises(ValueError, match="r_choices"):
        RolloutBudgetConfig(**base, r_choices=(30,))
    with pytest.raises(ValueError, match="score-blind"):
        RolloutBudgetConfig(**(base | {"merged_path": tmp_path / "scores_x.npz"}))


def test_action_value_smoothness_helper():
    from crlo.eval.value_statistics import action_value_smoothness

    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 8))
    v_smooth = a @ rng.normal(size=8)  # linear in the action
    rho, ratio = action_value_smoothness(a, v_smooth)
    # even a perfectly linear value gives rho ~0.28 in 8-D (direction
    # matters, not just distance) — the geometric ceiling of the statistic
    assert rho > 0.25 and ratio < 0.9
    v_chaotic = rng.normal(size=64)
    rho2, ratio2 = action_value_smoothness(a, v_chaotic)
    assert abs(rho2) < 0.2 and ratio2 > 0.8


def test_action_persistence_config_guards(tmp_path):
    from crlo.eval.config_schemas import ActionPersistenceConfig

    base = {
        "manifest_path": tmp_path,
        "pools_dir": tmp_path,
        "merged_root": tmp_path,
        "output_dir": tmp_path,
    }
    cfg = ActionPersistenceConfig(**base)
    assert cfg.m_grid == (5, 10, 20) and cfg.smoothness_target == 0.15
    with pytest.raises(ValueError, match="m_grid"):
        ActionPersistenceConfig(**base, m_grid=(1, 5))
    with pytest.raises(ValueError, match="m_grid"):
        ActionPersistenceConfig(**base, m_grid=(10, 5))
    with pytest.raises(ValueError, match="score-blind"):
        ActionPersistenceConfig(**(base | {"merged_root": tmp_path / "scores_x.npz"}))
