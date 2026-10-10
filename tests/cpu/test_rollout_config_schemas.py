"""Frozen executed-value config schemas: vocab, ranges, shard bounds, score-blindness."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from crlo.eval.config_schemas import (
    CandidateReturnsConfig,
    CandidateScoringConfig,
    ExecutedValueOrderingConfig,
    FrozenGoalRetrievalConfig,
    RolloutManifestConfig,
    ValueReliabilityConfig,
)

P = Path("/tmp/x")


def _manifests_cfg(**over):
    base = {
        "task": "antmaze-large-navigate-v0",
        "pools_dir": P,
        "val_data_path": P,
        "legacy_root": P,
        "legacy_files": ("a.csv", "b.csv"),
        "expected_exclusion_digest": "0" * 64,
        "expected_pool_sha16": "0" * 16,
        "output_dir": P,
    }
    return RolloutManifestConfig(**(base | over))


def test_manifests_config_valid_and_guards():
    cfg = _manifests_cfg()
    assert cfg.tie_seed == 7002
    with pytest.raises(ValueError, match="score-blind"):
        _manifests_cfg(pools_dir=Path("/tmp/scores_x.npz"))
    with pytest.raises(ValueError, match="sha256"):
        _manifests_cfg(expected_exclusion_digest="short")
    with pytest.raises(ValueError, match="disjoint"):
        _manifests_cfg(pilot_ranks=(450, 451), smoke_ranks=(451,))
    with pytest.raises(ValueError, match="non-empty"):
        _manifests_cfg(legacy_files=())


def _fullpool_cfg(**over):
    base = {
        "shard": 0,
        "task": "antmaze-large-navigate-v0",
        "pools_dir": P,
        "manifest_path": P,
        "diffusion_ckpt": P,
        "continuation_sha256": "0" * 64,
        "expected_pool_sha16": "0" * 16,
        "output_dir": P,
    }
    return CandidateReturnsConfig(**(base | over))


def test_fullpool_config_shard_and_seed_guards():
    cfg = _fullpool_cfg(n_shards=12, shard=11)
    assert cfg.rollout_seed == 7001 and cfg.horizon == 300
    with pytest.raises(ValueError, match="shard"):
        _fullpool_cfg(n_shards=4, shard=4)
    with pytest.raises(ValueError, match="shard"):
        _fullpool_cfg(n_shards=0, shard=0)
    with pytest.raises(ValueError, match="score-blind"):
        _fullpool_cfg(manifest_path=Path("/tmp/scores_td_q.npz"))
    with pytest.raises(ValueError, match="sha256"):
        _fullpool_cfg(continuation_sha256="abc")


def _precheck_cfg(**over):
    base = {"mode": "primary", "manifest_path": P, "merged_path": P, "output_dir": P}
    return ValueReliabilityConfig(**(base | over))


def test_precheck_config_modes_and_ranges():
    assert _precheck_cfg().h_min == 0.012
    with pytest.raises(ValueError, match="mode"):
        _precheck_cfg(mode="critic_comparison")
    with pytest.raises(ValueError, match="rollouts_per_half"):
        _precheck_cfg(rollouts_per_half=1)
    with pytest.raises(ValueError, match="h_min"):
        _precheck_cfg(h_min=0.0)


def _metrics_cfg(**over):
    base = {
        "mode": "critic_comparison",
        "manifest_path": P,
        "merged_path": P,
        "reliability_path": P,
        "scores_dir": P,
        "retrieval_csv": P,
        "resample_indices_path": P,
        "output_dir": P,
    }
    return ExecutedValueOrderingConfig(**(base | over))


def test_metrics_config_guards():
    cfg = _metrics_cfg()
    assert cfg.delta_tau == 0.10 and cfg.delta_v == 0.006
    with pytest.raises(ValueError, match="fresh seeds"):
        _metrics_cfg(mode="seed_replication", comparison_path=P)
    with pytest.raises(ValueError, match="comparison_path"):
        _metrics_cfg(mode="seed_replication", seeds=(42, 43, 44))
    with pytest.raises(ValueError, match="must include"):
        _metrics_cfg(critics=("td_q", "cosine", "hybrid"))
    with pytest.raises(ValueError, match="k_curve"):
        _metrics_cfg(k_curve=(1, 128))
    with pytest.raises(ValueError, match="k_curve"):
        _metrics_cfg(k_curve=(4, 2, 1))
    ok = _metrics_cfg(mode="seed_replication", seeds=(42, 43, 44), comparison_path=P)
    assert ok.seeds == (42, 43, 44)


def test_score_config_guards():
    base = {
        "task": "t",
        "pools_dir": P,
        "manifest_path": P,
        "checkpoint_dir": P,
        "critic": "td_q",
        "label": "td_q_s42",
        "training_seed": 42,
        "expected_pool_sha16": "0" * 16,
        "output_dir": P,
    }
    cfg = CandidateScoringConfig(**base)
    assert cfg.chunk == 8192
    with pytest.raises(ValueError, match="critic"):
        CandidateScoringConfig(**(base | {"critic": "nope"}))
    with pytest.raises(ValueError, match="label"):
        CandidateScoringConfig(**(base | {"label": ""}))


def test_retrieval_config_guards():
    base = {"mode": "build_manifest", "task": "t", "manifest_path": P, "output_dir": P}
    cfg = FrozenGoalRetrievalConfig(**base)
    assert (cfg.n_triples, cfg.kmax, cfg.seed) == (3000, 60, 0)
    with pytest.raises(ValueError, match="score mode requires"):
        FrozenGoalRetrievalConfig(**(base | {"mode": "score"}))
    ok = FrozenGoalRetrievalConfig(
        **(
            base
            | {
                "mode": "score",
                "checkpoint_dir": P,
                "critic": "cosine",
                "label": "cosine_s42",
                "training_seed": 42,
            }
        )
    )
    assert ok.critic == "cosine"
    with pytest.raises(ValueError, match="mode"):
        FrozenGoalRetrievalConfig(**(base | {"mode": "verify"}))


def test_frozen_dataclasses_are_frozen():
    cfg = _precheck_cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.h_min = 0.5  # type: ignore[misc]
