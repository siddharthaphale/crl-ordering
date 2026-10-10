"""Frozen executed-value retrieval manifest: build/load replay + seed-0 canonical regression."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("scipy")


class FakeDS:
    """Minimal stand-in exposing the fields the canonical sampler reads."""

    def __init__(self, n_steps=400, ep_len=40, obs_dim=6, act_dim=3, seed=0):
        rng = np.random.default_rng(seed)
        self.dataset = {
            "observations": rng.normal(size=(n_steps, obs_dim)).astype(np.float32),
            "actions": rng.normal(size=(n_steps, act_dim)).astype(np.float32),
        }
        self.terminal_locs = np.arange(ep_len - 1, n_steps, ep_len)


def _cfg(tmp_path, mode="build_manifest", **over):
    from crlo.eval.config_schemas import FrozenGoalRetrievalConfig

    base = {
        "mode": mode,
        "task": "antmaze-large-navigate-v0",
        "manifest_path": tmp_path / "retrieval_manifest_antmaze.npz",
        "output_dir": tmp_path,
        "n_triples": 50,
        "kmax": 10,
        "seed": 0,
    }
    return FrozenGoalRetrievalConfig(**(base | over))


@pytest.mark.parametrize(
    ("seed", "ep_len", "kmax", "expected_digest"),
    [
        (0, 40, 10, "e93ba66cd1a380f732d65edf039f2878c795bf108ba6644045695e3e31a802b0"),
        (42, 40, 60, "f36ca51f1ed53f5b02d4e28eb8dba60ba0d862bbabb8767605d3e52659b465f6"),
        (44, 3, 10, "672a53a34c987dd83dab4550f6537b7994e9a7b9fe4819c127dc87c778b237f6"),
    ],
)
def test_trajectory_sampler_preserves_frozen_outputs(seed, ep_len, kmax, expected_digest):
    """Byte parity with samplers at 18f8557, including rejected anchors and episode caps."""
    from crlo.eval.trajectory_queries import (
        sample_trajectory_queries,
        sample_trajectory_queries_with_metadata,
    )

    ds = FakeDS(ep_len=ep_len)
    queries = sample_trajectory_queries_with_metadata(ds, 0.99, 50, kmax, seed)
    actual_digest = hashlib.sha256(b"".join(a.tobytes() for a in queries)).hexdigest()
    assert actual_digest == expected_digest
    plain = sample_trajectory_queries(ds, 0.99, 50, kmax, seed)
    for actual, expected in zip(plain, queries[:4], strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(ds.dataset["observations"][queries.indices], queries.states)
    np.testing.assert_array_equal(
        ds.dataset["observations"][queries.indices + queries.offsets], queries.goals
    )
    np.testing.assert_array_equal(ds.terminal_locs[queries.episode_ids], queries.episode_ends)
    assert np.all(queries.indices + queries.offsets <= queries.episode_ends)


def test_build_manifest_freezes_indices_and_permutation(tmp_path, monkeypatch):
    import crlo.eval.evaluate_frozen_goal_retrieval as mod
    from crlo.eval.trajectory_queries import sample_trajectory_queries_with_metadata

    ds = FakeDS()
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        mod,
        "_dataset_triples",
        lambda c: (
            ds,
            sample_trajectory_queries_with_metadata(ds, c.discount, c.n_triples, c.kmax, c.seed),
        ),
    )
    mod.build_manifest(cfg)
    z = np.load(cfg.manifest_path)
    side = json.loads(cfg.manifest_path.with_suffix(".json").read_text())
    # canonical negative permutation: FIRST draw of default_rng(seed),
    # exactly as run_retrieval_eval consumes it before any other rng use
    assert np.array_equal(z["neg_permutation"], np.random.default_rng(0).permutation(50))
    assert len(z["t_index"]) == 50
    assert side["n_triples"] == 50 and side["seed"] == 0
    assert len(side["manifest_sha256"]) == 64


def test_score_mode_replay_verifies_identity(tmp_path, monkeypatch):
    import crlo.eval.evaluate_frozen_goal_retrieval as mod
    from crlo.eval.trajectory_queries import sample_trajectory_queries_with_metadata

    ds = FakeDS()

    def triples(c):
        return ds, sample_trajectory_queries_with_metadata(
            ds, c.discount, c.n_triples, c.kmax, c.seed
        )

    monkeypatch.setattr(mod, "_dataset_triples", triples)
    mod.build_manifest(_cfg(tmp_path))

    # replay against a *different* dataset must fail before scoring
    ds2 = FakeDS(seed=99)
    monkeypatch.setattr(
        mod,
        "_dataset_triples",
        lambda c: (
            ds2,
            sample_trajectory_queries_with_metadata(ds2, c.discount, c.n_triples, c.kmax, c.seed),
        ),
    )
    cfg = _cfg(
        tmp_path,
        mode="score",
        checkpoint_dir=tmp_path,
        critic="cosine",
        label="cosine_s42",
        training_seed=42,
    )
    # triple indices depend only on the rng + terminal structure, so a
    # same-shape different-content dataset reproduces them; the frozen
    # content checksums are what catch the identity mismatch.
    with pytest.raises(ValueError, match="no longer reproduces|mismatch"):
        mod.score(cfg)

    # tampering with the stored manifest must fail the hash check
    monkeypatch.setattr(mod, "_dataset_triples", triples)
    with np.load(cfg.manifest_path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["t_index"] = arrays["t_index"] + 1
    with cfg.manifest_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    with pytest.raises(ValueError, match="hash mismatch"):
        mod.score(cfg)


def test_score_mode_config_mismatch_fails(tmp_path, monkeypatch):
    import crlo.eval.evaluate_frozen_goal_retrieval as mod
    from crlo.eval.trajectory_queries import sample_trajectory_queries_with_metadata

    ds = FakeDS()
    monkeypatch.setattr(
        mod,
        "_dataset_triples",
        lambda c: (
            ds,
            sample_trajectory_queries_with_metadata(ds, c.discount, c.n_triples, c.kmax, c.seed),
        ),
    )
    mod.build_manifest(_cfg(tmp_path))
    cfg = _cfg(
        tmp_path,
        mode="score",
        n_triples=49,
        checkpoint_dir=tmp_path,
        critic="cosine",
        label="cosine_s42",
        training_seed=42,
    )
    with pytest.raises(ValueError, match="manifest n_triples"):
        mod.score(cfg)
