"""Frozen executed-value CRN cell keys: candidate-invariant, (q, h, r)-distinct (plan §5.4)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from crlo.eval.candidate_returns import cell_key  # noqa: E402


def test_cell_key_matches_frozen_fold_in_chain():
    want = jax.random.fold_in(
        jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(7001), 34), 0), 3
    )
    assert np.array_equal(cell_key(7001, 34, 0, 3), want)


def test_cell_key_candidate_free_and_distinct_across_cells():
    base = cell_key(7001, 34, 0, 3)
    # candidate index is not an input: every candidate in the cell shares it
    assert np.array_equal(base, cell_key(7001, 34, 0, 3))
    seen = {tuple(np.asarray(base).tolist())}
    for q, h, r in ((35, 0, 3), (34, 1, 3), (34, 0, 4), (0, 0, 0)):
        k = tuple(np.asarray(cell_key(7001, q, h, r)).tolist())
        assert k not in seen, (q, h, r)
        seen.add(k)


def test_noise_stream_identical_across_candidates():
    """Two 'candidates' replaying a cell consume identical subkey streams.

    The continuation stream advances by jax.random.split once per step,
    so after resetting to the cell key the i-th step's noise key is
    identical no matter which candidate is being rolled out; sampled
    actions may still differ once observations diverge (plan §10.2).
    """

    def stream(n):
        rng = cell_key(7001, 34, 1, 2)
        subs = []
        for _ in range(n):
            rng, sub = jax.random.split(rng)
            subs.append(np.asarray(sub))
        return subs

    a, b = stream(5), stream(5)
    for ka, kb in zip(a, b, strict=True):
        assert np.array_equal(ka, kb)
    # halves A and B are independent streams
    rng_b = cell_key(7001, 34, 0, 2)
    _, sub_b = jax.random.split(rng_b)
    assert not np.array_equal(a[0], np.asarray(sub_b))


def test_fast_sampler_rng_and_output_equivalence():
    """StochasticDiffusionContinuation must evolve rng identically to the eager sample path
    and produce the same actions up to compiler-fusion tolerance."""
    np_mod = np
    from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior
    from crlo.eval.candidate_returns import StochasticDiffusionContinuation, cell_key

    policy = DiffusionQLBehavior.create(
        seed=0, action_dim=3, state_dim=5, goal_dim=5, denoising_steps=4, hidden_dims=(16, 16)
    )
    s = np_mod.arange(5, dtype=np_mod.float32)
    g = s + 1.0

    policy.rng = cell_key(7001, 34, 0, 0)
    eager = [np_mod.asarray(policy.sample(s, g, 1)) for _ in range(3)]
    rng_eager = np_mod.asarray(policy.rng)

    policy.rng = cell_key(7001, 34, 0, 0)
    fast = StochasticDiffusionContinuation(policy)
    fast.rng = cell_key(7001, 34, 0, 0)
    fast_out = [fast.sample(s, g, 1) for _ in range(3)]
    rng_fast = np_mod.asarray(fast.rng)

    assert np_mod.array_equal(rng_eager, rng_fast)  # exact rng evolution
    for a, b in zip(eager, fast_out, strict=True):
        assert a.shape == b.shape == (1, 3)
        assert np_mod.allclose(a, b, atol=1e-5), np_mod.abs(a - b).max()


def test_deterministic_sampler_is_rng_free_and_repeatable():
    from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior
    from crlo.eval.candidate_returns import DeterministicDiffusionContinuation, cell_key

    policy = DiffusionQLBehavior.create(
        seed=0, action_dim=3, state_dim=5, goal_dim=5, denoising_steps=4, hidden_dims=(16, 16)
    )
    det = DeterministicDiffusionContinuation(policy)
    s = np.arange(5, dtype=np.float32)
    g = s + 1.0
    det.rng = cell_key(7001, 1, 0, 0)
    a1 = det.sample(s, g, 1)
    det.rng = cell_key(7001, 999, 1, 4)  # different key: no effect
    a2 = det.sample(s, g, 1)
    assert np.array_equal(a1, a2)  # bit-identical, rng-free
    assert a1.shape == (1, 3)
    # a different (s, g) produces a different mode action
    a3 = det.sample(s + 1.0, g, 1)
    assert not np.array_equal(a1, a3)
