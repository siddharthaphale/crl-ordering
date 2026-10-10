"""CPU tests for the DiffusionSampler protocol and MockSampler stand-in."""

from __future__ import annotations

import numpy as np
import pytest

from crlo.eval.diffusion_sampler import DiffusionSampler

from .gaussian_sampler import MockSampler


def test_mock_sampler_satisfies_protocol() -> None:
    sampler = MockSampler(action_dim=2, seed=0)
    assert isinstance(sampler, DiffusionSampler)


def test_sample_shape_single_query() -> None:
    sampler = MockSampler(action_dim=2, seed=0)
    s = np.array([0.0, 0.0])
    g = np.array([1.0, 0.0])
    samples = sampler.sample(s, g, K=64)
    assert samples.shape == (64, 2)


def test_sample_rejects_zero_K() -> None:  # noqa: N802
    sampler = MockSampler(seed=0)
    with pytest.raises(ValueError):
        sampler.sample(np.zeros(2), np.zeros(2), K=0)


def test_log_likelihood_shape_and_finiteness() -> None:
    sampler = MockSampler(action_dim=2, seed=0)
    s = np.array([0.0, 0.0])
    g = np.array([1.0, 0.0])
    samples = sampler.sample(s, g, K=32)
    ll = sampler.log_likelihood(samples, s, g)
    assert ll.shape == (32,)
    assert np.all(np.isfinite(ll))


def test_log_likelihood_higher_near_mean() -> None:
    sampler = MockSampler(action_dim=2, alpha=0.5, sigma=0.3, seed=0)
    s = np.array([0.0, 0.0])
    g = np.array([1.0, 0.0])
    # Mean is alpha*(g-s) = (0.5, 0)
    near_mean = np.array([[0.5, 0.0]])
    far = np.array([[5.0, 5.0]])
    ll_near = sampler.log_likelihood(near_mean, s, g)
    ll_far = sampler.log_likelihood(far, s, g)
    assert ll_near[0] > ll_far[0]


def test_sample_determinism_with_seed() -> None:
    s = np.array([0.0, 0.0])
    g = np.array([1.0, 0.0])
    a1 = MockSampler(seed=42).sample(s, g, K=8)
    a2 = MockSampler(seed=42).sample(s, g, K=8)
    np.testing.assert_array_equal(a1, a2)


def test_action_dim_attribute() -> None:
    sampler = MockSampler(action_dim=5, seed=0)
    assert sampler.action_dim == 5
    samples = sampler.sample(np.zeros(2), np.zeros(2), K=3)
    assert samples.shape == (3, 5)
