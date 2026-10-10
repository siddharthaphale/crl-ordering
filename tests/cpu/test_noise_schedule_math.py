"""Numpy DDPM schedule correctness: linear monotonic, cosine endpoints, cumprod."""

from __future__ import annotations

import numpy as np

from .diffusion_reference import (
    cosine_beta_schedule,
    linear_beta_schedule,
    schedule_coefficients,
)


def test_linear_schedule_monotone_and_endpoints() -> None:
    betas = linear_beta_schedule(100, 1e-4, 0.02)
    assert betas.shape == (100,)
    assert np.isclose(betas[0], 1e-4)
    assert np.isclose(betas[-1], 0.02)
    assert np.all(np.diff(betas) >= 0)


def test_cosine_schedule_endpoints_and_alpha_bar_decreasing() -> None:
    T = 200
    betas = cosine_beta_schedule(T)
    assert betas.shape == (T,)
    assert np.all(betas >= 0.0) and np.all(betas <= 0.999)
    coeffs = schedule_coefficients(betas)
    ab = coeffs["alpha_bar"]
    assert ab[0] < 1.0 and ab[-1] < ab[0]
    assert np.all(np.diff(ab) <= 0)


def test_schedule_coefficients_invariants() -> None:
    betas = linear_beta_schedule(50)
    c = schedule_coefficients(betas)
    assert np.allclose(c["alphas"], 1.0 - betas)
    assert np.allclose(c["alpha_bar"], np.cumprod(c["alphas"]))
    assert np.allclose(c["sqrt_alpha_bar"], np.sqrt(c["alpha_bar"]))
    assert np.allclose(c["sqrt_one_minus_alpha_bar"], np.sqrt(1.0 - c["alpha_bar"]))
    assert np.allclose(c["sqrt_recip_alphas"], 1.0 / np.sqrt(c["alphas"]))


def test_invalid_T_raises() -> None:  # noqa: N802
    import pytest

    with pytest.raises(ValueError):
        linear_beta_schedule(0)
    with pytest.raises(ValueError):
        cosine_beta_schedule(0)
