"""q_sample closed-form: x_t mean == sqrt(ab_t) x_0; variance == 1 - ab_t."""

from __future__ import annotations

import numpy as np

from .diffusion_reference import (
    linear_beta_schedule,
    q_sample,
    schedule_coefficients,
)


def test_q_sample_zero_noise_matches_scaled_x0() -> None:
    coeffs = schedule_coefficients(linear_beta_schedule(100))
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(8, 4))
    eps = np.zeros_like(x0)
    for t in [0, 1, 50, 99]:
        xt = q_sample(x0, t, eps, coeffs)
        assert np.allclose(xt, coeffs["sqrt_alpha_bar"][t] * x0)


def test_q_sample_mean_and_variance_empirical() -> None:
    coeffs = schedule_coefficients(linear_beta_schedule(50))
    rng = np.random.default_rng(0)
    x0 = np.ones((1, 3))
    t = 25
    samples = []
    for _ in range(20_000):
        eps = rng.normal(size=x0.shape)
        samples.append(q_sample(x0, t, eps, coeffs))
    samples = np.stack(samples).reshape(-1, 3)
    expected_mean = coeffs["sqrt_alpha_bar"][t] * x0[0]
    expected_var = 1.0 - coeffs["alpha_bar"][t]
    assert np.allclose(samples.mean(axis=0), expected_mean, atol=0.05)
    assert np.allclose(samples.var(axis=0), expected_var, atol=0.05)
