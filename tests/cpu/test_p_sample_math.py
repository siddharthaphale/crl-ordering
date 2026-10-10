"""Single DDPM reverse step matches Ho et al. eq. 11."""

from __future__ import annotations

import numpy as np

from .diffusion_reference import (
    linear_beta_schedule,
    p_sample_step,
    schedule_coefficients,
)


def test_p_sample_step_matches_closed_form() -> None:
    coeffs = schedule_coefficients(linear_beta_schedule(20))
    rng = np.random.default_rng(0)
    x_t = rng.normal(size=(4, 3))
    eps = rng.normal(size=(4, 3))
    noise = rng.normal(size=(4, 3))
    t = 5
    beta_t = coeffs["betas"][t]
    sra = coeffs["sqrt_recip_alphas"][t]
    somab = coeffs["sqrt_one_minus_alpha_bar"][t]
    expected_mean = sra * (x_t - (beta_t / somab) * eps)
    expected = expected_mean + np.sqrt(beta_t) * noise
    actual = p_sample_step(x_t, t, eps, noise, coeffs)
    assert np.allclose(actual, expected)


def test_p_sample_step_at_t0_drops_noise() -> None:
    coeffs = schedule_coefficients(linear_beta_schedule(20))
    rng = np.random.default_rng(0)
    x_t = rng.normal(size=(2, 3))
    eps = rng.normal(size=(2, 3))
    noise = rng.normal(size=(2, 3))
    out = p_sample_step(x_t, 0, eps, noise, coeffs)
    beta_0 = coeffs["betas"][0]
    sra_0 = coeffs["sqrt_recip_alphas"][0]
    somab_0 = coeffs["sqrt_one_minus_alpha_bar"][0]
    expected = sra_0 * (x_t - (beta_0 / somab_0) * eps)
    assert np.allclose(out, expected)
