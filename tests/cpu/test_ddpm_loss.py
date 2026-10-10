"""DDPM simple-objective loss: MSE reduces under a known eps predictor."""

from __future__ import annotations

import numpy as np

from .diffusion_reference import ddpm_simple_loss


def test_loss_zero_when_pred_equals_target() -> None:
    eps = np.random.default_rng(0).normal(size=(10, 3))
    assert ddpm_simple_loss(eps, eps) == 0.0


def test_loss_matches_mean_squared_error() -> None:
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(50, 4))
    pred = rng.normal(size=(50, 4))
    expected = float(np.mean((eps - pred) ** 2))
    assert np.isclose(ddpm_simple_loss(eps, pred), expected)


def test_loss_decreases_under_linear_interpolation() -> None:
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(100, 2))
    bad = rng.normal(size=(100, 2))
    losses = [ddpm_simple_loss(eps, (1 - a) * bad + a * eps) for a in np.linspace(0.0, 1.0, 6)]
    assert all(losses[i + 1] < losses[i] for i in range(len(losses) - 1))
