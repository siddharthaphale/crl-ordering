"""Pure-numpy DDPM math reference (for Diffusion-QL behavior policy).

Closed-form, no JAX. Used as the ground truth for equivalence tests in
``tests/cpu/jax_reference/`` and (indirectly, via the JAX wrappers) in
``tests/cpu/jax/``. Equations are tagged with Ho et al. 2020 / Nichol &
Dhariwal 2021 references.

Definitions follow the standard DDPM notation:
  - ``betas`` is the noise schedule, with ``alphas = 1 - betas`` and
    ``alpha_bar_t = prod_{s<=t} alphas[s]``.
  - Forward marginal (Ho et al. eq. 4):

        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0,
                         (1 - alpha_bar_t) * I)

  - DDPM reverse mean / variance (Ho et al. eq. 11):

        mu_theta(x_t, t) =
            (1 / sqrt(alpha_t)) *
            (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
        Sigma_theta(t) = beta_t * I    # fixed-small variance choice

  - Training loss (Ho et al. eq. 14, "simple" objective):

        L_simple(theta) = E_{x_0, t, eps} || eps - eps_theta(x_t, t) ||^2
        with x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps.

For the Diffusion-QL behavior policy we condition the score on ``(s, g)``;
the noise schedule and forward/reverse math are unchanged. Conditioning is
handled by the score network, which is implemented in ``_diffusion_ql/``.
"""

from __future__ import annotations

import numpy as np


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """Ho et al. 2020 linear schedule of length ``T``."""
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal 2021 cosine schedule.

    Constructs ``alpha_bar_t = cos((t/T + s) / (1 + s) * pi/2)^2`` and
    derives ``beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}`` clipped to
    ``[0, 0.999]``.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos((steps / T + s) / (1.0 + s) * np.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0.0, 0.999)


def schedule_coefficients(betas: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-compute the alpha / alpha_bar coefficients used by q_sample / p_sample."""
    betas = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": np.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": np.sqrt(1.0 - alpha_bar),
        "sqrt_recip_alphas": 1.0 / np.sqrt(alphas),
    }


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, coeffs: dict[str, np.ndarray]) -> np.ndarray:
    """Forward marginal sample x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps.

    ``x0`` and ``eps`` have matching shape ``(..., d)``. ``t`` is a python int
    in ``[0, T)``.
    """
    sab = coeffs["sqrt_alpha_bar"][t]
    somab = coeffs["sqrt_one_minus_alpha_bar"][t]
    return sab * x0 + somab * eps


def p_sample_step(
    x_t: np.ndarray, t: int, eps_pred: np.ndarray, noise: np.ndarray, coeffs: dict[str, np.ndarray]
) -> np.ndarray:
    """One DDPM reverse step (Ho et al. eq. 11) with fixed-small variance.

    Returns ``x_{t-1}`` given ``x_t``, the model's noise prediction at step
    ``t``, and a sample of standard-normal noise to inject (set ``noise``
    to zeros when ``t == 0``).
    """
    beta_t = coeffs["betas"][t]
    somab_t = coeffs["sqrt_one_minus_alpha_bar"][t]
    sra_t = coeffs["sqrt_recip_alphas"][t]
    mean = sra_t * (x_t - (beta_t / somab_t) * eps_pred)
    if t == 0:
        return mean
    return mean + np.sqrt(beta_t) * noise


def ddpm_simple_loss(eps: np.ndarray, eps_pred: np.ndarray) -> float:
    """Mean-squared error between true and predicted noise (Ho et al. eq. 14)."""
    diff = (eps - eps_pred).reshape(-1)
    return float(np.mean(diff * diff))


def p_sample_loop(
    eps_theta_fn, shape: tuple[int, ...], coeffs: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Full reverse-time loop from x_T ~ N(0, I) down to x_0.

    ``eps_theta_fn(x_t, t) -> eps_pred`` is the (numpy) score network. This
    is the closed-form reference used to validate the JAX sampler.
    """
    T = coeffs["betas"].shape[0]
    x = rng.normal(size=shape)
    for t in range(T - 1, -1, -1):
        eps_pred = eps_theta_fn(x, t)
        noise = rng.normal(size=shape) if t > 0 else np.zeros(shape)
        x = p_sample_step(x, t, eps_pred, noise, coeffs)
    return x
