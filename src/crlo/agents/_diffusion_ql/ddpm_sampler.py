"""JAX DDPM forward (q_sample) and reverse (p_sample_loop) operations.

Pure functional: takes pre-computed schedule coefficients (numpy-equivalent
to ``tests/cpu/diffusion_reference.schedule_coefficients``) and a network apply function
``eps_theta(params, a_t, t, s, g) -> eps_pred``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def q_sample(
    x0: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray, coeffs: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Forward marginal x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps.

    ``t`` is a ``(B,)`` integer array; ``x0`` and ``eps`` are ``(B, d)``.
    """
    sab = coeffs["sqrt_alpha_bar"][t][:, None]
    somab = coeffs["sqrt_one_minus_alpha_bar"][t][:, None]
    return sab * x0 + somab * eps


def p_sample_step(
    x_t: jnp.ndarray,
    t: int,
    eps_pred: jnp.ndarray,
    noise: jnp.ndarray,
    coeffs: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """One reverse step (Ho et al. eq. 11) with fixed-small variance.

    ``t`` is a scalar python int; ``x_t``, ``eps_pred``, ``noise`` are
    ``(B, d)``. When ``t == 0`` the noise term is gated to zero.
    """
    beta_t = coeffs["betas"][t]
    somab_t = coeffs["sqrt_one_minus_alpha_bar"][t]
    sra_t = coeffs["sqrt_recip_alphas"][t]
    mean = sra_t * (x_t - (beta_t / somab_t) * eps_pred)
    return mean + (t > 0) * jnp.sqrt(beta_t) * noise


def p_sample_loop(
    eps_theta: Callable[..., jnp.ndarray],
    params: Any,
    s: jnp.ndarray,
    g: jnp.ndarray,
    coeffs: dict[str, jnp.ndarray],
    rng: jax.Array,
    action_dim: int,
) -> jnp.ndarray:
    """Reverse-time loop from x_T ~ N(0, I) to x_0.

    Returns sampled actions of shape ``(B, action_dim)``.
    """
    T = coeffs["betas"].shape[0]
    B = s.shape[0]
    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, (B, action_dim))
    for t in range(T - 1, -1, -1):
        t_arr = jnp.full((B,), t, dtype=jnp.int32)
        eps_pred = eps_theta(params, x, t_arr, s, g)
        rng, sub = jax.random.split(rng)
        noise = jax.random.normal(sub, (B, action_dim))
        x = p_sample_step(x, t, eps_pred, noise, coeffs)
    return x
