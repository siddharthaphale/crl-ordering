"""JAX noise schedules. Mirrors ``tests/cpu/diffusion_reference`` for parity testing."""

from __future__ import annotations

import jax.numpy as jnp


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> jnp.ndarray:
    return jnp.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> jnp.ndarray:
    steps = jnp.arange(T + 1, dtype=jnp.float32)
    f = jnp.cos((steps / T + s) / (1.0 + s) * jnp.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return jnp.clip(betas, 0.0, 0.999)


def schedule_coefficients(betas: jnp.ndarray) -> dict[str, jnp.ndarray]:
    alphas = 1.0 - betas
    alpha_bar = jnp.cumprod(alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": jnp.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": jnp.sqrt(1.0 - alpha_bar),
        "sqrt_recip_alphas": 1.0 / jnp.sqrt(alphas),
    }
