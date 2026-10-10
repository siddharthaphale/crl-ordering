"""DDPM training step for the Diffusion-QL behavior policy.

Implements the Ho et al. 2020 "simple" objective (eq. 14):

    L(theta) = E_{(s,a,g), t ~ U[0,T), eps ~ N(0,I)}
               || eps - eps_theta(sqrt(ab_t) a + sqrt(1-ab_t) eps,
                                  t, s, g) ||^2

The behavior policy is trained on offline (s, a, g) tuples sampled by the
GCDataset wrapper (same one used by CRL).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from crlo.agents._diffusion_ql.ddpm_sampler import q_sample


def ddpm_loss(
    params: Any,
    eps_theta_apply,
    batch: dict[str, jnp.ndarray],
    coeffs: dict[str, jnp.ndarray],
    T: int,
    rng: jax.Array,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Ho et al. simple objective on (s, a, g) batch."""
    s = batch["observations"]
    a0 = batch["actions"]
    g = batch["value_goals"]
    B = a0.shape[0]
    rng_t, rng_eps = jax.random.split(rng)
    t = jax.random.randint(rng_t, (B,), 0, T)
    eps = jax.random.normal(rng_eps, a0.shape)
    a_t = q_sample(a0, t, eps, coeffs)
    eps_pred = eps_theta_apply(params, a_t, t, s, g)
    loss = jnp.mean((eps - eps_pred) ** 2)
    return loss, {"ddpm_loss": loss, "eps_pred_norm": jnp.mean(eps_pred**2)}


def make_train_step(
    eps_theta_apply, optimizer: optax.GradientTransformation, coeffs: dict[str, jnp.ndarray], T: int
):
    """Returns a jitted ``(params, opt_state, batch, rng) -> ...`` train step."""

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        (loss, info), grads = jax.value_and_grad(ddpm_loss, has_aux=True)(
            params, eps_theta_apply, batch, coeffs, T, rng
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, info

    return train_step
