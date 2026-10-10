"""JAX q_sample / p_sample_step match the pure-numpy reference."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from crlo.agents._diffusion_ql import ddpm_sampler as jax_samp
from crlo.agents._diffusion_ql import noise_schedule as jax_ns

from . import diffusion_reference as ref


def test_q_sample_jax_matches_ref() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(4, 3))
    eps = rng.normal(size=(4, 3))
    t_np = 7
    betas = ref.linear_beta_schedule(20)
    cref = ref.schedule_coefficients(betas)
    cjax = jax_ns.schedule_coefficients(jnp.asarray(betas))
    ref_xt = ref.q_sample(x0, t_np, eps, cref)
    t_arr = jnp.full((4,), t_np, dtype=jnp.int32)
    jax_xt = np.asarray(jax_samp.q_sample(jnp.asarray(x0), t_arr, jnp.asarray(eps), cjax))
    assert np.allclose(ref_xt, jax_xt, atol=1e-6)


def test_p_sample_step_jax_matches_ref() -> None:
    rng = np.random.default_rng(0)
    betas = ref.linear_beta_schedule(20)
    cref = ref.schedule_coefficients(betas)
    cjax = jax_ns.schedule_coefficients(jnp.asarray(betas))
    x_t = rng.normal(size=(4, 3))
    eps_pred = rng.normal(size=(4, 3))
    noise = rng.normal(size=(4, 3))
    for t in [0, 1, 10, 19]:
        ref_out = ref.p_sample_step(x_t, t, eps_pred, noise, cref)
        jax_out = np.asarray(
            jax_samp.p_sample_step(
                jnp.asarray(x_t),
                t,
                jnp.asarray(eps_pred),
                jnp.asarray(noise),
                cjax,
            )
        )
        assert np.allclose(ref_out, jax_out, atol=1e-6), f"mismatch at t={t}"
