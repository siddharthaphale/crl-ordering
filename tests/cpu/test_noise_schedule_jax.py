"""JAX schedule == numpy reference within 1e-5."""

from __future__ import annotations

import numpy as np

from crlo.agents._diffusion_ql import noise_schedule as jax_ns

from . import diffusion_reference as ref


def test_linear_schedule_matches_numpy_ref() -> None:
    betas_ref = ref.linear_beta_schedule(100)
    betas_jax = np.asarray(jax_ns.linear_beta_schedule(100))
    assert np.allclose(betas_ref, betas_jax, atol=1e-5)


def test_cosine_schedule_matches_numpy_ref() -> None:
    betas_ref = ref.cosine_beta_schedule(100)
    betas_jax = np.asarray(jax_ns.cosine_beta_schedule(100))
    assert np.allclose(betas_ref, betas_jax, atol=1e-5)


def test_schedule_coeffs_match() -> None:
    betas_ref = ref.linear_beta_schedule(50)
    cref = ref.schedule_coefficients(betas_ref)
    import jax.numpy as jnp

    cjax = jax_ns.schedule_coefficients(jnp.asarray(betas_ref))
    for k in cref:
        assert np.allclose(cref[k], np.asarray(cjax[k]), atol=1e-5)
