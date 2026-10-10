"""Finite zero-embedding derivatives and unchanged nonzero cosine scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crlo.agents.embeddings import normalize_embeddings


def test_zero_embedding_has_finite_jacobian():
    zero = jnp.zeros((4,), jnp.float32)
    np.testing.assert_array_equal(normalize_embeddings(zero), zero)
    jacobian = jax.jacrev(normalize_embeddings)(zero)
    assert np.isfinite(np.asarray(jacobian)).all()
    np.testing.assert_array_equal(jacobian, jnp.eye(4) / 1e-8)


def test_nonzero_scores_preserve_original_formula():
    x = jnp.asarray(np.random.default_rng(41).normal(size=(32, 8)), jnp.float32)
    expected = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    np.testing.assert_array_equal(normalize_embeddings(x), expected)
