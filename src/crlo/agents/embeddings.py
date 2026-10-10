"""Embedding normalization with defined derivatives at the zero vector."""

from __future__ import annotations

import jax.numpy as jnp


def normalize_embeddings(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute x/(norm(x)+eps), defining the norm derivative at zero as zero.

    Keep the original nonzero score formula. Mask inside the square root so
    reverse-mode differentiation never evaluates its singular derivative at zero.
    """
    squared = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    nonzero = squared > 0
    safe_norm = jnp.sqrt(jnp.where(nonzero, squared, 1.0))
    norm = jnp.where(nonzero, safe_norm, 0.0)
    return x / (norm + eps)
