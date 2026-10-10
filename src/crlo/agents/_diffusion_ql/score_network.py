"""MLP epsilon-prediction network.

Inputs (after concatenation): (state s, action a_t, timestep embedding t,
goal g). Output: predicted noise of shape ``(action_dim,)``.

Time embedding is a sinusoidal positional encoding (Vaswani et al. 2017,
adopted by Ho et al. 2020) projected through a small MLP.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def sinusoidal_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Vaswani 2017 sinusoidal positional encoding for scalar timesteps.

    ``t`` is a ``(B,)`` array of integer-valued floats; ``dim`` must be even.
    Returns ``(B, dim)``.
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ScoreNetwork(nn.Module):
    """Conditional epsilon predictor eps_theta(a_t, t | s, g)."""

    action_dim: int
    hidden_dims: tuple[int, ...]
    time_embed_dim: int

    @nn.compact
    def __call__(
        self, a_t: jnp.ndarray, t: jnp.ndarray, s: jnp.ndarray, g: jnp.ndarray
    ) -> jnp.ndarray:
        t_emb = sinusoidal_time_embedding(t.astype(jnp.float32), self.time_embed_dim)
        t_emb = nn.Dense(self.time_embed_dim)(t_emb)
        t_emb = nn.silu(t_emb)
        h = jnp.concatenate([a_t, s, g, t_emb], axis=-1)
        for d in self.hidden_dims:
            h = nn.Dense(d)(h)
            h = nn.silu(h)
        return nn.Dense(self.action_dim)(h)
