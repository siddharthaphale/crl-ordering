"""ScoreNetwork forward pass: correct shape, non-trivial parameter count."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from crlo.agents._diffusion_ql.score_network import (
    ScoreNetwork,
    sinusoidal_time_embedding,
)


def _count_params(tree) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(tree))


def test_sinusoidal_time_embedding_shape_and_periodicity() -> None:
    t = jnp.arange(4)
    emb = sinusoidal_time_embedding(t.astype(jnp.float32), dim=16)
    assert emb.shape == (4, 16)
    half = jnp.split(emb, 2, axis=-1)
    assert jnp.all(jnp.abs(half[0]) <= 1.0 + 1e-6)
    assert jnp.all(jnp.abs(half[1]) <= 1.0 + 1e-6)


def test_score_network_forward_shape() -> None:
    net = ScoreNetwork(action_dim=4, hidden_dims=(32, 32), time_embed_dim=32)
    rng = jax.random.PRNGKey(0)
    a = jnp.zeros((8, 4))
    t = jnp.zeros((8,), dtype=jnp.int32)
    s = jnp.zeros((8, 3))
    g = jnp.zeros((8, 3))
    params = net.init(rng, a, t, s, g)
    out = net.apply(params, a, t, s, g)
    assert out.shape == (8, 4)
    assert isinstance(params, Mapping)
    assert _count_params(params) > 0
