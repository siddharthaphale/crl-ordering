"""Round-trip serialisation for diffusion-policy checkpoints.

The CRL-family critic round-trip lives in ``tests/gpu/`` because rebuilding
the OGBench agent skeleton needs JAX + Mujoco. The diffusion-policy
round-trip is pure JAX so it runs here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior
from crlo.training import checkpoints


def test_diffusion_save_load_roundtrip(tmp_path) -> None:
    original = DiffusionQLBehavior.create(
        seed=0,
        action_dim=2,
        state_dim=3,
        goal_dim=3,
        denoising_steps=4,
        hidden_dims=(16,),
    )
    batch = {
        "observations": jnp.zeros((4, 3)),
        "actions": jnp.zeros((4, 2)),
        "value_goals": jnp.zeros((4, 3)),
    }
    original.update(batch)
    ckpt_dir = tmp_path / "ckpt"
    checkpoints.save_diffusion(original, ckpt_dir, meta={"task": "toy"})

    restored = checkpoints.load_diffusion(ckpt_dir, seed=42)
    assert restored.action_dim == original.action_dim
    assert restored.state_dim == original.state_dim
    assert restored.goal_dim == original.goal_dim
    assert restored.T == original.T

    s = np.zeros((1, 3), dtype=np.float32)
    g = np.ones((1, 3), dtype=np.float32)
    a_orig = np.asarray(
        original.network.apply(
            original.params,
            jnp.zeros((1, 2)),
            jnp.zeros((1,), jnp.int32),
            jnp.asarray(s),
            jnp.asarray(g),
        )
    )
    a_restored = np.asarray(
        restored.network.apply(
            restored.params,
            jnp.zeros((1, 2)),
            jnp.zeros((1,), jnp.int32),
            jnp.asarray(s),
            jnp.asarray(g),
        )
    )
    assert np.allclose(a_orig, a_restored, atol=1e-6)


def test_diffusion_meta_preserved(tmp_path) -> None:
    import pickle

    policy = DiffusionQLBehavior.create(
        seed=0,
        action_dim=2,
        state_dim=2,
        goal_dim=2,
        denoising_steps=2,
        hidden_dims=(8,),
    )
    checkpoints.save_diffusion(
        policy,
        tmp_path,
        meta={"task": "cube", "seed": 7},
    )
    meta = pickle.loads((tmp_path / "meta.pkl").read_bytes())
    assert meta["task"] == "cube"
    assert meta["seed"] == 7
    assert meta["action_dim"] == 2
    assert meta["T"] == 2
