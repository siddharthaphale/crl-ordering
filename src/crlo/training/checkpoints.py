"""Checkpoint I/O for critic and diffusion training and evaluation.

Each checkpoint stores parameters, optimizer state, architecture metadata, and
training RNG/update state. Inference accepts historical parameter-only bundles;
resumption requires complete optimizer and RNG state. Legacy names are translated
in memory without rewriting archived parameters.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from crlo.agents._diffusion_ql.policy import DiffusionQLBehavior

_Agent = TypeVar("_Agent")


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def save_critic(agent: Any, ckpt_dir: Path, meta: dict) -> None:
    """Persist the agent's params + opt_state + factory metadata.

    ``meta`` is whatever the loader will need to rebuild the agent skeleton
    via its factory — e.g. ``{"critic": "raw_bilinear", "task": "...",
    "bc_alpha": 3.0, "discount": 0.99, "example_batch_shapes": {...}}``.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        ckpt_dir / "params.msgpack",
        fser.to_bytes(agent.network.params),
    )
    _atomic_write(
        ckpt_dir / "opt_state.msgpack",
        fser.to_bytes(agent.network.opt_state),
    )
    meta = {
        **meta,
        "format_version": 2,
        "agent_config": dict(agent.config),
        "numpy_rng_state": np.random.get_state(),
    }
    _atomic_write(
        ckpt_dir / "training_state.msgpack",
        fser.msgpack_serialize(
            {
                "rng": np.asarray(agent.rng),
                "step": np.asarray(agent.network.step),
            }
        ),
    )
    _atomic_write(ckpt_dir / "meta.pkl", pickle.dumps(meta))


def load_critic(
    ckpt_dir: Path,
    rebuild: Callable[[dict[str, Any]], _Agent],
    *,
    resume: bool = False,
) -> tuple[_Agent, dict[str, Any]]:
    """Reconstruct a critic for inference or resume its recorded training state.

    Args:
        ckpt_dir: Checkpoint directory containing ``meta.pkl`` and ``params.msgpack``.
        rebuild: Callable receiving migrated metadata and returning an agent
            with the recorded architecture and optimizer structure.
        resume: If false, restore parameters only, retaining the skeleton's
            optimizer, step, and JAX RNG. If true, also restore optimizer state,
            step, and JAX RNG; restore global NumPy RNG state when recorded.

    Returns:
        The restored agent and migrated metadata. Legacy inference-only
        checkpoints remain readable with ``resume=False``.

    Raises:
        ValueError: Resumption was requested but optimizer or training-state
            files are missing.
        FileNotFoundError: Required metadata or parameter files are missing.
    """
    meta = pickle.loads((ckpt_dir / "meta.pkl").read_bytes())
    meta = migrate_metadata(meta)
    skeleton = rebuild(meta)
    params = fser.from_bytes(
        skeleton.network.params,
        (ckpt_dir / "params.msgpack").read_bytes(),
    )
    # opt_state only matters for resuming training; archived eval-only
    # checkpoints (for example, the historical inference bundle) may omit it.
    opt_state = skeleton.network.opt_state
    opt_path = ckpt_dir / "opt_state.msgpack"
    if resume and opt_path.exists():
        opt_state = fser.from_bytes(opt_state, opt_path.read_bytes())
    restored_network = skeleton.network.replace(
        params=params,
        opt_state=opt_state,
    )
    restored = skeleton.replace(network=restored_network)
    if resume:
        if "numpy_rng_state" in meta:
            np.random.set_state(meta["numpy_rng_state"])
        if not opt_path.exists() or not (ckpt_dir / "training_state.msgpack").exists():
            raise ValueError("Checkpoint lacks optimizer/RNG state required for resumption")
        state = fser.msgpack_restore((ckpt_dir / "training_state.msgpack").read_bytes())
        restored = restored.replace(
            rng=jnp.asarray(state["rng"]), network=restored.network.replace(step=state["step"])
        )
    return restored, meta


def save_diffusion(
    policy: DiffusionQLBehavior,
    ckpt_dir: Path,
    meta: dict | None = None,
) -> None:
    """Persist a :class:`DiffusionQLBehavior` (params + opt_state + arch).

    Stores enough info that :func:`load_diffusion` rebuilds via
    ``DiffusionQLBehavior.create`` with the same shapes, schedule, and
    network architecture.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    full_meta = {
        "action_dim": policy.action_dim,
        "state_dim": policy.state_dim,
        "goal_dim": policy.goal_dim,
        "T": policy.T,
        "hidden_dims": tuple(policy.network.hidden_dims),
        "time_embed_dim": policy.network.time_embed_dim,
        "lr": policy.lr,
        "schedule": policy.schedule,
        "steps": policy.steps,
        "format_version": 2,
        "numpy_rng_state": np.random.get_state(),
        **(meta or {}),
    }
    _atomic_write(
        ckpt_dir / "params.msgpack",
        fser.to_bytes(policy.params),
    )
    _atomic_write(
        ckpt_dir / "opt_state.msgpack",
        fser.to_bytes(policy.opt_state),
    )
    _atomic_write(
        ckpt_dir / "training_state.msgpack",
        fser.msgpack_serialize(
            {
                "rng": np.asarray(policy.rng),
                "steps": policy.steps,
            }
        ),
    )
    _atomic_write(ckpt_dir / "meta.pkl", pickle.dumps(full_meta))


def load_diffusion(ckpt_dir: Path, seed: int = 0, *, resume: bool = False) -> DiffusionQLBehavior:
    """Reconstruct a diffusion policy for inference or training resumption.

    Args:
        ckpt_dir: Directory containing saved parameters and architecture metadata.
        seed: Initialization seed for the fresh policy and its inference RNG.
            Recorded JAX RNG state replaces it when resuming.
        resume: Restore optimizer, update count, and JAX RNG as well as parameters.
            Also restore global NumPy RNG state when present in metadata.
            With ``resume=False``, retain the freshly initialized training state.

    Returns:
        A diffusion policy with the recorded architecture and saved parameters.

    Raises:
        ValueError: Resumption was requested without optimizer or training-state files.
        FileNotFoundError: Required metadata or parameter files are missing.
    """
    from crlo.agents._diffusion_ql.policy import (  # noqa: PLC0415
        DiffusionQLBehavior,
    )

    meta = pickle.loads((ckpt_dir / "meta.pkl").read_bytes())
    meta = migrate_metadata(meta)
    skeleton = DiffusionQLBehavior.create(
        seed=seed,
        action_dim=meta["action_dim"],
        state_dim=meta["state_dim"],
        goal_dim=meta["goal_dim"],
        denoising_steps=meta["T"],
        hidden_dims=tuple(meta["hidden_dims"]),
        lr=meta.get("lr", 3e-4),
        schedule=meta.get("schedule", "cosine"),
        time_embed_dim=meta.get("time_embed_dim", 32),
    )
    skeleton.params = fser.from_bytes(
        skeleton.params,
        (ckpt_dir / "params.msgpack").read_bytes(),
    )
    # opt_state only matters for resuming training; archived eval-only
    # checkpoints (for example, the historical inference bundle) may omit it.
    opt_path = ckpt_dir / "opt_state.msgpack"
    if resume:
        if "numpy_rng_state" in meta:
            np.random.set_state(meta["numpy_rng_state"])
        if not opt_path.exists() or not (ckpt_dir / "training_state.msgpack").exists():
            raise ValueError("Checkpoint lacks optimizer/RNG state required for resumption")
        state = fser.msgpack_restore((ckpt_dir / "training_state.msgpack").read_bytes())
        skeleton.rng = jnp.asarray(state["rng"])
        skeleton.steps = state["steps"]
        skeleton.opt_state = fser.from_bytes(
            skeleton.opt_state,
            opt_path.read_bytes(),
        )
    skeleton.params = jax.tree.map(jnp.asarray, skeleton.params)
    skeleton.opt_state = jax.tree.map(jnp.asarray, skeleton.opt_state)
    return skeleton


LEGACY_MODEL_NAMES = {
    "twohead_a": "two_head_separate_encoders",
    "twohead_b1": "two_head_shared_encoder_stopped_gradient",
    "twohead_b2": "two_head_shared_encoder",
    "twohead_b0": "two_head_shared_encoder_td_only",
}


def migrate_metadata(meta: dict) -> dict:
    """Translate legacy identifiers without rewriting checkpoint parameter bytes."""
    result = dict(meta)
    name = result.get("critic")
    if name:
        suffix = "_with_actor" if name.endswith("_full") else ""
        base = name.removesuffix("_full")
        result["critic"] = LEGACY_MODEL_NAMES.get(base, base) + suffix
    return result
