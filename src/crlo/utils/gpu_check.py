"""Verify that JAX executes on an actual GPU."""

from __future__ import annotations


def gpu_check() -> None:
    """Fail if JAX cannot run and synchronize a computation on the GPU."""
    import jax
    import jax.numpy as jnp

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("JAX has no GPU device; check the CUDA dependency extra")
    with jax.default_device(devices[0]):
        value = (jnp.ones((32, 32)) @ jnp.ones((32, 32))).block_until_ready()
    if value.devices() != {devices[0]}:
        raise RuntimeError("Preflight did not execute on the requested GPU")
