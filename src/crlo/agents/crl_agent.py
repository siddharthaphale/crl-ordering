"""Raw bilinear contrastive critic and DDPG+BC actor factory."""

from __future__ import annotations

from typing import Any

from crlo.agents._base_config import _apply_capacity_overrides, base_crl_config
from crlo.utils.ogbench_vendor import ensure_ogbench


def make_crl_agent(
    seed: int,
    example_batch: dict,
    bc_alpha: float,
    discount: float,
    model_config: dict | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    latent_dim: int | None = None,
) -> Any:
    """Instantiate Gaussian CRL agent (DDPG+BC actor, contrastive critic).

    Args:
        seed: JAX random seed.
        example_batch: Dict with 'observations' and 'actions' keys
            (batch_size=1).
        bc_alpha: BC regularisation coefficient for DDPG+BC actor loss.
        discount: Discount factor gamma.
        hidden_dims: Override actor/value MLP widths. ``None`` keeps the
            OGBench default (large; sensible for the real tasks). The 2D toy
            passes a small net to avoid overfitting jagged surfaces.
        model_config: Exact checkpoint configuration overrides.
        latent_dim: Override the contrastive latent width. ``None`` = default.

    Returns:
        CRLAgent instance.
    """
    ensure_ogbench()
    import flax  # noqa: PLC0415
    from agents.crl import CRLAgent  # noqa: PLC0415

    config = base_crl_config(bc_alpha=bc_alpha, discount=discount)
    config["agent_name"] = "crl"
    _apply_capacity_overrides(config, hidden_dims, latent_dim)
    config.update(model_config or {})
    return CRLAgent.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        flax.core.FrozenDict(config),
    )
