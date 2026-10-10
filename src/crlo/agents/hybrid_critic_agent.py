"""Cosine critic trained with contrastive loss and a warmed-up Bellman loss."""

from __future__ import annotations

from typing import Any

from crlo.agents._base_config import _apply_capacity_overrides, base_crl_config
from crlo.utils.ogbench_vendor import ensure_ogbench


def make_hybrid_critic_agent(
    seed: int,
    example_batch: dict,
    bc_alpha: float,
    discount: float,
    td_warmup_steps: int = None,
    model_config: dict | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    latent_dim: int | None = None,
) -> Any:
    """Instantiate the hybrid contrastive + TD critic agent.

    Args:
        seed: JAX random seed.
        example_batch: Dict with 'observations' and 'actions' keys.
        bc_alpha: BC regularisation coefficient for DDPG+BC actor loss.
        discount: Discount factor gamma.
        td_warmup_steps: Train the contrastive objective this many steps before
            adding the Bellman objective. Defaults to
            100,000 steps for direct library use.

        model_config: Exact checkpoint configuration overrides.
        hidden_dims: Optional actor and critic widths.
        latent_dim: Optional representation width.

    Returns:
        Hybrid critic agent instance (CRL-like API).
    """
    if td_warmup_steps is None:
        td_warmup_steps = 100000
    ensure_ogbench()
    import flax  # noqa: PLC0415

    from crlo.agents._ogbench_overrides.crl_cosine import (  # noqa: PLC0415
        use_cosine_value_module,
    )
    from crlo.agents._ogbench_overrides.crl_hybrid import (  # noqa: PLC0415
        CRLHybridAgent,
    )

    config = base_crl_config(bc_alpha=bc_alpha, discount=discount)
    config["agent_name"] = "hybrid_critic"
    config["cosine_temperature"] = 0.1
    config["td_warmup_steps"] = td_warmup_steps
    _apply_capacity_overrides(config, hidden_dims, latent_dim)
    config.update(model_config or {})
    with use_cosine_value_module():
        return CRLHybridAgent.create(
            seed,
            example_batch["observations"],
            example_batch["actions"],
            flax.core.FrozenDict(config),
        )
