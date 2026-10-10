r"""Two-head contrastive representation and independent scalar value readouts.

Recipes declare separate or shared encoders, TD-gradient routing,
contrastive weight, and whether an advantage-weighted actor is trained.
"""

from __future__ import annotations

from typing import Any

from crlo.agents._base_config import _apply_capacity_overrides, base_crl_config
from crlo.utils.ogbench_vendor import ensure_ogbench


def make_two_head_critic(
    seed: int,
    example_batch: dict,
    bc_alpha: float,
    discount: float,
    variant: str = "two_head_shared_encoder",
    model_config: dict | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    latent_dim: int | None = None,
) -> Any:
    """Instantiate a two-head hybrid agent for ``variant``."""
    ensure_ogbench()
    import flax  # noqa: PLC0415

    from crlo.agents._ogbench_overrides.crl_cosine import (  # noqa: PLC0415
        use_cosine_value_module,
    )
    from crlo.agents._ogbench_overrides.crl_twohead import (  # noqa: PLC0415
        CRLTwoHeadAgent,
    )

    variants = {
        "two_head_separate_encoders": {
            "qhead": "raw",
            "share_grad": False,
            "contrastive_weight": 1.0,
            "train_actor": False,
        },
        "two_head_shared_encoder_stopped_gradient": {
            "qhead": "shared",
            "share_grad": False,
            "contrastive_weight": 1.0,
            "train_actor": False,
        },
        "two_head_shared_encoder": {
            "qhead": "shared",
            "share_grad": True,
            "contrastive_weight": 1.0,
            "train_actor": False,
        },
        "two_head_shared_encoder_td_only": {
            "qhead": "shared",
            "share_grad": True,
            "contrastive_weight": 0.0,
            "train_actor": False,
        },
        "two_head_separate_encoders_with_actor": {
            "qhead": "raw",
            "share_grad": False,
            "contrastive_weight": 1.0,
            "train_actor": True,
        },
        "two_head_shared_encoder_stopped_gradient_with_actor": {
            "qhead": "shared",
            "share_grad": False,
            "contrastive_weight": 1.0,
            "train_actor": True,
        },
        "two_head_shared_encoder_with_actor": {
            "qhead": "shared",
            "share_grad": True,
            "contrastive_weight": 1.0,
            "train_actor": True,
        },
        "two_head_shared_encoder_td_only_with_actor": {
            "qhead": "shared",
            "share_grad": True,
            "contrastive_weight": 0.0,
            "train_actor": True,
        },
    }
    if variant not in variants:
        raise ValueError(f"unknown two-head variant {variant}")
    recipe = variants[variant]
    qhead, share_grad, contrastive_weight, train_actor = (
        recipe[k] for k in ("qhead", "share_grad", "contrastive_weight", "train_actor")
    )

    config = base_crl_config(bc_alpha=bc_alpha, discount=discount)
    config["agent_name"] = variant
    config["cosine_temperature"] = 0.1
    config["expectile"] = 0.9
    config["tau"] = 0.005
    config["qhead"] = qhead
    config["share_grad"] = share_grad
    config["contrastive_weight"] = contrastive_weight
    config["train_actor"] = train_actor
    config["actor_loss"] = "awr"
    config["alpha"] = bc_alpha  # AWR advantage temperature (matches td_q/GCIQL)
    _apply_capacity_overrides(config, hidden_dims, latent_dim)
    config.update(model_config or {})
    with use_cosine_value_module():
        return CRLTwoHeadAgent.create(
            seed,
            example_batch["observations"],
            example_batch["actions"],
            flax.core.FrozenDict(config),
        )
