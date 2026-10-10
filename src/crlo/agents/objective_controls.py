"""Objective/function-class control factory.

Two new cells completing the objective × function-class 2×2:
``td_bilinear`` (Bellman on the raw bilinear scalar) and
``contrastive_mlp`` (binary NCE on a TD-Q-parameter-matched MLP logit).
See objective_control_critics.py for the design; both are critic-only diagnostics.
"""

from __future__ import annotations

from typing import Any

from crlo.agents._base_config import _apply_capacity_overrides, base_crl_config
from crlo.utils.ogbench_vendor import ensure_ogbench


def make_objective_control(
    seed: int,
    example_batch: dict,
    bc_alpha: float,
    discount: float,
    variant: str = "td_bilinear",
    model_config: dict | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    latent_dim: int | None = None,
) -> Any:
    """Instantiate an objective-control agent for ``variant``."""
    ensure_ogbench()
    import flax  # noqa: PLC0415

    from crlo.agents._ogbench_overrides.objective_control_critics import (  # noqa: PLC0415
        OBJECTIVE_CONTROL_VARIANTS,
        ContrastiveMLPAgent,
        TDBilinearAgent,
    )

    if variant not in OBJECTIVE_CONTROL_VARIANTS:
        raise ValueError(f"unknown objective-control variant {variant}")

    config = base_crl_config(bc_alpha=bc_alpha, discount=discount)
    config["agent_name"] = variant
    config["train_actor"] = False
    if variant == "td_bilinear":
        config["expectile"] = 0.9
        config["tau"] = 0.005
        cls = TDBilinearAgent
    else:
        config["num_negatives"] = 127
        cls = ContrastiveMLPAgent
    _apply_capacity_overrides(config, hidden_dims, latent_dim)
    config.update(model_config or {})
    return cls.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        flax.core.FrozenDict(config),
    )
