r"""Cosine-normalised CRL agent factory.

Replaces the raw bilinear $f = \\phi^\\top \\psi$ with its cosine-normalised
counterpart $f_\\text{cos} = \\hat\\phi^\\top \\hat\\psi$ where the
embeddings are projected to the unit sphere before the inner product. By
Cauchy-Schwarz this gives $|f_\\text{cos}| \\le 1$, removing the unbounded
growth channel of \\cref{thm:extrapolation}.

The factory mirrors ``make_crl_agent`` but instantiates ``CRLCosineAgent``
from ``crlo.agents._ogbench_overrides`` (which overrides ``contrastive_loss``
to L2-normalise phi/psi before the inner product).
"""

from __future__ import annotations

from typing import Any

from crlo.agents._base_config import _apply_capacity_overrides, base_crl_config
from crlo.utils.ogbench_vendor import ensure_ogbench


def make_crl_cosine_agent(
    seed: int,
    example_batch: dict,
    bc_alpha: float,
    discount: float,
    model_config: dict | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    latent_dim: int | None = None,
) -> Any:
    """Instantiate the cosine-normalised CRL agent.

    ``hidden_dims``/``latent_dim`` override net capacity (``None`` = OGBench
    default); the 2D toy passes a small net to avoid jagged surfaces.
    """
    ensure_ogbench()
    import flax  # noqa: PLC0415

    from crlo.agents._ogbench_overrides.crl_cosine import (  # noqa: PLC0415
        CRLCosineAgent,
        use_cosine_value_module,
    )

    config = base_crl_config(bc_alpha=bc_alpha, discount=discount)
    config["agent_name"] = "crl_cosine"
    config["cosine_temperature"] = 0.1
    _apply_capacity_overrides(config, hidden_dims, latent_dim)
    config.update(model_config or {})
    with use_cosine_value_module():
        return CRLCosineAgent.create(
            seed,
            example_batch["observations"],
            example_batch["actions"],
            flax.core.FrozenDict(config),
        )
