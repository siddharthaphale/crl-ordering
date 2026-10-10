"""GCIQL agent factory: TD-Q critic + DDPG+BC actor."""

from __future__ import annotations

from typing import Any

from crlo.utils.ogbench_vendor import ensure_ogbench


def make_gciql_agent(
    seed: int,
    example_batch: dict,
    alpha: float = 1.0,
    discount: float = 0.99,
    value_p_curgoal: float = None,
    value_p_trajgoal: float = None,
    value_p_randomgoal: float = None,
    gc_negative: bool = True,
    model_config: dict | None = None,
    hidden_dims: tuple[int, ...] | None = None,
) -> Any:
    """Instantiate GCIQL agent with the published OGBench hyperparameters.

    Goal-sampling defaults match GCIQL's published baseline (mixed
    cur/traj/random with ``gc_negative=True``); the toy and CRL-mirroring
    drivers override these via the call-site arguments. ``hidden_dims``
    overrides the actor/value net widths (``None`` = OGBench default); the 2D
    toy passes a small net to avoid jagged surfaces. GCIQL has no latent_dim.
    """
    if value_p_curgoal is None:
        value_p_curgoal = 0.2
    if value_p_trajgoal is None:
        value_p_trajgoal = 0.5
    if value_p_randomgoal is None:
        value_p_randomgoal = 0.3
    ensure_ogbench()
    import flax  # noqa: PLC0415
    from agents.gciql import GCIQLAgent  # noqa: PLC0415

    hd = tuple(hidden_dims) if hidden_dims is not None else (512, 512, 512)
    config = flax.core.FrozenDict(
        agent_name="gciql",
        lr=0.0003,
        batch_size=1024,
        actor_hidden_dims=hd,
        value_hidden_dims=hd,
        layer_norm=True,
        discount=discount,
        tau=0.005,
        expectile=0.9,
        actor_loss="ddpgbc",
        alpha=alpha,
        const_std=True,
        discrete=False,
        encoder=None,
        dataset_class="GCDataset",
        value_p_curgoal=value_p_curgoal,
        value_p_trajgoal=value_p_trajgoal,
        value_p_randomgoal=value_p_randomgoal,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=gc_negative,
        p_aug=0.0,
        frame_stack=None,
    )
    config = flax.core.FrozenDict(dict(config) | (model_config or {}))
    return GCIQLAgent.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )
