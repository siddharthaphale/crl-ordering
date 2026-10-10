"""Shared base config for CRL-family agents.

The raw bilinear CRL, cosine-normalised, and hybrid critic factories all
build the same FrozenDict scaffolding (lr, hidden dims, goal-sampling
policy, actor settings, dataset config). Each factory calls
``base_crl_config`` and overrides only its own deltas (agent_name plus any
loss-specific keys like ``cosine_temperature`` / ``td_warmup_steps``).

GCIQL has a different shape (no ``latent_dim``, has ``tau`` / ``expectile``)
and stays separate in ``gciql_agent.py``.
"""

from __future__ import annotations


def base_crl_config(*, bc_alpha: float, discount: float) -> dict:
    """Default CRL-family config (CRL goal-sampling, DDPG+BC actor)."""
    return dict(
        lr=0.0003,
        batch_size=1024,
        actor_hidden_dims=(512, 512, 512),
        value_hidden_dims=(512, 512, 512),
        latent_dim=512,
        layer_norm=True,
        discount=discount,
        actor_loss="ddpgbc",
        alpha=bc_alpha,
        const_std=True,
        discrete=False,
        encoder=None,
        critic_loss="nce_binary",
        dataset_class="GCDataset",
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=False,
        p_aug=0.0,
        frame_stack=None,
    )


def _apply_capacity_overrides(
    config: dict,
    hidden_dims: tuple[int, ...] | None,
    latent_dim: int | None,
) -> dict:
    """Optionally shrink/grow a critic config's net capacity in place.

    ``None`` leaves the OGBench-default capacity untouched (correct for the
    real high-dim tasks). The 2D toy passes a small net so the critic does
    not overfit jagged surfaces. ``latent_dim`` is ignored for configs
    without that key (e.g. GCIQL).
    """
    if hidden_dims is not None:
        hd = tuple(hidden_dims)
        config["actor_hidden_dims"] = hd
        config["value_hidden_dims"] = hd
    if latent_dim is not None and "latent_dim" in config:
        config["latent_dim"] = latent_dim
    return config
