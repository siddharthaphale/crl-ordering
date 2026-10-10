"""One construction boundary for training and checkpoint reconstruction."""

from __future__ import annotations

import numpy as np


def build_critic(
    critic: str,
    seed: int,
    example_batch: dict,
    bc_alpha: float,
    discount: float,
    hidden_dims: tuple[int, ...] | None = None,
    latent_dim: int | None = None,
    controlled: bool = False,
    model_config: dict | None = None,
):
    """Dispatch to the right agent factory for ``critic``."""
    if critic == "raw_bilinear":
        from crlo.agents.crl_agent import make_crl_agent

        return make_crl_agent(
            seed,
            example_batch,
            bc_alpha,
            discount,
            model_config=model_config,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
    if critic == "cosine":
        from crlo.agents.crl_cosine_agent import make_crl_cosine_agent

        return make_crl_cosine_agent(
            seed,
            example_batch,
            bc_alpha,
            discount,
            model_config=model_config,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
    if critic == "td_q":
        from crlo.agents.gciql_agent import make_gciql_agent

        return make_gciql_agent(
            seed,
            example_batch,
            bc_alpha,
            discount,
            model_config=model_config,
            hidden_dims=hidden_dims,
            **(
                {
                    "value_p_curgoal": 0.0,
                    "value_p_trajgoal": 1.0,
                    "value_p_randomgoal": 0.0,
                    "gc_negative": False,
                }
                if controlled
                else {}
            ),
        )
    if critic == "hybrid":
        from crlo.agents.hybrid_critic_agent import make_hybrid_critic_agent

        return make_hybrid_critic_agent(
            seed,
            example_batch,
            bc_alpha,
            discount,
            model_config=model_config,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
    if critic.startswith("two_head_"):
        from crlo.agents.two_head_critic import make_two_head_critic

        return make_two_head_critic(
            seed,
            example_batch,
            bc_alpha,
            discount,
            variant=critic,
            model_config=model_config,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
    if critic in ("td_bilinear", "contrastive_mlp"):
        from crlo.agents.objective_controls import make_objective_control

        return make_objective_control(
            seed,
            example_batch,
            bc_alpha,
            discount,
            variant=critic,
            model_config=model_config,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
    raise ValueError(f"unknown critic {critic}")


def build_critic_from_metadata(meta: dict):
    """Reconstruct the architecture recorded by a checkpoint."""
    return build_critic(
        meta["critic"],
        seed=meta.get("seed", 0),
        example_batch={
            "observations": np.zeros((1, meta["obs_dim"]), np.float32),
            "actions": np.zeros((1, meta["action_dim"]), np.float32),
        },
        bc_alpha=meta["bc_alpha"],
        discount=meta["discount"],
        hidden_dims=meta.get("hidden_dims"),
        latent_dim=meta.get("latent_dim"),
        controlled=meta.get("controlled", False),
        model_config=meta.get("agent_config"),
    )


def critic_rebuild_function(critic: str):
    """Bind the declared model type for legacy metadata without a critic field."""

    def rebuild(meta):
        return build_critic_from_metadata({**meta, "critic": critic})

    return rebuild
