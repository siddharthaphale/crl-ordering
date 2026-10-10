"""Cosine critic trained with contrastive and Bellman objectives.

The Bellman mean-squared error is added after the contrastive warmup. Both
objectives act on the same cosine embeddings; selection uses the cosine score.
The bootstrap and random-key conventions preserve the historical implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crlo.agents._ogbench_overrides.crl_cosine import (
    CRLCosineAgent,
    normalize_embeddings,
)
from crlo.utils.ogbench_vendor import ensure_ogbench

ensure_ogbench()
from agents.crl import CRLAgent  # noqa: E402


class CRLHybridAgent(CRLCosineAgent):
    """Add a warmed-up Bellman objective to the cosine critic."""

    def td_loss(self, batch, grad_params):
        """Single-step Bellman residual on the cosine score."""
        s, g, a = batch["observations"], batch["value_goals"], batch["actions"]
        s_next = batch["next_observations"]
        r = batch["rewards"]
        gamma = self.config["discount"]
        # Forward through the contrastive critic to get phi(s,a), psi(g).
        _, phi, psi = self.network.select("critic")(s, g, actions=a, info=True, params=grad_params)
        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        phi = normalize_embeddings(phi)
        psi = normalize_embeddings(psi)
        # Bootstrap target uses the same (next_obs, goal) with sampled next a.
        # For DDPG+BC, the actor proposes the next action.
        next_dist = self.network.select("actor")(s_next, g)
        a_next = next_dist.sample(seed=jax.random.PRNGKey(0))
        _, phi_next, psi_next = self.network.select("critic")(
            s_next, g, actions=a_next, info=True, params=grad_params
        )
        if len(phi_next.shape) == 2:
            phi_next = phi_next[None, ...]
            psi_next = psi_next[None, ...]
        phi_next = normalize_embeddings(phi_next)
        psi_next = normalize_embeddings(psi_next)
        q = jnp.einsum("eik,eik->ei", phi, psi).mean(axis=0)
        q_next = jnp.einsum("eik,eik->ei", phi_next, psi_next).mean(axis=0)
        target = r + gamma * jax.lax.stop_gradient(q_next)
        td_residual = q - target
        td_mse = jnp.mean(td_residual**2)
        return td_mse, {
            "td_loss": td_mse,
            "q_mean": q.mean(),
            "target_mean": target.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None):
        # Reuse upstream total_loss for contrastive + actor; add TD.
        upstream_loss, upstream_info = CRLAgent.total_loss(self, batch, grad_params, rng=rng)
        # Apply TD only after warmup so cosine embeddings stabilize first.
        # The step counter lives on the TrainState, not in info.
        warmup = self.config["td_warmup_steps"]
        td_weight = jnp.where(self.network.step < warmup, 0.0, 1.0)
        td_mse, td_info = self.td_loss(batch, grad_params)
        loss = upstream_loss + td_weight * td_mse
        info = {**upstream_info, **td_info, "td_weight": td_weight}
        return loss, info
