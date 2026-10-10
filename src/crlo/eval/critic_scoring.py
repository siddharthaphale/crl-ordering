"""Compute the deployed scalar used to order candidate actions.

Raw and TD bilinear scores use ``phi . psi / sqrt(latent_dim)``; scalar
TD-Q and contrastive-MLP scores average their twin heads. Cosine uses normalized
embeddings divided by the supplied temperature, while hybrid uses unscaled
cosine. Two-head critics deploy the free scalar TD head; their ``_cos`` readout
uses temperature-scaled cosine. All ensemble outputs are averaged.

Cosine value modules already normalize their forward scalar. This boundary
recomputes cosine from the raw embeddings returned by ``info=True`` to keep
evaluation scaling explicit, including for historical checkpoints.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from crlo.agents.embeddings import normalize_embeddings


def critic_score(
    agent: Any,
    critic: str,
    s: np.ndarray,
    g: np.ndarray,
    a: np.ndarray,
    cosine_temperature: float = 0.1,
) -> np.ndarray:
    """Return deployed critic scores for aligned state-goal-action triples.

    Args:
        agent: Constructed agent whose network matches the requested readout.
        critic: Critic name from ``CRITICS``, or a supported embedding readout
            with the ``_cos`` suffix. Two-head names without that suffix use
            the free TD head.
        s: States of shape ``(n, observation_dim)``.
        g: Goals of shape ``(n, goal_dim)`` aligned with states.
        a: Actions of shape ``(n, action_dim)`` aligned with states.
        cosine_temperature: Positive divisor for ``cosine`` and two-head
            ``_cos`` scores. Supplied explicitly; not read from agent metadata.
            Other readouts ignore it.

    Returns:
        NumPy array of shape ``(n,)`` in input order, with ensemble heads
        averaged. Scores remain on each critic's native scale; no ranking,
        tie breaking, or RNG advancement occurs.

    Raises:
        ValueError: The critic/readout name is unknown.
    """
    from crlo.eval.config_schemas import CRITICS

    if critic not in CRITICS and not (critic.endswith("_cos") and critic[:-4] in CRITICS):
        raise ValueError(f"Unknown critic readout: {critic}")
    sj, gj, aj = jnp.asarray(s), jnp.asarray(g), jnp.asarray(a)
    if critic in ("td_q", "contrastive_mlp"):
        # monolithic twin-MLP logit on concat(s, g, a); deployed score is the
        # ensemble mean (contrastive_mlp is the parameter-matched objective control)
        q1, q2 = agent.network.select("critic")(sj, gj, aj)
        out = (np.asarray(q1) + np.asarray(q2)) / 2.0
        return out.reshape(-1)
    if critic.startswith("two_head_") and critic.endswith("_cos"):
        # Contrastive head of a two-head checkpoint: bounded cosine readout on
        # its CRL embeddings (same checkpoint, different deployed scalar).
        v, phi, psi = agent.network.select("critic")(sj, gj, actions=aj, info=True)
        score = np.asarray(jnp.sum(normalize_embeddings(phi) * normalize_embeddings(psi), axis=-1))
        score = score / cosine_temperature
        return (score.mean(0) if score.ndim == 2 else score).reshape(-1)
    if critic.startswith("two_head_"):
        # Deployed two-head score is the free scalar q_td head (the IQL value
        # baseline is training-only scaffolding, never used for selection).
        q = np.asarray(agent._q_td(sj, gj, aj, grad_params=None, target=False))
        return (q.mean(0) if q.ndim == 2 else q).reshape(-1)
    v, phi, psi = agent.network.select("critic")(sj, gj, actions=aj, info=True)
    if critic in ("raw_bilinear", "td_bilinear"):
        # raw bilinear scalar phi(s,a)^T psi(g)/sqrt(d), ensemble mean
        # (td_bilinear is the Bellman-trained bilinear control)
        v = np.asarray(v)
        return (v.mean(0) if v.ndim == 2 else v).reshape(-1)
    # cosine / hybrid: score on L2-normalized embeddings (matches training).
    score = jnp.sum(normalize_embeddings(phi) * normalize_embeddings(psi), axis=-1)
    if critic == "cosine":
        score = score / cosine_temperature
    score = np.asarray(score)
    return (score.mean(0) if score.ndim == 2 else score).reshape(-1)
