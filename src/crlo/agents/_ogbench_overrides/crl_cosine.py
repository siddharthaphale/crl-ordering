"""Cosine-normalized CRL: project phi and psi to the unit sphere.

Subclasses upstream ``CRLAgent`` and overrides ``contrastive_loss`` so that the
inner product is taken between L2-normalized embeddings. By Cauchy-Schwarz
this bounds the score in [-1, 1] (Prop 5 / `\\eqref{eq:cos}`), removing the
unbounded-growth channel of `\\cref{thm:extrapolation}`.

The factory wraps ``create`` in :func:`use_cosine_value_module`, which swaps
the critic's value module for :class:`CosineBilinearValue` so the bounded
cosine score is what the *forward* returns -- propagating the bound to the
actor loss and ``sample_actions``, not just the training loss. Without this,
the agent trains a cosine critic but its actor maximizes the unbounded
``phi . psi``, which is internally inconsistent.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import optax

from crlo.utils.ogbench_vendor import ensure_ogbench

ensure_ogbench()
import agents.crl as _crl_mod  # noqa: E402
from agents.crl import CRLAgent  # noqa: E402
from utils.networks import GCBilinearValue  # noqa: E402

from crlo.agents.embeddings import normalize_embeddings


class CosineBilinearValue(GCBilinearValue):
    """GCBilinearValue whose forward value is the *cosine* similarity.

    The base module returns the unnormalized ``(phi . psi)/sqrt(d)``. This
    subclass L2-normalizes phi and psi first, so the value the actor and
    ``sample_actions`` consume is the bounded cosine score in [-1, 1] --
    matching what ``CRLCosineAgent.contrastive_loss`` trains. ``info=True``
    still returns the raw phi/psi so the loss path is unchanged (it
    re-normalizes them itself). Scale (1/tau) is omitted: the DDPG+BC actor
    loss is scale-invariant and candidate ranking is monotone in the score.
    """

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(goals)
        v = (normalize_embeddings(phi) * normalize_embeddings(psi)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        if info:
            return v, phi, psi
        return v


@contextlib.contextmanager
def use_cosine_value_module():
    """Patch ``agents.crl.GCBilinearValue`` -> :class:`CosineBilinearValue`.

    ``CRLAgent.create`` resolves ``GCBilinearValue`` from its own module
    namespace, so swapping that name for the duration of ``create`` makes the
    constructed critic (and the unused AWR value) use the cosine-normalized
    forward. This propagates the bound to the *actor* and ``sample_actions`` --
    not just the eval-time ``critic_score`` -- so the whole agent is
    internally consistent. The param tree is unchanged (only ``__call__``
    differs), so existing checkpoints still load.
    """
    orig = _crl_mod.GCBilinearValue
    _crl_mod.GCBilinearValue = CosineBilinearValue
    try:
        yield
    finally:
        _crl_mod.GCBilinearValue = orig


class CRLCosineAgent(CRLAgent):
    """CRL with cosine-normalized phi/psi embeddings."""

    def contrastive_loss(self, batch, grad_params, module_name="critic"):
        batch_size = batch["observations"].shape[0]
        actions = batch["actions"] if module_name == "critic" else None
        v, phi, psi = self.network.select(module_name)(
            batch["observations"],
            batch["value_goals"],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        # === Cosine normalization (the override) ===
        phi = normalize_embeddings(phi)
        psi = normalize_embeddings(psi)
        # Logits are now in [-1, 1]; we drop the 1/sqrt(d) scaling because the
        # cosine score is already bounded. Temperature 1/tau scales the loss.
        tau = self.config["cosine_temperature"]
        logits = jnp.einsum("eik,ejk->ije", phi, psi) / tau
        eye = jnp.eye(batch_size)

        def binary_cross_entropy(lg):
            return optax.sigmoid_binary_cross_entropy(logits=lg, labels=eye)

        contrastive_loss = jax.vmap(binary_cross_entropy, in_axes=-1, out_axes=-1)(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        v = jnp.exp(v)
        logits_mean = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits_mean, axis=1) == jnp.argmax(eye, axis=1)
        logits_pos = jnp.sum(logits_mean * eye) / jnp.sum(eye)
        logits_neg = jnp.sum(logits_mean * (1 - eye)) / jnp.sum(1 - eye)
        return contrastive_loss, {
            "contrastive_loss": contrastive_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
            "binary_accuracy": jnp.mean((logits_mean > 0) == eye),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits": logits_mean.mean(),
            "phi_norm_mean": jnp.linalg.norm(phi, axis=-1).mean(),
            "cosine_normalize": jnp.array(True),
        }
