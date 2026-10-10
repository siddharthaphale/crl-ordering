"""Objective/function-class control agents.

Completes the objective × function-class 2×2 on the encoder family:

  |                | bilinear scalar          | nonlinear MLP head       |
  |----------------|--------------------------|--------------------------|
  | contrastive    | existing ``raw_bilinear``| ``contrastive_mlp`` (new)|
  | TD / Bellman   | ``td_bilinear`` (new)    | existing ``td_q`` (GCIQL)|

``td_bilinear``
    Bellman targets on the raw bilinear scalar Q(s,a,g) = φ(s,a)ᵀψ(g)/√d.
    Function class identical to ``raw_bilinear`` (same GCBilinearValue,
    ensemble of 2 = twin Q); objective identical to the two-head q_td recipe
    (IQL expectile V on raw [s,g] + Bellman Q + Polyak target, expectile 0.9,
    tau 0.005 — the repo's established TD-in-CRL-harness precedent).

``contrastive_mlp``
    NCE-binary contrastive loss on a monolithic GCValue MLP logit
    f(s,a,g) — parameter-matched to the deployed TD-Q (GCIQL twin-Q MLP on
    concat(s,g,a), same hidden dims / layer norm / ensemble of 2).
    Full in-batch B×B logits are infeasible for an MLP (B²≈1M forward rows
    per step), so each anchor uses its positive plus ``NUM_NEGATIVES``
    uniformly sampled in-batch negative goals (j != i). This is the same
    binary-NCE objective at a reduced negative count.

Both are **critic-only diagnostics** (``train_actor=False``), exactly like
the two-head diagnostic variants two_head_separate_encoders/b1/b2/b0: the fixed-query
protocol scores the critic scalar on cached DDPM pools and never consults
an actor. Deployed scores: ``td_bilinear`` = ensemble-mean raw bilinear
value (same readout as raw_bilinear); ``contrastive_mlp`` = ensemble-mean
MLP logit (same readout as td_q).
"""

from __future__ import annotations

import copy

import flax
import jax
import jax.numpy as jnp
import optax

from crlo.utils.ogbench_vendor import ensure_ogbench

ensure_ogbench()
import agents.crl as _crl_mod  # noqa: E402
from utils.flax_utils import ModuleDict, TrainState  # noqa: E402
from utils.networks import GCValue  # noqa: E402

from crlo.agents._ogbench_overrides.crl_twohead import (  # noqa: E402
    CRLTwoHeadAgent,
)

OBJECTIVE_CONTROL_VARIANTS = ("td_bilinear", "contrastive_mlp")


class TDBilinearAgent(CRLTwoHeadAgent):
    """Bellman-trained raw bilinear scalar (function class of raw_bilinear,
    objective of the two-head q_td recipe)."""

    def _q(self, obs, goals, actions, grad_params=None, target=False):
        module = "target_critic" if target else "critic"
        params = None if target else grad_params
        return self.network.select(module)(obs, goals, actions=actions, params=params)

    def value_loss(self, batch, grad_params):
        q1, q2 = self._q(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            target=True,
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("v_td")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        diff = q - v
        loss = self.expectile_loss(diff, diff, self.config["expectile"]).mean()
        return loss, {"value_loss": loss, "v_mean": v.mean()}

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select("v_td")(batch["next_observations"], batch["value_goals"])
        target = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v
        q1, q2 = self._q(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            grad_params=grad_params,
        )
        loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        return loss, {"critic_td_loss": loss, "q_mean": ((q1 + q2) / 2).mean()}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        vloss, vinfo = self.value_loss(batch, grad_params)
        for k, v in vinfo.items():
            info[f"value/{k}"] = v
        qloss, qinfo = self.critic_loss(batch, grad_params)
        for k, v in qinfo.items():
            info[f"critic_td/{k}"] = v
        loss = vloss + qloss
        info["total_loss"] = loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_goals = ex_observations

        # Deployed scalar: raw bilinear critic, identical to raw_bilinear's
        # GCBilinearValue (ensemble of 2 doubles as the twin Q).
        critic_def = _crl_mod.GCBilinearValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            value_exp=False,
            state_encoder=None,
            goal_encoder=None,
        )
        # IQL scaffolding: action-free V on raw [s, g] (never deployed).
        v_td_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            v_td=(v_td_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        params["modules_target_critic"] = params["modules_critic"]
        network = TrainState.create(network_def, params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


class ContrastiveMLPAgent(CRLTwoHeadAgent):
    """Binary-NCE contrastive loss on a monolithic GCValue MLP logit,
    parameter-matched to the deployed TD-Q (GCIQL twin-Q architecture)."""

    def contrastive_mlp_loss(self, batch, grad_params, rng):
        obs, goals, actions = (
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
        )
        b = obs.shape[0]
        k = self.config["num_negatives"]
        # per-anchor negative goals: k uniform in-batch indices j != i
        neg = jax.random.randint(rng, (b, k), 0, b - 1)
        rows = jnp.arange(b)[:, None]
        neg = jnp.where(neg >= rows, neg + 1, neg)  # skip j == i
        idx = jnp.concatenate([rows, neg], axis=1)  # (b, 1+k), col 0 = positive
        goals_all = goals[idx.reshape(-1)]  # (b*(1+k), gdim)
        obs_all = jnp.repeat(obs, 1 + k, axis=0)
        act_all = jnp.repeat(actions, 1 + k, axis=0)
        logits = self.network.select("critic")(
            obs_all, goals_all, actions=act_all, params=grad_params
        )  # (2, b*(1+k))
        logits = logits.reshape(2, b, 1 + k)
        labels = jnp.zeros((b, 1 + k)).at[:, 0].set(1.0)
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels[None]).mean()
        mean_logits = logits.mean(0)
        return loss, {
            "contrastive_loss": loss,
            "binary_accuracy": jnp.mean((mean_logits > 0) == labels),
            "logits_pos": mean_logits[:, 0].mean(),
            "logits_neg": mean_logits[:, 1:].mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        loss, cinfo = self.contrastive_mlp_loss(batch, grad_params, rng)
        for k, v in cinfo.items():
            info[f"contrastive/{k}"] = v
        info["total_loss"] = loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_goals = ex_observations

        # Parameter-matched to TD-Q: GCValue twin MLP on concat(s, g, a),
        # same hidden dims / layer norm / ensemble of 2 as GCIQL's critic.
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
