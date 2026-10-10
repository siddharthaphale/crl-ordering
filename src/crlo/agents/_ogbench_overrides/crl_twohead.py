"""Two-head contrastive representation and independent scalar value readouts.

Recipes declare separate or shared encoders, TD-gradient routing,
contrastive weight, and whether an advantage-weighted actor is trained.
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
from utils.networks import GCActor, GCValue  # noqa: E402

from crlo.agents._ogbench_overrides.crl_cosine import (  # noqa: E402
    CRLCosineAgent,
    normalize_embeddings,
)


class CRLTwoHeadAgent(CRLCosineAgent):
    """CRL cosine embeddings + a free scalar IQL `q_td` selection head."""

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * (diff**2)

    def _q_feat(self, obs, goals, actions, grad_params, stop):
        """[phi_hat(s,a), psi_hat(g)] from the contrastive critic (shared heads)."""
        params = None if stop else grad_params
        _, phi, psi = self.network.select("critic")(
            obs, goals, actions=actions, info=True, params=params
        )
        phi = normalize_embeddings(phi)
        psi = normalize_embeddings(psi)
        if phi.ndim == 3:  # (E,B,d) -> mean over the contrastive ensemble
            phi = phi.mean(0)
            psi = psi.mean(0)
        feat = jnp.concatenate([phi, psi], axis=-1)
        if stop:
            feat = jax.lax.stop_gradient(feat)
        return feat

    def _q_td(self, obs, goals, actions, grad_params=None, target=False):
        """Scalar Q from the deployed head (ensemble -> (2,B))."""
        module = "target_q_td" if target else "q_td"
        sel_params = None if target else grad_params
        if self.config["qhead"] == "raw":
            return self.network.select(module)(obs, goals, actions, params=sel_params)
        stop = target or (not self.config["share_grad"])
        feat = self._q_feat(obs, goals, actions, grad_params, stop)
        return self.network.select(module)(feat, params=sel_params)

    def value_loss(self, batch, grad_params):
        q1, q2 = self._q_td(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            grad_params=None,
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
        q1, q2 = self._q_td(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            grad_params=grad_params,
            target=False,
        )
        loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        return loss, {"critic_td_loss": loss, "q_mean": ((q1 + q2) / 2).mean()}

    def actor_loss(self, batch, grad_params):
        """AWR policy head (option B). adv = q_td - v_td is a *fixed* signal
        (value heads called without grad_params, GCIQL-style), so only the
        actor params are differentiated. q_td is the deployed selection scalar.
        """
        q1, q2 = self._q_td(
            batch["observations"],
            batch["actor_goals"],
            batch["actions"],
            grad_params=None,
            target=False,
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("v_td")(batch["observations"], batch["actor_goals"])
        adv = q - v
        exp_a = jnp.minimum(jnp.exp(adv * self.config["alpha"]), 100.0)
        dist = self.network.select("actor")(
            batch["observations"], batch["actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        loss = -(exp_a * log_prob).mean()
        return loss, {
            "actor_loss": loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        cw = self.config["contrastive_weight"]
        contrastive = 0.0
        if cw > 0:
            contrastive, cinfo = self.contrastive_loss(batch, grad_params, "critic")
            for k, v in cinfo.items():
                info[f"contrastive/{k}"] = v
        vloss, vinfo = self.value_loss(batch, grad_params)
        for k, v in vinfo.items():
            info[f"value/{k}"] = v
        qloss, qinfo = self.critic_loss(batch, grad_params)
        for k, v in qinfo.items():
            info[f"critic_td/{k}"] = v
        loss = cw * contrastive + vloss + qloss
        if self.config.get("train_actor", False):
            aloss, ainfo = self.actor_loss(batch, grad_params)
            for k, v in ainfo.items():
                info[f"actor/{k}"] = v
            loss = loss + aloss
        info["total_loss"] = loss
        return loss, info

    def target_update(self, network, module_name):
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "q_td")
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        # Contrastive critic: GCBilinearValue, rebound to CosineBilinearValue by
        # use_cosine_value_module() in the factory (bounded forward).
        critic_def = _crl_mod.GCBilinearValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            value_exp=False,
            state_encoder=None,
            goal_encoder=None,
        )
        actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config["const_std"],
            gc_encoder=None,
        )
        # v_td: action-free V on raw [s,g] (IQL scaffolding; never deployed).
        v_td_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
        )
        # q_td: free scalar Q (ensemble of 2). raw -> [s,a,g]; shared -> feature MLP.
        q_td_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
        )
        if config["qhead"] == "raw":
            q_in = (ex_observations, ex_goals, ex_actions)
        else:
            ex_feat = jnp.zeros((ex_observations.shape[0], 2 * config["latent_dim"]), jnp.float32)
            q_in = (ex_feat,)

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            v_td=(v_td_def, (ex_observations, ex_goals)),
            q_td=(q_td_def, q_in),
            target_q_td=(copy.deepcopy(q_td_def), q_in),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        params["modules_target_q_td"] = params["modules_q_td"]
        network = TrainState.create(network_def, params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
