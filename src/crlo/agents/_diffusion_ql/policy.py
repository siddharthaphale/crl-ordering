"""JAX DDPM behavior policy with a negative denoising-loss diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from crlo.agents._diffusion_ql import ddpm_sampler, noise_schedule, train
from crlo.agents._diffusion_ql.score_network import ScoreNetwork


@dataclass
class DiffusionQLBehavior:
    """JAX Diffusion-QL behavior policy.

    Attributes:
        action_dim: action-space dimension (satisfies the protocol).
        state_dim: state-space dimension used to build the network.
        goal_dim: goal-space dimension.
        T: number of denoising steps.
        params: Flax parameters of ``ScoreNetwork``.
        opt_state: optimizer state.
        optimizer: optax transformation.
        network: ``ScoreNetwork`` instance (apply via ``network.apply``).
        coeffs: pre-computed schedule coefficients.
        rng: PRNG key advanced on each ``sample``.
    """

    action_dim: int
    state_dim: int
    goal_dim: int
    T: int
    params: Any
    opt_state: Any
    optimizer: optax.GradientTransformation
    network: ScoreNetwork
    coeffs: dict[str, jnp.ndarray]
    rng: jax.Array
    lr: float
    schedule: str
    steps: int = 0

    @classmethod
    def create(
        cls,
        seed: int,
        action_dim: int,
        state_dim: int = 0,
        goal_dim: int = 0,
        denoising_steps: int | None = None,
        hidden_dims: tuple[int, ...] | None = None,
        lr: float | None = None,
        schedule: str | None = None,
        time_embed_dim: int | None = None,
    ) -> DiffusionQLBehavior:
        denoising_steps = 20 if denoising_steps is None else denoising_steps
        hidden_dims = (256, 256) if hidden_dims is None else tuple(hidden_dims)
        lr = 0.0003 if lr is None else lr
        schedule = "cosine" if schedule is None else schedule
        time_embed_dim = 32 if time_embed_dim is None else time_embed_dim
        if time_embed_dim < 2 or time_embed_dim % 2:
            raise ValueError("time_embed_dim must be a positive even width")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        network = ScoreNetwork(
            action_dim=action_dim, hidden_dims=hidden_dims, time_embed_dim=time_embed_dim
        )
        a_dummy = jnp.zeros((1, action_dim))
        t_dummy = jnp.zeros((1,), dtype=jnp.int32)
        s_dummy = jnp.zeros((1, max(state_dim, 1)))
        g_dummy = jnp.zeros((1, max(goal_dim, 1)))
        params = network.init(init_rng, a_dummy, t_dummy, s_dummy, g_dummy)
        if schedule == "cosine":
            betas = noise_schedule.cosine_beta_schedule(denoising_steps)
        elif schedule == "linear":
            betas = noise_schedule.linear_beta_schedule(denoising_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        coeffs = noise_schedule.schedule_coefficients(betas)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        return cls(
            action_dim=action_dim,
            state_dim=state_dim,
            goal_dim=goal_dim,
            T=denoising_steps,
            params=params,
            opt_state=opt_state,
            optimizer=optimizer,
            network=network,
            coeffs=coeffs,
            rng=rng,
            lr=lr,
            schedule=schedule,
        )

    def update(self, batch: dict[str, jnp.ndarray]) -> dict[str, float]:
        """One DDPM training step on ``batch``."""
        step_fn = self._get_train_step()
        self.rng, sub = jax.random.split(self.rng)
        self.params, self.opt_state, loss, info = step_fn(self.params, self.opt_state, batch, sub)
        self.steps += 1
        return {k: float(v) for k, v in info.items()}

    def _get_train_step(self):
        """Build the jitted train step once, then reuse it.

        ``make_train_step`` wraps a *fresh* ``@jax.jit`` closure on every
        call, so invoking it inside ``update`` (i.e. once per step) created a
        new function object each step, defeated the jit cache, and forced a
        full XLA recompile per step — catastrophic at real scale. Cache the
        compiled step on the instance so it is built exactly once.
        """
        fn = getattr(self, "_train_step_fn", None)
        if fn is None:
            fn = train.make_train_step(self.network.apply, self.optimizer, self.coeffs, self.T)
            self._train_step_fn = fn
        return fn

    def sample(self, s: np.ndarray, g: np.ndarray, K: int) -> np.ndarray:
        """Sample K candidate actions for a single (s, g) query.

        ``s``, ``g`` may be 1D (single query) or 2D (batched). When 1D, K
        independent draws are returned, all conditioned on the same (s, g).
        """
        s_j = jnp.atleast_2d(jnp.asarray(s))
        g_j = jnp.atleast_2d(jnp.asarray(g))
        if s_j.shape[0] == 1:
            s_j = jnp.broadcast_to(s_j, (K,) + s_j.shape[1:])
            g_j = jnp.broadcast_to(g_j, (K,) + g_j.shape[1:])
        elif s_j.shape[0] != K:
            raise ValueError(
                f"Batched sample requires K == batch size; got K={K}, batch={s_j.shape[0]}"
            )
        self.rng, sub = jax.random.split(self.rng)
        out = ddpm_sampler.p_sample_loop(
            self.network.apply,
            self.params,
            s_j,
            g_j,
            self.coeffs,
            sub,
            self.action_dim,
        )
        return np.asarray(out)

    def negative_denoising_loss(self, a: np.ndarray, s: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Return negative mean squared noise-prediction error at t=T//2.

        This stochastic diagnostic is not a log density or a likelihood bound.
        Each invocation advances the policy RNG. Units are squared noise error.
        """
        a_j = jnp.atleast_2d(jnp.asarray(a))
        s_j = jnp.atleast_2d(jnp.asarray(s))
        g_j = jnp.atleast_2d(jnp.asarray(g))
        N = a_j.shape[0]
        if s_j.shape[0] == 1:
            s_j = jnp.broadcast_to(s_j, (N,) + s_j.shape[1:])
            g_j = jnp.broadcast_to(g_j, (N,) + g_j.shape[1:])
        t_mid = jnp.full((N,), self.T // 2, dtype=jnp.int32)
        self.rng, sub = jax.random.split(self.rng)
        eps = jax.random.normal(sub, a_j.shape)
        a_t = ddpm_sampler.q_sample(a_j, t_mid, eps, self.coeffs)
        eps_pred = self.network.apply(self.params, a_t, t_mid, s_j, g_j)
        per_sample = -jnp.mean((eps - eps_pred) ** 2, axis=-1)
        return np.asarray(per_sample)
