"""Frozen executed-value rollout-cell mechanics: CRN keys and exact state restoration.

Implements the frozen common-random-numbers mapping (plan §5.4) and the
locomaze restore primitive (plan §5.5). The DDPM continuation itself is
driven by the caller (``experiments/evaluate_candidate_returns.py``) using
the tested sequential ``policy.sample(ob, g, 1)`` path: resetting
``policy.rng`` to the cell key before each rollout gives every candidate
in a ``(query, half, rollout)`` cell an identical noise stream at each
continuation step index, because the stream's evolution is a
deterministic function of the key and the number of ``sample`` calls,
never of the observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cell_key(rollout_seed: int, query_id: int, half_id: int, rollout_id: int):
    """Frozen JAX cell key: fold_in(fold_in(fold_in(seed, q), h), r)."""
    import jax  # noqa: PLC0415  (keep this module importable without jax)

    key = jax.random.PRNGKey(rollout_seed)
    for word in (query_id, half_id, rollout_id):
        key = jax.random.fold_in(key, word)
    return key


class StochasticDiffusionContinuation:
    """Jitted drop-in for ``DiffusionQLBehavior.sample`` (execution-only).

    Wraps the frozen checkpoint's ``p_sample_loop`` in one ``jax.jit``
    closure. RNG semantics are bit-identical to the eager path — one
    ``jax.random.split`` of ``self.rng`` per call, the subkey consumed by
    the same ``p_sample_loop`` — so the frozen CRN cell-key mapping and
    the noise-stream evolution are unchanged; only compiler fusion may
    perturb float ops (the cross-device numerics case the plan already
    covers). ~50x faster per continuation step than eager dispatch.
    """

    def __init__(self, policy: Any) -> None:
        import jax  # noqa: PLC0415

        from crlo.agents._diffusion_ql import ddpm_sampler  # noqa: PLC0415

        self._policy = policy
        self.rng = policy.rng
        apply_fn = policy.network.apply
        action_dim = policy.action_dim

        def _loop(params, s, g, coeffs, rng):
            return ddpm_sampler.p_sample_loop(apply_fn, params, s, g, coeffs, rng, action_dim)

        self._fn = jax.jit(_loop)

    def sample(self, s: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
        """Sample ``k`` actions for one (s, g); mirrors the eager contract."""
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        s_j = jnp.atleast_2d(jnp.asarray(s))
        g_j = jnp.atleast_2d(jnp.asarray(g))
        if s_j.shape[0] == 1:
            s_j = jnp.broadcast_to(s_j, (k,) + s_j.shape[1:])
            g_j = jnp.broadcast_to(g_j, (k,) + g_j.shape[1:])
        elif s_j.shape[0] != k:
            raise ValueError(f"batched sample needs k == batch; got {k}, {s_j.shape[0]}")
        self.rng, sub = jax.random.split(self.rng)
        return np.asarray(self._fn(self._policy.params, s_j, g_j, self._policy.coeffs, sub))


class DeterministicDiffusionContinuation:
    """Mode-path continuation for Arm B (plan §17.2): fully deterministic.

    The identical reverse process with every stochastic term at its mode:
    ``x_T = 0`` (mode of N(0, I)) and per-step noise gated to zero. The
    result is a deterministic function of (s, g) — an rng-free policy.
    ``rng`` is accepted for driver compatibility and ignored, so the two
    protocol halves become independent executions whose bit-equality is
    the Arm-B duplicate-verification check.
    """

    def __init__(self, policy: Any) -> None:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        from crlo.agents._diffusion_ql import ddpm_sampler  # noqa: PLC0415

        self._policy = policy
        self.rng = policy.rng  # ignored; kept for interface parity
        apply_fn = policy.network.apply
        action_dim = policy.action_dim

        def _mode_loop(params, s, g, coeffs):
            t_total = coeffs["betas"].shape[0]
            x = jnp.zeros((s.shape[0], action_dim))
            zero = jnp.zeros((s.shape[0], action_dim))
            for t in range(t_total - 1, -1, -1):
                t_arr = jnp.full((s.shape[0],), t, dtype=jnp.int32)
                eps_pred = apply_fn(params, x, t_arr, s, g)
                x = ddpm_sampler.p_sample_step(x, t, eps_pred, zero, coeffs)
            return x

        self._fn = jax.jit(_mode_loop)

    def sample(self, s: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
        """Deterministic mode action; identical for every call and k."""
        import jax.numpy as jnp  # noqa: PLC0415

        s_j = jnp.atleast_2d(jnp.asarray(s))
        g_j = jnp.atleast_2d(jnp.asarray(g))
        if s_j.shape[0] == 1:
            s_j = jnp.broadcast_to(s_j, (k,) + s_j.shape[1:])
            g_j = jnp.broadcast_to(g_j, (k,) + g_j.shape[1:])
        elif s_j.shape[0] != k:
            raise ValueError(f"batched sample needs k == batch; got {k}, {s_j.shape[0]}")
        return np.asarray(self._fn(self._policy.params, s_j, g_j, self._policy.coeffs))


def reset_wrapper_bookkeeping(env: Any) -> None:
    """Zero every TimeLimit-style elapsed-step counter in the wrapper stack.

    ``set_state`` rewinds physics but not the gym step budget; without
    this, later rollouts on the same env instance would truncate early
    so each candidate starts from the same simulator state.
    """
    e = env
    while e is not None:
        if hasattr(e, "_elapsed_steps"):
            e._elapsed_steps = 0
        e = getattr(e, "env", None)


def restore_state(
    env: Any, qpos: np.ndarray, qvel: np.ndarray, expected_ob: np.ndarray, atol: float, rtol: float
) -> np.ndarray:
    """Restore exact physics state and verify the observation.

    Fails closed on any observation mismatch (plan §5.5). Returns the
    restored observation.
    """
    u = env.unwrapped
    u.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
    # set_state restores qpos/qvel but NOT the solver's hidden state; the
    # warm-start accumulator would make each cell's outcome depend on the
    # previously executed cell (traversal order), violating plan §5.4/§5.5.
    # Zero it (and the clock) so every cell is a pure function of
    # (restored state, candidate, cell key).
    d = getattr(u, "data", None)
    if d is not None and hasattr(d, "qacc_warmstart"):
        d.qacc_warmstart[:] = 0.0
        d.time = 0.0
    reset_wrapper_bookkeeping(env)
    ob = np.asarray(u.get_ob())
    if not np.allclose(ob, expected_ob, atol=atol, rtol=rtol):
        max_err = float(np.max(np.abs(ob - np.asarray(expected_ob))))
        raise RuntimeError(
            f"state restoration mismatch (max abs err {max_err:.3e} > "
            f"atol={atol}, rtol={rtol}); failing closed"
        )
    return ob


def run_cell(
    env: Any,
    policy: Any,
    ob0: np.ndarray,
    goal: np.ndarray,
    goal_xy: np.ndarray,
    goal_tol: float,
    candidate: np.ndarray,
    act_lo: np.ndarray,
    act_hi: np.ndarray,
    horizon: int,
    gamma: float,
    persist: int = 1,
) -> tuple[int, int, float]:
    """Execute one candidate rollout cell after state restoration.

    Applies the (bound-clipped) candidate for up to ``persist``
    consecutive steps (success checked after every step; plan §18
    decision-scale audit; ``persist=1`` is the original protocol), then
    follows the sequential DDPM continuation until locomaze goal
    attainment or the horizon. The caller must have set ``policy.rng``
    to the frozen cell key and restored the state so that
    ``get_ob() == ob0``.

    Returns ``(success, steps_to_goal, disc_return)`` with the first
    candidate application counted as step 1, ``disc_return =
    gamma**steps`` on success and 0.0 otherwise (Y = gamma^T 1{T<=H}).
    """
    a0 = np.clip(np.asarray(candidate, np.float64), act_lo, act_hi)
    steps = 0
    success = False
    for _ in range(min(int(persist), horizon)):
        ob, *_ = env.step(a0)
        ob = np.asarray(ob)
        steps += 1
        success = bool(np.linalg.norm(ob[:2] - goal_xy) <= goal_tol)
        if success:
            break
    while not success and steps < horizon:
        a = np.asarray(policy.sample(ob.astype("float32"), np.asarray(goal, "float32"), 1))[0]
        ob, *_ = env.step(np.clip(a, act_lo, act_hi))
        ob = np.asarray(ob)
        steps += 1
        success = bool(np.linalg.norm(ob[:2] - goal_xy) <= goal_tol)
    return int(success), (steps if success else -1), (float(gamma**steps) if success else 0.0)
