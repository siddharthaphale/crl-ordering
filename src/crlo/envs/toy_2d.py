"""Two-dimensional controlled goal-reaching environment.

State, action, goal in R^2. Dynamics: s' = s + a + epsilon. Reward: -||s' - g||.
Offline data manifold is a disk of radius ``support_radius`` around the origin in
action space, sampled with optional goal-aligned bias.

The environment is small enough that Q* can be computed by dense evaluation on a
grid in action space at a fixed (s, g).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Toy2DConfig:
    """Physical parameters of the controlled goal-reaching environment."""

    noise_std: float = 0.0
    support_radius: float = 0.4
    action_low: float = -1.0
    action_high: float = 1.0
    state_low: float = -2.0
    state_high: float = 2.0

    def __post_init__(self):
        """Validate noise and offline support."""
        if self.noise_std < 0 or self.support_radius <= 0:
            raise ValueError("Noise must be nonnegative and support radius positive")


class Toy2DEnv:
    """Deterministic-API 2D goal-reaching env.

    Step semantics are deterministic given the action and the noise RNG; reset
    samples a state and goal uniformly in the configured ranges.
    """

    def __init__(self, config: Toy2DConfig | None = None, seed: int = 0) -> None:
        self.config = config or Toy2DConfig()
        self._rng = np.random.default_rng(seed)
        self.state: np.ndarray | None = None
        self.goal: np.ndarray | None = None

    def reset(
        self,
        state: np.ndarray | None = None,
        goal: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset to ``state`` and ``goal``; sample uniformly when omitted."""
        if state is None:
            state = self._rng.uniform(self.config.state_low, self.config.state_high, size=2)
        if goal is None:
            goal = self._rng.uniform(self.config.state_low, self.config.state_high, size=2)
        self.state = np.asarray(state, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        return self.state.copy(), self.goal.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply ``action``; return ``(next_state, reward, done, info)``."""
        if self.state is None or self.goal is None:
            raise RuntimeError("reset() must be called before step().")
        action = np.asarray(action, dtype=np.float64)
        noise = self._rng.normal(scale=self.config.noise_std, size=2)
        next_state = self.state + action + noise
        reward = -float(np.linalg.norm(next_state - self.goal))
        self.state = next_state
        done = False  # single-step reward; no terminal condition
        info = {"distance": -reward}
        return next_state.copy(), reward, done, info

    def sample_offline_data(self, n: int = 5000, seed: int | None = None) -> dict[str, np.ndarray]:
        """Sample an offline dataset whose actions live on a curved slice.

        Concretely: states/goals uniform in the configured state range; actions
        sampled in a disk of radius ``support_radius`` around the origin with a
        goal-aligned bias so the dataset overrepresents goal-reaching actions.
        Returns dict with keys ``states``, ``actions``, ``goals``, ``next_states``,
        ``rewards``.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        cfg = self.config
        states = rng.uniform(cfg.state_low, cfg.state_high, size=(n, 2))
        goals = rng.uniform(cfg.state_low, cfg.state_high, size=(n, 2))
        # Goal-aligned bias: half the actions are sampled along the (g-s)
        # direction within the support disk; the rest are sampled uniformly
        # inside the disk.
        bias_dir = goals - states
        bias_dir = bias_dir / (np.linalg.norm(bias_dir, axis=1, keepdims=True) + 1e-8)
        radial = rng.uniform(0.0, cfg.support_radius, size=(n, 1))
        biased = radial * bias_dir
        uniform_actions = self._sample_in_disk(rng, n, cfg.support_radius)
        mix = rng.uniform(size=(n, 1)) < 0.5
        actions = np.where(mix, biased, uniform_actions)
        noise = rng.normal(scale=cfg.noise_std, size=(n, 2))
        next_states = states + actions + noise
        rewards = -np.linalg.norm(next_states - goals, axis=1)
        return {
            "states": states,
            "actions": actions,
            "goals": goals,
            "next_states": next_states,
            "rewards": rewards,
        }

    @staticmethod
    def _sample_in_disk(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
        # Rejection sampling for the disk; cheap at small n.
        out = np.empty((n, 2))
        i = 0
        while i < n:
            batch = rng.uniform(-radius, radius, size=(2 * n, 2))
            keep = batch[np.linalg.norm(batch, axis=1) < radius]
            take = min(len(keep), n - i)
            out[i : i + take] = keep[:take]
            i += take
        return out

    def q_star(self, state: np.ndarray, action: np.ndarray, goal: np.ndarray) -> float:
        """Closed-form Q*(s, a, g) under the single-step reward model.

        Since reward is computed at the deterministic part of the next state,
        Q*(s, a, g) = -||s + a - g||  (expectation of -||s' - g|| over zero-mean
        noise approximated by -||E[s']-g||; exact for the noiseless mean).
        """
        s = np.asarray(state, dtype=np.float64)
        a = np.asarray(action, dtype=np.float64)
        g = np.asarray(goal, dtype=np.float64)
        return -float(np.linalg.norm(s + a - g))

    def q_star_grid(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        grid_size: int = 60,
    ) -> np.ndarray:
        """Dense Q* evaluation on a ``grid_size`` x ``grid_size`` action grid.

        Grid spans ``[action_low, action_high]`` along each action coordinate.
        Returns array of shape (grid_size, grid_size); axes are (a1, a0) so that
        plotting with ``imshow`` works without transposition.
        """
        cfg = self.config
        a0 = np.linspace(cfg.action_low, cfg.action_high, grid_size)
        a1 = np.linspace(cfg.action_low, cfg.action_high, grid_size)
        A0, A1 = np.meshgrid(a0, a1)
        actions = np.stack([A0.ravel(), A1.ravel()], axis=-1)
        s = np.asarray(state)
        g = np.asarray(goal)
        deltas = s[None, :] + actions - g[None, :]
        q = -np.linalg.norm(deltas, axis=-1)
        return q.reshape(grid_size, grid_size)
