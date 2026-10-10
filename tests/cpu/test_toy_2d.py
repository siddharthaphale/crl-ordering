"""CPU tests for the 2D toy environment."""

from __future__ import annotations

import numpy as np
import pytest

from crlo.envs.toy_2d import Toy2DConfig, Toy2DEnv


def test_reset_returns_correct_shapes() -> None:
    env = Toy2DEnv(seed=0)
    s, g = env.reset()
    assert s.shape == (2,)
    assert g.shape == (2,)


def test_reset_accepts_explicit_state_and_goal() -> None:
    env = Toy2DEnv(seed=0)
    s, g = env.reset(state=np.array([0.5, -0.5]), goal=np.array([0.0, 0.0]))
    np.testing.assert_array_equal(s, [0.5, -0.5])
    np.testing.assert_array_equal(g, [0.0, 0.0])


def test_step_deterministic_dynamics_with_zero_noise() -> None:
    env = Toy2DEnv(config=Toy2DConfig(noise_std=0.0), seed=0)
    env.reset(state=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    next_state, reward, done, info = env.step(np.array([0.5, 0.0]))
    np.testing.assert_allclose(next_state, [0.5, 0.0])
    assert not done
    # Reward = -||(0.5, 0) - (1, 0)|| = -0.5
    assert reward == pytest.approx(-0.5)
    assert info["distance"] == pytest.approx(0.5)


def test_step_requires_reset() -> None:
    env = Toy2DEnv(seed=0)
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0]))


def test_q_star_closed_form() -> None:
    env = Toy2DEnv(seed=0)
    # Q*(s=0, a=(0.5,0), g=(1,0)) = -||(0,0) + (0.5,0) - (1,0)|| = -0.5
    q = env.q_star(np.zeros(2), np.array([0.5, 0.0]), np.array([1.0, 0.0]))
    assert q == pytest.approx(-0.5)


def test_q_star_grid_shape_and_max_at_oracle() -> None:
    env = Toy2DEnv(seed=0)
    state = np.array([0.1, 0.2])
    goal = np.array([0.6, 0.3])
    grid = env.q_star_grid(state, goal, grid_size=60)
    assert grid.shape == (60, 60)
    # Argmax should be at the action a* ≈ g - s = (0.5, 0.1).
    flat_argmax = int(np.argmax(grid))
    i_a1, i_a0 = divmod(flat_argmax, 60)
    a0_axis = np.linspace(-1.0, 1.0, 60)
    a1_axis = np.linspace(-1.0, 1.0, 60)
    a_star_grid = np.array([a0_axis[i_a0], a1_axis[i_a1]])
    a_star_oracle = goal - state
    # Should be within one grid step.
    grid_step = (1.0 - (-1.0)) / (60 - 1)
    assert np.max(np.abs(a_star_grid - a_star_oracle)) < grid_step


def test_offline_data_shapes_and_support() -> None:
    env = Toy2DEnv(seed=0)
    data = env.sample_offline_data(n=1000, seed=0)
    for key in ("states", "actions", "goals", "next_states"):
        assert data[key].shape == (1000, 2)
    assert data["rewards"].shape == (1000,)
    # Actions should be within the support radius (with small slack for biased
    # actions which are sampled radially within the same radius).
    norms = np.linalg.norm(data["actions"], axis=1)
    assert (norms < env.config.support_radius + 1e-6).all()


def test_offline_data_reward_matches_step_dynamics() -> None:
    env = Toy2DEnv(seed=0)
    data = env.sample_offline_data(n=100, seed=1)
    expected_reward = -np.linalg.norm(data["next_states"] - data["goals"], axis=1)
    np.testing.assert_allclose(data["rewards"], expected_reward)
