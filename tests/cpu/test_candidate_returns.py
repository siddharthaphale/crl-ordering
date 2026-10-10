"""Frozen executed-value rollout cell: deterministic mock-env return + restore bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from crlo.eval.candidate_returns import (
    reset_wrapper_bookkeeping,
    restore_state,
    run_cell,
)


class MockEnv:
    """1-D kinematic env: position += action[0]; goal at x >= target."""

    def __init__(self, target=3.0):
        self.x = 0.0
        self.target = target

    def step(self, a):
        self.x += float(np.asarray(a).ravel()[0])
        return np.array([self.x, 0.0]), 0.0, False, False, {}


class MockPolicy:
    """Continuation that always advances by +1."""

    def __init__(self):
        self.rng = None
        self.calls = 0

    def sample(self, ob, g, k):
        self.calls += 1
        return np.ones((k, 1))


def _run(candidate, horizon=300, target=3.0):
    env, pol = MockEnv(target), MockPolicy()
    goal = np.array([target, 0.0])
    return run_cell(
        env,
        pol,
        np.zeros(2),
        goal,
        goal[:2],
        goal_tol=0.5,
        candidate=np.array([candidate]),
        act_lo=np.array([-1.0]),
        act_hi=np.array([1.0]),
        horizon=horizon,
        gamma=0.99,
    ), pol


def test_candidate_counts_as_step_one():
    # candidate +1, then 2 continuation steps of +1 reach x=3 (tol 0.5 at 2.5)
    (success, steps, y), pol = _run(1.0)
    assert success == 1 and steps == 3
    assert y == pytest.approx(0.99**3)
    assert pol.calls == 2


def test_horizon_exhaustion_gives_zero_return():
    (success, steps, y), _ = _run(-1.0, horizon=3)
    assert (success, steps, y) == (0, -1, 0.0)


def test_candidate_clipped_to_bounds():
    # candidate 10.0 must clip to +1.0, so the path equals the +1 case
    (success, steps, y), _ = _run(10.0)
    assert success == 1 and steps == 3 and y == pytest.approx(0.99**3)


def test_immediate_goal_after_candidate():
    (success, steps, y), pol = _run(1.0, target=1.0)
    assert success == 1 and steps == 1
    assert y == pytest.approx(0.99)
    assert pol.calls == 0  # no continuation needed


class _TimeLimitLike:
    def __init__(self, inner):
        self.env = inner
        self._elapsed_steps = 123


class _Inner:
    env = None


def test_reset_wrapper_bookkeeping_walks_stack():
    inner = _Inner()
    outer = _TimeLimitLike(_TimeLimitLike(inner))
    reset_wrapper_bookkeeping(outer)
    assert outer._elapsed_steps == 0
    assert outer.env._elapsed_steps == 0


class _RestorableEnv:
    """set_state/get_ob pair for the fail-closed restoration contract."""

    class _U:
        def __init__(self, drift):
            self.drift = drift
            self.state = None

        def set_state(self, qpos, qvel):
            self.state = np.concatenate([qpos, qvel]) + self.drift

        def get_ob(self):
            return self.state

    def __init__(self, drift=0.0):
        self.unwrapped = self._U(drift)
        self.env = None


def test_restore_state_verifies_and_fails_closed():
    qpos, qvel = np.arange(3.0), np.arange(2.0)
    expected = np.concatenate([qpos, qvel])
    env = _RestorableEnv(drift=0.0)
    ob = restore_state(env, qpos, qvel, expected, atol=1e-5, rtol=1e-5)
    assert np.allclose(ob, expected)
    bad = _RestorableEnv(drift=1e-3)
    with pytest.raises(RuntimeError, match="failing closed"):
        restore_state(bad, qpos, qvel, expected, atol=1e-5, rtol=1e-5)


def test_action_persistence_semantics():
    # persist=3, target at 3.0: candidate +1 held 3 steps reaches 2.5-tol goal
    env, pol = MockEnv(3.0), MockPolicy()
    goal = np.array([3.0, 0.0])
    out = run_cell(
        env,
        pol,
        np.zeros(2),
        goal,
        goal[:2],
        0.5,
        np.array([1.0]),
        np.array([-1.0]),
        np.array([1.0]),
        horizon=300,
        gamma=0.99,
        persist=3,
    )
    assert out == (1, 3, pytest.approx(0.99**3))
    assert pol.calls == 0  # goal hit inside persistence


def test_persistence_stops_at_success_mid_persist():
    env, pol = MockEnv(1.0), MockPolicy()
    goal = np.array([1.0, 0.0])
    out = run_cell(
        env,
        pol,
        np.zeros(2),
        goal,
        goal[:2],
        0.5,
        np.array([1.0]),
        np.array([-1.0]),
        np.array([1.0]),
        horizon=300,
        gamma=0.99,
        persist=10,
    )
    assert out[0] == 1 and out[1] == 1  # stopped at step 1, not 10


def test_persist_one_equals_original_protocol():
    for target in (1.0, 3.0):
        a = _run(1.0, target=target)[0]
        env, pol = MockEnv(target), MockPolicy()
        goal = np.array([target, 0.0])
        b = run_cell(
            env,
            pol,
            np.zeros(2),
            goal,
            goal[:2],
            0.5,
            np.array([1.0]),
            np.array([-1.0]),
            np.array([1.0]),
            horizon=300,
            gamma=0.99,
            persist=1,
        )
        assert a == b
