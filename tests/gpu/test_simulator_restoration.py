"""Real MuJoCo state restoration for all three navigation embodiments."""

from __future__ import annotations

import numpy as np
import ogbench
import pytest

from crlo.eval.candidate_returns import restore_state


@pytest.mark.parametrize(
    "task",
    ["pointmaze-large-navigate-v0", "antmaze-large-navigate-v0", "humanoidmaze-large-navigate-v0"],
)
def test_repeated_physics_state(task):
    env = ogbench.make_env_and_datasets(task, env_only=True)
    try:
        observation, _ = env.reset(seed=42)
        qpos, qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
        action = np.zeros(env.action_space.shape, np.float32)
        outputs = []
        for _ in range(2):
            restored = restore_state(env, qpos, qvel, observation, 1e-5, 1e-5)
            np.testing.assert_allclose(restored, observation, atol=1e-5, rtol=1e-5)
            outputs.append(env.step(action)[0].copy())
        np.testing.assert_array_equal(*outputs)
    finally:
        env.close()
