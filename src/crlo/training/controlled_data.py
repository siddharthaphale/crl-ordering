"""Offline controlled benchmark batches with explicit sampling RNG."""

from __future__ import annotations

import numpy as np


def sample_controlled_batch(
    rng: np.random.Generator,
    states: np.ndarray,
    actions: np.ndarray,
    goals: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Sample controlled transitions in the OGBench critic batch format."""
    idx = rng.integers(0, states.shape[0], size=batch_size)
    masks = np.linalg.norm(next_states[idx] - goals[idx], axis=1) > 0.05
    return {
        "observations": states[idx].astype(np.float32),
        "actions": actions[idx].astype(np.float32),
        "next_observations": next_states[idx].astype(np.float32),
        "value_goals": goals[idx].astype(np.float32),
        "actor_goals": goals[idx].astype(np.float32),
        "rewards": rewards[idx].astype(np.float32),
        "masks": masks.astype(np.float32),
    }
