"""Behavior sampler interface; density estimation is not part of this contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DiffusionSampler(Protocol):
    """A policy that proposes actions for a state and goal."""

    action_dim: int

    def sample(self, s: np.ndarray, g: np.ndarray, K: int) -> np.ndarray:
        """Return K actions for one query, or one action per batched query."""
        ...
