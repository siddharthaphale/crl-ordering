"""Analytic Gaussian sampler used only for interface tests."""

from __future__ import annotations

import numpy as np


class MockSampler:
    """Gaussian-policy fallback for the diffusion behavior model.

    The "behavior" is a goal-direction-aware Gaussian centred on
    ``alpha * (g - s)`` with isotropic variance. Defaults come from
    numerical test fixtures.
    """

    def __init__(
        self,
        action_dim: int = 2,
        alpha: float = 0.5,
        sigma: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.action_dim = action_dim
        self.alpha = alpha
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def _mean(self, s: np.ndarray, g: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        g = np.atleast_1d(np.asarray(g, dtype=np.float64))
        diff = g - s
        if diff.shape[-1] >= self.action_dim:
            return self.alpha * diff[..., : self.action_dim]
        # Pad with zeros if action_dim > state_dim
        pad = np.zeros(diff.shape[:-1] + (self.action_dim - diff.shape[-1],))
        return self.alpha * np.concatenate([diff, pad], axis=-1)

    def sample(self, s: np.ndarray, g: np.ndarray, K: int) -> np.ndarray:
        """Draw ``K`` samples from the Gaussian behavior at ``(s, g)``."""
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        mean = self._mean(s, g)
        if mean.ndim == 1:
            mean = mean[None, :]  # (1, action_dim)
        # Broadcast K samples per query, then collapse to (K, action_dim).
        noise = self._rng.normal(scale=self.sigma, size=(K,) + mean.shape[-1:])
        if mean.shape[0] == 1:
            return mean[0] + noise
        # Batched: produce one sample per query, K must equal batch size or
        # we tile.
        if mean.shape[0] != K:
            raise ValueError(
                f"Batched sample requires K == batch size; got K={K}, batch={mean.shape[0]}"
            )
        return mean + noise

    def log_likelihood(self, a: np.ndarray, s: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Return per-row Gaussian log-density ``log p(a | s, g)``."""
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        mean = self._mean(s, g)
        if mean.ndim == 1:
            mean = np.broadcast_to(mean, a.shape)
        d = a.shape[-1]
        log_norm = -0.5 * d * np.log(2 * np.pi * self.sigma**2)
        sq = np.sum((a - mean) ** 2, axis=-1) / (2 * self.sigma**2)
        return log_norm - sq
