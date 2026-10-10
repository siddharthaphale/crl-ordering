"""CPU-only fixtures; set the backend before importing JAX."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
"""Shared fixtures for CPU and GPU test suites."""


import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for tests; reseeded per test."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for tests that write CSV / npy outputs."""
    d = tmp_path / "data"
    d.mkdir()
    return d
