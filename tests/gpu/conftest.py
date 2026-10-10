"""GPU acceptance tests require JAX CUDA and never silently skip."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def pytest_sessionstart(session):
    from crlo.utils.gpu_check import gpu_check

    gpu_check()
