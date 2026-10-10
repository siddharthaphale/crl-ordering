"""Light CPU test for the OGBench env loader.

Mujoco isn't available in the sandbox, so we only verify that
``load_crl_env_and_dataset`` is importable and has the expected
signature. The real smoke test that constructs an env lives in
``tests/gpu/test_train_critic_smoke.py``.
"""

from __future__ import annotations

import inspect

import pytest


def test_loader_function_exists() -> None:
    pytest.importorskip("ogbench", reason="OGBench not installed in sandbox")
    from crlo.envs import ogbench_loader

    fn = ogbench_loader.load_crl_env_and_dataset
    sig = inspect.signature(fn)
    assert {"env_name", "discount"} <= set(sig.parameters)
