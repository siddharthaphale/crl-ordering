"""CPU test for the cosine CRL agent factory.

The factory itself instantiates a JAX agent and so cannot run on CPU without
JAX installed. We verify the module is importable, the factory function
exists, and its signature matches the plan.
"""

from __future__ import annotations

import inspect


def test_module_importable() -> None:
    from crlo.agents import crl_cosine_agent  # noqa: F401


def test_factory_signature() -> None:
    from crlo.agents.crl_cosine_agent import make_crl_cosine_agent

    sig = inspect.signature(make_crl_cosine_agent)
    expected = {"seed", "example_batch", "bc_alpha", "discount"}
    assert expected.issubset(set(sig.parameters))
