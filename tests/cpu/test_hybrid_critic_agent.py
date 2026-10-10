"""CPU test for the hybrid joint contrastive/Bellman critic agent factory."""

from __future__ import annotations

import inspect


def test_module_importable() -> None:
    from crlo.agents import hybrid_critic_agent  # noqa: F401


def test_factory_signature() -> None:
    from crlo.agents.hybrid_critic_agent import make_hybrid_critic_agent

    sig = inspect.signature(make_hybrid_critic_agent)
    expected = {
        "seed",
        "example_batch",
        "bc_alpha",
        "discount",
        "td_warmup_steps",
    }
    assert expected.issubset(set(sig.parameters))


def test_default_hybrid_warmup() -> None:
    from crlo.agents.hybrid_critic_agent import make_hybrid_critic_agent

    sig = inspect.signature(make_hybrid_critic_agent)
    assert sig.parameters["td_warmup_steps"].default is None
