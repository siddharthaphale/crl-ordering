"""OGBench agent overrides for the v2 paper.

Pattern: never modify upstream ``vendor/OGBench/impls/`` files. Instead, write
subclasses or wrapper functions here that change exactly the behavior we need.

Two overrides:
- ``crl_cosine.CRLCosineAgent``: cosine-normalized score (Lemma 1, Prop 5).
- ``crl_hybrid.CRLHybridAgent``: Q_TD + alpha * f_cos.

The base ``CRLAgent`` lives at ``vendor/OGBench/impls/agents/crl.py`` (pinned
to commit ``1d41409`` by ``python -m crlo.utils.ogbench_vendor --install``).
"""
