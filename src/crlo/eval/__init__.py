"""Shared evaluation utilities used by the experiment drivers.

Pure-numpy/scipy where possible; JAX is imported lazily in the modules
that actually need it (training-adjacent ones).
"""
