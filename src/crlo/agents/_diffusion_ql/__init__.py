"""JAX Diffusion-QL behavior policy (command: ``experiments/train_diffusion.py``).


Five modules, each with a numpy counterpart in ``tests/cpu/diffusion_reference.py`` so
that JAX correctness is verified against closed-form math:

- ``noise_schedule``: linear / cosine beta schedules and derived coefficients.
- ``score_network``: MLP epsilon predictor (s, a, t, g) -> noise.
- ``ddpm_sampler``: ``q_sample`` (forward) and ``p_sample_loop`` (reverse).
- ``policy``: ``DiffusionQLBehavior`` wraps schedule + network + sampler and
  satisfies the ``crlo.eval.diffusion_sampler.DiffusionSampler`` protocol.
- ``train``: DDPM "simple" loss and a single train step.
"""
