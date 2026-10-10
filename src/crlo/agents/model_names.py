"""Public critic identifiers used by recipes and checkpoint factories."""

from __future__ import annotations

CRITICS = (
    "raw_bilinear",
    "cosine",
    "td_q",
    "hybrid",
    # Two-head hybrid variants (diagnostic #4): contrastive embeddings + a
    # free scalar IQL q_td selection head.
    "two_head_separate_encoders",
    "two_head_shared_encoder_stopped_gradient",
    "two_head_shared_encoder",
    "two_head_shared_encoder_td_only",
    # Complete-agent variants (option B): + AWR policy head.
    "two_head_separate_encoders_with_actor",
    "two_head_shared_encoder_stopped_gradient_with_actor",
    "two_head_shared_encoder_with_actor",
    "two_head_shared_encoder_td_only_with_actor",
    # Objective/function-class controls: Bellman on the raw
    # bilinear scalar; contrastive NCE on a TD-Q-parameter-matched MLP.
    "td_bilinear",
    "contrastive_mlp",
)
