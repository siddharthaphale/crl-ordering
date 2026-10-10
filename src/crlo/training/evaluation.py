"""Evaluation utilities shared across all OGBench experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from crlo.utils.ogbench_vendor import ensure_ogbench


def run_eval(
    agent: Any,
    env: Any,
    num_episodes: int = None,
) -> float:
    """Evaluate across all env tasks; return mean success rate in [0, 100]."""
    if num_episodes is None:
        num_episodes = 250
    ensure_ogbench()
    from utils.evaluation import evaluate  # noqa: PLC0415

    unwrapped = env.unwrapped
    task_infos = getattr(unwrapped, "task_infos", None) or env.task_infos
    config_stub = {"discrete": False}

    all_success = []
    for task_id in range(1, len(task_infos) + 1):
        stats, _, _ = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config_stub,
            num_eval_episodes=num_episodes,
            num_video_episodes=0,
            eval_temperature=0,
            eval_gaussian=None,
        )
        all_success.append(float(stats.get("success", 0.0)))

    return float(np.mean(all_success)) * 100.0
