r"""Train one of four critics on one OGBench task.

Usage::

    python experiments/train_critic.py \
        --config config/critic.yaml \
        --task cube-single-play-v0 --critic raw_bilinear --output-dir checkpoints/raw

Per-eval rows are written to ``training_log.csv`` with columns:
    step, success_rate, exploit_gap, eval_return,
    mean_phi_norm, mean_cos_align, critic_loss, actor_loss
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.training.config_schemas import CriticModelConfig, TrainCriticConfig

LOG_COLUMNS = (
    "step",
    "success_rate",
    "eval_return",
    "mean_phi_norm",
    "mean_cos_align",
    "critic_loss",
    "actor_loss",
)


from crlo.agents.critic_factory import build_critic


def _first_present(info: dict, *keys: str) -> float:
    for k in keys:
        if k in info:
            return float(info[k])
    return float("nan")


def _extract_diagnostics(info: dict) -> dict[str, float]:
    return {
        "mean_phi_norm": _first_present(
            info,
            "phi_norm_mean",
            "critic/phi_norm_mean",
        ),
        "mean_cos_align": _first_present(
            info,
            "cosine_normalize",
            "critic/cosine_normalize",
        ),
        "critic_loss": _first_present(
            info,
            "contrastive_loss",
            "critic/contrastive_loss",
            "value_loss",
            "critic/value_loss",
        ),
        "actor_loss": _first_present(info, "actor_loss", "actor/actor_loss"),
    }


def train_critic(configuration: ConfigMap) -> int:
    """Train one configured critic and save its complete model state."""
    cfg = TrainCriticConfig.from_mapping(
        configuration,
    )
    np.random.seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    from crlo.envs.ogbench_loader import load_crl_env_and_dataset
    from crlo.training.evaluation import run_eval

    env, train_dataset = load_crl_env_and_dataset(
        env_name=cfg.task,
        discount=cfg.discount,
        dataset_dir=cfg.dataset_dir,
    )
    example_batch = train_dataset.sample(1)
    agent = build_critic(
        cfg.critic,
        cfg.seed,
        example_batch,
        cfg.bc_alpha,
        cfg.discount,
        model_config=CriticModelConfig(**cfg.model).agent_config(
            cfg.critic, batch_size=cfg.batch_size
        ),
    )

    # Goal sampling must agree with the constructed agent's reward convention.
    train_dataset.config.update(dict(agent.config))
    max_steps = cfg.steps
    eval_interval = cfg.eval_interval
    print(
        f"[train_critic] critic={cfg.critic} task={cfg.task} "
        f"steps={max_steps} seed={cfg.seed} bc_alpha={cfg.bc_alpha} "
        f"discount={cfg.discount}",
        flush=True,
    )

    log_path = cfg.output_dir / "training_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()

        start = time.time()
        last_info: dict = {}
        for step in range(1, max_steps + 1):
            batch = train_dataset.sample(cfg.batch_size)
            agent, last_info = agent.update(batch)

            if eval_interval > 0 and step % eval_interval == 0:
                success = run_eval(agent, env, num_episodes=cfg.eval_episodes)
                writer.writerow(
                    {
                        "step": step,
                        "success_rate": success,
                        "eval_return": success,
                        **_extract_diagnostics(last_info),
                    }
                )
                f.flush()
                elapsed = (time.time() - start) / 60.0
                print(
                    f"Step {step:>7} | {cfg.critic}: {success:.1f}% | elapsed: {elapsed:.1f}min",
                    flush=True,
                )

    from crlo.training.checkpoints import save_critic  # noqa: PLC0415
    from crlo.utils.logging import RunLogger

    obs_dim = int(np.asarray(example_batch["observations"]).shape[-1])
    action_dim = int(np.asarray(example_batch["actions"]).shape[-1])
    save_critic(
        agent,
        cfg.output_dir,
        meta={
            "critic": cfg.critic,
            "task": cfg.task,
            "steps": max_steps,
            "seed": cfg.seed,
            "bc_alpha": cfg.bc_alpha,
            "discount": cfg.discount,
            "eval_episodes": cfg.eval_episodes,
            "eval_interval": eval_interval,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        },
    )
    RunLogger(cfg.output_dir).log_event(
        "metadata",
        {
            "script": "train_critic",
            "critic": cfg.critic,
            "task": cfg.task,
            "steps": max_steps,
            "seed": cfg.seed,
            "bc_alpha": cfg.bc_alpha,
            "discount": cfg.discount,
            "eval_episodes": cfg.eval_episodes,
            "eval_interval": eval_interval,
            "elapsed_min": (time.time() - start) / 60.0,
        },
    )
    print(f"[train_critic] done. log={log_path}", flush=True)
    return 0
