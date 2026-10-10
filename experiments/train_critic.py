"""Train critic."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.agents.model_names import CRITICS
from crlo.training.config_schemas import TrainCriticConfig
from crlo.utils.config_loader import CONFIG_DIR, ConfigMap, load_config
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DIR / "critic.yaml", help="Training hyperparameters."
    )
    parser.add_argument("--task", type=str, required=True, help="Task.")
    parser.add_argument("--critic", choices=CRITICS, type=str, required=True, help="Critic.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--seed", type=int, default=TrainCriticConfig.seed, help="Seed.")
    parser.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Steps.")
    parser.add_argument(
        "--eval-interval", type=int, default=TrainCriticConfig.eval_interval, help="Eval interval."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=TrainCriticConfig.eval_episodes, help="Eval episodes."
    )
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Batch size.")
    parser.add_argument("--discount", type=float, default=argparse.SUPPRESS, help="Discount.")
    parser.add_argument(
        "--bc-alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Actor regularization coefficient.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=TrainCriticConfig.dataset_dir,
        help="Local OGBench dataset directory.",
    )
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    config = load_config(path)
    TrainCriticConfig.validate_recipe(config)
    config.update(args)
    TrainCriticConfig.from_mapping(config)
    return config


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="train_critic", argv=argv)
    from crlo.training.train_critic import train_critic

    return train_critic(config)


if __name__ == "__main__":
    raise SystemExit(main())
