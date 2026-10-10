"""Train controlled critic."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.training.config_schemas import ControlledCriticConfig
from crlo.utils.config_loader import CONFIG_DIR, ConfigMap, load_config
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "controlled_critic.yaml",
        help="Training hyperparameters.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--seed", type=int, default=ControlledCriticConfig.seed, help="Seed.")
    parser.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Steps.")
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Batch size.")
    parser.add_argument("--n-train", type=int, default=argparse.SUPPRESS, help="N train.")
    parser.add_argument(
        "--critics", type=str, nargs="+", default=ControlledCriticConfig.critics, help="Critics."
    )
    parser.add_argument(
        "--bc-alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Actor regularization coefficient.",
    )
    parser.add_argument("--discount", type=float, default=argparse.SUPPRESS, help="Discount.")
    parser.add_argument(
        "--seed-protocol",
        type=str,
        choices=("geometry", "selection"),
        default=ControlledCriticConfig.seed_protocol,
        help="Recorded controlled-experiment sampling convention.",
    )
    parser.add_argument(
        "--critic-seed-stride",
        type=int,
        default=ControlledCriticConfig.critic_seed_stride,
        help="Offset between controlled critics' data-sampling seeds.",
    )
    parser.add_argument(
        "--model-seed-stride",
        type=int,
        default=ControlledCriticConfig.model_seed_stride,
        help="Seed multiplier used by the controlled selection convention.",
    )
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    config = load_config(path)
    ControlledCriticConfig.validate_recipe(config)
    config.update(args)
    ControlledCriticConfig.from_mapping(config)
    return config


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="train_controlled_critic", argv=argv)
    from crlo.training.train_controlled_critic import train_controlled_critic

    return train_controlled_critic(config)


if __name__ == "__main__":
    raise SystemExit(main())
