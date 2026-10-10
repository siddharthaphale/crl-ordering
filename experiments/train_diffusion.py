"""Train diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.training.config_schemas import TrainDiffusionConfig
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
        default=CONFIG_DIR / "diffusion.yaml",
        help="Training hyperparameters.",
    )
    parser.add_argument("--task", type=str, required=True, help="Task.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--seed", type=int, default=TrainDiffusionConfig.seed, help="Seed.")
    parser.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Steps.")
    parser.add_argument(
        "--denoising-steps", type=int, default=argparse.SUPPRESS, help="Denoising steps."
    )
    parser.add_argument(
        "--lr", type=float, default=argparse.SUPPRESS, help="Optimizer learning rate."
    )
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Batch size.")
    parser.add_argument("--discount", type=float, default=argparse.SUPPRESS, help="Discount.")
    parser.add_argument(
        "--log-interval", type=int, default=TrainDiffusionConfig.log_interval, help="Log interval."
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=argparse.SUPPRESS, help="Hidden dims."
    )
    parser.add_argument(
        "--schedule",
        choices=("cosine", "linear"),
        type=str,
        default=argparse.SUPPRESS,
        help="Schedule.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=TrainDiffusionConfig.validation_samples,
        help="Validation samples.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=TrainDiffusionConfig.dataset_dir,
        help="Local OGBench dataset directory.",
    )
    parser.add_argument(
        "--time-embed-dim", type=int, default=argparse.SUPPRESS, help="Time embed dim."
    )
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    config = load_config(path)
    TrainDiffusionConfig.validate_recipe(config)
    config.update(args)
    TrainDiffusionConfig.from_mapping(config)
    return config


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="train_diffusion", argv=argv)
    from crlo.training.train_diffusion import train_diffusion

    return train_diffusion(config)


if __name__ == "__main__":
    raise SystemExit(main())
