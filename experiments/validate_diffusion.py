"""Validate diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task", type=str, required=True, help="Task.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint dir.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--dataset-dir", type=Path, default="data/ogbench", help="Local OGBench dataset directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount.")
    parser.add_argument("--n-samples", type=int, default=4096, help="N samples.")
    parser.add_argument("--max-denoising-loss", type=float, default=1.0, help="Max denoising loss.")
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="validate_diffusion", argv=argv)
    from crlo.eval.validate_diffusion import validate_diffusion

    return validate_diffusion(config)


if __name__ == "__main__":
    raise SystemExit(main())
