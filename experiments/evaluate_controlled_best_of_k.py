"""Evaluate controlled best of k."""

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
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint dir.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--n-queries", type=int, default=200, help="Number of sampled evaluation queries."
    )
    parser.add_argument("--proposal-std", type=float, default=0.55, help="Proposal std.")
    parser.add_argument("--clip", type=float, default=0.98, help="Clip.")
    parser.add_argument(
        "--candidate-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="Candidate counts.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_controlled_best_of_k", argv=argv)
    from crlo.eval.evaluate_controlled_best_of_k import evaluate_controlled_best_of_k

    return evaluate_controlled_best_of_k(config)


if __name__ == "__main__":
    raise SystemExit(main())
