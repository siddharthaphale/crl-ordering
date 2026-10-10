"""Evaluate two head critics."""

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
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Checkpoint root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount.")
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "validation"),
        default="validation",
        help="Official dataset split to evaluate.",
    )
    parser.add_argument(
        "--n-queries", type=int, default=3000, help="Number of sampled evaluation queries."
    )
    parser.add_argument("--max-goal-distance", type=int, default=60, help="Max goal distance.")
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument(
        "--critics",
        type=str,
        nargs="+",
        default=["two_head_separate_encoders", "two_head_shared_encoder"],
        help="Critics.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_two_head_critics", argv=argv)
    from crlo.eval.evaluate_two_head_critics import evaluate_two_head_critics

    return evaluate_two_head_critics(config)


if __name__ == "__main__":
    raise SystemExit(main())
