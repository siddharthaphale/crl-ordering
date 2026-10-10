"""Evaluate controlled selection."""

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
    parser.add_argument(
        "--critics",
        type=str,
        nargs="+",
        default=["raw_bilinear", "cosine", "td_q", "hybrid"],
        help="Critics.",
    )
    parser.add_argument("--candidates", type=int, default=256, help="Candidates.")
    parser.add_argument("--support-radius", type=float, default=0.4, help="Support radius.")
    parser.add_argument("--query-seed", type=int, default=7, help="Query seed.")
    parser.add_argument(
        "--goal-radius-fraction", type=float, default=0.6, help="Goal radius fraction."
    )
    parser.add_argument(
        "--proposal-radius-multiplier", type=float, default=1.2, help="Proposal radius multiplier."
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_controlled_selection", argv=argv)
    from crlo.eval.evaluate_controlled_selection import evaluate_controlled_selection

    return evaluate_controlled_selection(config)


if __name__ == "__main__":
    raise SystemExit(main())
