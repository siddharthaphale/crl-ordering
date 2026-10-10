"""Evaluate cross query selection."""

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
    parser.add_argument(
        "--critics",
        type=str,
        nargs="+",
        default=["td_q", "raw_bilinear", "cosine", "hybrid"],
        help="Critics.",
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount.")
    parser.add_argument(
        "--n-triples",
        type=int,
        default=3000,
        help="Number of sampled state-action-goal trajectory triples.",
    )
    parser.add_argument(
        "--kmax", type=int, default=60, help="Maximum temporal distance to a sampled goal."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument("--n-pools", type=int, default=8000, help="N pools.")
    parser.add_argument(
        "--k-grid",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Candidate counts for the selection curve.",
    )
    parser.add_argument("--evals-root", type=Path, required=True, help="Evals root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_cross_query_selection", argv=argv)
    from crlo.eval.evaluate_cross_query_selection import evaluate_cross_query_selection

    return evaluate_cross_query_selection(config)


if __name__ == "__main__":
    raise SystemExit(main())
