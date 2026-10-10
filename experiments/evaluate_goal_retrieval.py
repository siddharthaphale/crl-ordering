"""Evaluate goal retrieval."""

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
        default=["raw_bilinear", "cosine", "td_q", "hybrid"],
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
    parser.add_argument("--n-query", type=int, default=300, help="Number of retrieval queries.")
    parser.add_argument(
        "--n-pool", type=int, default=50, help="Number of candidates in each retrieval pool."
    )
    parser.add_argument("--evals-root", type=Path, required=True, help="Evals root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_goal_retrieval", argv=argv)
    from crlo.eval.evaluate_goal_retrieval import evaluate_goal_retrieval

    return evaluate_goal_retrieval(config)


if __name__ == "__main__":
    raise SystemExit(main())
