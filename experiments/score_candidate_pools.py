"""Score candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.agents.model_names import CRITICS
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pools-dir",
        type=Path,
        required=True,
        help="Directory containing the shared candidate pools.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint dir.")
    parser.add_argument("--critic", choices=CRITICS, type=str, required=True, help="Critic.")
    parser.add_argument("--rebuild-critic", type=str, default=None, help="Rebuild critic.")
    parser.add_argument("--label", type=str, required=True, help="Label.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=8192,
        help="Maximum number of candidate scores computed per batch.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir.")
    parser.add_argument(
        "--population",
        type=str,
        choices=("all_pools", "manifest"),
        default="all_pools",
        help="Population.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task.")
    parser.add_argument(
        "--manifest-path", type=Path, default=None, help="Frozen query manifest to evaluate."
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Training seed recorded for the evaluated checkpoint.",
    )
    parser.add_argument(
        "--expected-pool-sha16",
        type=str,
        default=None,
        help="Expected first 16 hex characters of the candidate-pool SHA-256.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="score_candidate_pools", argv=argv)
    from crlo.eval.score_candidate_pools import score_candidate_pools

    return score_candidate_pools(config)


if __name__ == "__main__":
    raise SystemExit(main())
