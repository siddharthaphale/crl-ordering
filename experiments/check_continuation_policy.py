"""Check continuation policy."""

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
    parser.add_argument(
        "--pools-dir",
        type=Path,
        required=True,
        help="Directory containing the shared candidate pools.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint dir.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output json.")
    parser.add_argument("--task", type=str, default="antmaze-large-navigate-v0", help="Task.")
    parser.add_argument(
        "--continuation-mode",
        type=str,
        choices=("stochastic", "deterministic"),
        default="deterministic",
        help="Continuation mode.",
    )
    parser.add_argument("--rank-start", type=int, default=460, help="Rank start.")
    parser.add_argument("--rank-end", type=int, default=489, help="Rank end.")
    parser.add_argument("--stratum", type=int, default=1, help="Stratum.")
    parser.add_argument("--horizon", type=int, default=300, help="Horizon.")
    parser.add_argument("--minimum-successes", type=int, default=15, help="Minimum successes.")
    parser.add_argument("--namespace-half", type=int, default=9, help="Namespace half.")
    parser.add_argument("--namespace-rollout", type=int, default=0, help="Namespace rollout.")
    parser.add_argument("--rollout-seed", type=int, default=7001, help="Rollout seed.")
    parser.add_argument(
        "--restore-atol",
        type=float,
        default=1e-05,
        help="Absolute tolerance for restored simulator observations.",
    )
    parser.add_argument(
        "--restore-rtol",
        type=float,
        default=1e-05,
        help="Relative tolerance for restored simulator observations.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default="data/ogbench", help="Local OGBench dataset directory."
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="check_continuation_policy", argv=argv)
    from crlo.eval.check_continuation_policy import check_continuation_policy

    return check_continuation_policy(config)


if __name__ == "__main__":
    raise SystemExit(main())
