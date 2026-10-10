"""Evaluate selected action returns."""

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
        "--pools-dir",
        type=Path,
        required=True,
        help="Directory containing the shared candidate pools.",
    )
    parser.add_argument("--diffusion-ckpt", type=Path, required=True, help="Diffusion ckpt.")
    parser.add_argument("--critics", type=str, nargs="+", required=True, help="Critics.")
    parser.add_argument(
        "--refs",
        type=str,
        nargs="+",
        default=["reference_seed7", "reference_seed8", "reference_seed9"],
        help="Reference score labels.",
    )
    parser.add_argument("--n-per-stratum", type=int, default=34, help="N per stratum.")
    parser.add_argument("--n-rollouts", type=int, default=5, help="N rollouts.")
    parser.add_argument("--horizon", type=int, default=300, help="Horizon.")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount.")
    parser.add_argument("--audit-seed", type=int, default=202607, help="Audit seed.")
    parser.add_argument("--replicate", type=int, default=0, help="Replicate.")
    parser.add_argument(
        "--n-shards", type=int, default=1, help="Number of partitions in this rollout population."
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="Zero-based partition to evaluate in this invocation."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--dataset-dir", type=Path, default="data/ogbench", help="Local OGBench dataset directory."
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_selected_action_returns", argv=argv)
    from crlo.eval.evaluate_selected_action_returns import evaluate_selected_action_returns

    return evaluate_selected_action_returns(config)


if __name__ == "__main__":
    raise SystemExit(main())
