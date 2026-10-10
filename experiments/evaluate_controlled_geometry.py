"""Evaluate controlled geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import ControlledGeometryConfig
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--seed", type=int, default=ControlledGeometryConfig.seed, help="Seed.")
    parser.add_argument(
        "--grid-size", type=int, default=ControlledGeometryConfig.grid_size, help="Grid size."
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=ControlledGeometryConfig.n_candidates,
        help="N candidates.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint dir.")
    parser.add_argument(
        "--proposal-std",
        type=float,
        default=ControlledGeometryConfig.proposal_std,
        help="Proposal std.",
    )
    parser.add_argument(
        "--proposal-clip",
        type=float,
        default=ControlledGeometryConfig.proposal_clip,
        help="Proposal clip.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_controlled_geometry", argv=argv)
    from crlo.eval.evaluate_controlled_geometry import evaluate_controlled_geometry

    return evaluate_controlled_geometry(config)


if __name__ == "__main__":
    raise SystemExit(main())
