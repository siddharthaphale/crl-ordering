"""Analyze action persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import ActionPersistenceConfig
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--manifest-path", type=Path, required=True, help="Frozen query manifest to evaluate."
    )
    parser.add_argument(
        "--pools-dir",
        type=Path,
        required=True,
        help="Directory containing the shared candidate pools.",
    )
    parser.add_argument("--merged-root", type=Path, required=True, help="Merged root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--m-grid", type=int, nargs="+", default=ActionPersistenceConfig.m_grid, help="M grid."
    )
    parser.add_argument(
        "--frac-target",
        type=float,
        default=ActionPersistenceConfig.frac_target,
        help="Frac target.",
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=ActionPersistenceConfig.h_min,
        help="Minimum resolvable return separation.",
    )
    parser.add_argument(
        "--smoothness-target",
        type=float,
        default=ActionPersistenceConfig.smoothness_target,
        help="Smoothness target.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="analyze_action_persistence", argv=argv)
    from crlo.eval.analyze_action_persistence import analyze_action_persistence

    return analyze_action_persistence(config)


if __name__ == "__main__":
    raise SystemExit(main())
