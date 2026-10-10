"""Calibrate rollout budget."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import RolloutBudgetConfig
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
    parser.add_argument("--merged-path", type=Path, required=True, help="Merged path.")
    parser.add_argument(
        "--baseline-merged-path", type=Path, required=True, help="Baseline merged path."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--k", type=int, default=RolloutBudgetConfig.k, help="K.")
    parser.add_argument(
        "--rollouts-per-half",
        type=int,
        default=RolloutBudgetConfig.rollouts_per_half,
        help="Independent continuation rollouts in each verification half.",
    )
    parser.add_argument(
        "--r-grid", type=int, nargs="+", default=RolloutBudgetConfig.r_grid, help="R grid."
    )
    parser.add_argument(
        "--r-extrap", type=int, nargs="+", default=RolloutBudgetConfig.r_extrap, help="R extrap."
    )
    parser.add_argument(
        "--r-choices", type=int, nargs="+", default=RolloutBudgetConfig.r_choices, help="R choices."
    )
    parser.add_argument(
        "--agreement-target",
        type=float,
        default=RolloutBudgetConfig.agreement_target,
        help="Agreement target.",
    )
    parser.add_argument(
        "--frac-target", type=float, default=RolloutBudgetConfig.frac_target, help="Frac target."
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="calibrate_rollout_budget", argv=argv)
    from crlo.eval.calibrate_rollout_budget import calibrate_rollout_budget

    return calibrate_rollout_budget(config)


if __name__ == "__main__":
    raise SystemExit(main())
