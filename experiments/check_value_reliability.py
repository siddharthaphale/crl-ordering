"""Check value reliability."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import ValueReliabilityConfig
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("pilot", "primary", "deterministic_primary"),
        required=True,
        help="Mode.",
    )
    parser.add_argument(
        "--manifest-path", type=Path, required=True, help="Frozen query manifest to evaluate."
    )
    parser.add_argument("--merged-path", type=Path, required=True, help="Merged path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--k", type=int, default=ValueReliabilityConfig.k, help="K.")
    parser.add_argument(
        "--rollouts-per-half",
        type=int,
        default=ValueReliabilityConfig.rollouts_per_half,
        help="Independent continuation rollouts in each verification half.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=ValueReliabilityConfig.bootstrap_seed,
        help="Random seed for episode-cluster bootstrap resampling.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=ValueReliabilityConfig.n_boot,
        help="Number of episode-cluster bootstrap resamples.",
    )
    parser.add_argument(
        "--frac-target", type=float, default=ValueReliabilityConfig.frac_target, help="Frac target."
    )
    parser.add_argument(
        "--agreement-target",
        type=float,
        default=ValueReliabilityConfig.agreement_target,
        help="Agreement target.",
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=ValueReliabilityConfig.h_min,
        help="Minimum resolvable return separation.",
    )
    parser.add_argument(
        "--pilot-ub-q",
        type=float,
        default=ValueReliabilityConfig.pilot_ub_q,
        help="Bootstrap quantile used for the pilot upper bound.",
    )
    parser.add_argument(
        "--gate-lb-q",
        type=float,
        default=ValueReliabilityConfig.gate_lb_q,
        help="Bootstrap quantile used for the reliability lower bound.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="check_value_reliability", argv=argv)
    from crlo.eval.check_value_reliability import check_value_reliability

    return check_value_reliability(config)


if __name__ == "__main__":
    raise SystemExit(main())
