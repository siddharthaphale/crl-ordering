"""Evaluate executed value ordering."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import ExecutedValueOrderingConfig
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
        choices=("critic_comparison", "seed_replication"),
        required=True,
        help="Mode.",
    )
    parser.add_argument(
        "--manifest-path", type=Path, required=True, help="Frozen query manifest to evaluate."
    )
    parser.add_argument("--merged-path", type=Path, required=True, help="Merged path.")
    parser.add_argument("--reliability-path", type=Path, required=True, help="Reliability path.")
    parser.add_argument("--scores-dir", type=Path, required=True, help="Scores dir.")
    parser.add_argument("--retrieval-csv", type=Path, required=True, help="Retrieval csv.")
    parser.add_argument(
        "--resample-indices-path", type=Path, required=True, help="Resample indices path."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=ExecutedValueOrderingConfig.comparison_path,
        help="Comparison path.",
    )
    parser.add_argument(
        "--critics",
        type=str,
        nargs="+",
        default=ExecutedValueOrderingConfig.critics,
        help="Critics.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=ExecutedValueOrderingConfig.seeds, help="Seeds."
    )
    parser.add_argument("--k", type=int, default=ExecutedValueOrderingConfig.k, help="K.")
    parser.add_argument(
        "--rollouts-per-half",
        type=int,
        default=ExecutedValueOrderingConfig.rollouts_per_half,
        help="Independent continuation rollouts in each verification half.",
    )
    parser.add_argument(
        "--k-curve",
        type=int,
        nargs="+",
        default=ExecutedValueOrderingConfig.k_curve,
        help="K curve.",
    )
    parser.add_argument(
        "--delta-tau",
        type=float,
        default=ExecutedValueOrderingConfig.delta_tau,
        help="Minimum improvement in Kendall ordering correlation.",
    )
    parser.add_argument(
        "--delta-v",
        type=float,
        default=ExecutedValueOrderingConfig.delta_v,
        help="Minimum improvement in selected-action return.",
    )
    parser.add_argument(
        "--tau-cap",
        type=float,
        default=ExecutedValueOrderingConfig.tau_cap,
        help="Maximum contrastive ordering correlation for the comparison criterion.",
    )
    parser.add_argument(
        "--auc-min",
        type=float,
        default=ExecutedValueOrderingConfig.auc_min,
        help="Minimum retrieval AUC for the comparison criterion.",
    )
    parser.add_argument(
        "--top-decile-size",
        type=int,
        default=ExecutedValueOrderingConfig.top_decile_size,
        help="Top decile size.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_executed_value_ordering", argv=argv)
    from crlo.eval.evaluate_executed_value_ordering import evaluate_executed_value_ordering

    return evaluate_executed_value_ordering(config)


if __name__ == "__main__":
    raise SystemExit(main())
