"""Evaluate reference ordering."""

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
    parser.add_argument(
        "--mode", choices=("precheck", "metrics"), type=str, default="precheck", help="Mode."
    )
    parser.add_argument("--task", type=str, required=True, help="Task.")
    parser.add_argument(
        "--refs",
        type=str,
        nargs="+",
        default=["reference_seed7", "reference_seed8", "reference_seed9"],
        help="Reference score labels.",
    )
    parser.add_argument("--critics", type=str, nargs="+", default=None, help="Critics.")
    parser.add_argument(
        "--k-grid",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Candidate counts for the selection curve.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=1000, help="Number of episode-cluster bootstrap resamples."
    )
    parser.add_argument("--boot-seed", type=int, default=12345, help="Boot seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir.")
    parser.add_argument("--figure-data-path", type=Path, default=None, help="Figure data path.")
    parser.add_argument(
        "--expected-resolvable-queries", type=int, default=688, help="Expected resolvable queries."
    )
    parser.add_argument(
        "--representative-query", type=int, default=200, help="Representative query."
    )
    parser.add_argument(
        "--representative-replicate", type=int, default=4, help="Representative replicate."
    )
    parser.add_argument(
        "--figure-critic-labels",
        nargs=4,
        metavar=("TD_Q", "RAW_BILINEAR", "COSINE", "HYBRID"),
        default=("td_q_seed42", "raw_bilinear_seed42", "cosine_seed42", "hybrid_seed42"),
        help="Score labels used for the representative-query figure, in the displayed order.",
    )
    config = ConfigMap(vars(parser.parse_args(argv)))
    config["figure_critic_labels"] = dict(
        zip(
            ("td_q", "raw_bilinear", "cosine", "hybrid"),
            config["figure_critic_labels"],
            strict=True,
        )
    )
    return config


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_reference_ordering", argv=argv)
    from crlo.eval.evaluate_reference_ordering import evaluate_reference_ordering

    return evaluate_reference_ordering(config)


if __name__ == "__main__":
    raise SystemExit(main())
