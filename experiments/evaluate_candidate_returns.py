"""Evaluate candidate returns."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import CandidateReturnsConfig
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task", type=str, default="antmaze-large-navigate-v0", help="Task.")
    parser.add_argument(
        "--pools-dir",
        type=Path,
        required=True,
        help="Directory containing the shared candidate pools.",
    )
    parser.add_argument(
        "--manifest-path", type=Path, required=True, help="Frozen query manifest to evaluate."
    )
    parser.add_argument("--diffusion-ckpt", type=Path, required=True, help="Diffusion ckpt.")
    parser.add_argument(
        "--continuation-sha256",
        type=str,
        required=True,
        help="Expected SHA-256 of the continuation checkpoint parameters.",
    )
    parser.add_argument(
        "--expected-pool-sha16",
        type=str,
        required=True,
        help="Expected first 16 hex characters of the candidate-pool SHA-256.",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=CandidateReturnsConfig.val_data_path,
        help="Val data path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--horizon", type=int, default=CandidateReturnsConfig.horizon, help="Horizon."
    )
    parser.add_argument(
        "--discount", type=float, default=CandidateReturnsConfig.discount, help="Discount."
    )
    parser.add_argument(
        "--rollouts-per-half",
        type=int,
        default=CandidateReturnsConfig.rollouts_per_half,
        help="Independent continuation rollouts in each verification half.",
    )
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=CandidateReturnsConfig.rollout_seed,
        help="Rollout seed.",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=CandidateReturnsConfig.n_shards,
        help="Number of partitions in this rollout population.",
    )
    parser.add_argument(
        "--shard",
        type=int,
        required=True,
        help="Zero-based partition to evaluate in this invocation.",
    )
    parser.add_argument(
        "--restore-atol",
        type=float,
        default=CandidateReturnsConfig.restore_atol,
        help="Absolute tolerance for restored simulator observations.",
    )
    parser.add_argument(
        "--restore-rtol",
        type=float,
        default=CandidateReturnsConfig.restore_rtol,
        help="Relative tolerance for restored simulator observations.",
    )
    parser.add_argument(
        "--continuation-mode",
        type=str,
        choices=("stochastic", "deterministic"),
        default=CandidateReturnsConfig.continuation_mode,
        help="Continuation mode.",
    )
    parser.add_argument(
        "--action-persist",
        type=int,
        default=CandidateReturnsConfig.action_persist,
        help="Action persist.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=CandidateReturnsConfig.dataset_dir,
        help="Local OGBench dataset directory.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_candidate_returns", argv=argv)
    from crlo.eval.evaluate_candidate_returns import evaluate_candidate_returns

    return evaluate_candidate_returns(config)


if __name__ == "__main__":
    raise SystemExit(main())
