"""Evaluate frozen goal retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.agents.model_names import CRITICS
from crlo.eval.config_schemas import FrozenGoalRetrievalConfig
from crlo.utils.config_loader import ConfigMap
from crlo.utils.logging import log_run_config


def parse_args(argv: list[str] | None = None) -> ConfigMap:
    """Parse explicit experiment arguments before importing model backends."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode", type=str, choices=("build_manifest", "score"), required=True, help="Mode."
    )
    parser.add_argument("--task", type=str, default="antmaze-large-navigate-v0", help="Task.")
    parser.add_argument(
        "--manifest-path", type=Path, required=True, help="Frozen query manifest to evaluate."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--discount", type=float, default=FrozenGoalRetrievalConfig.discount, help="Discount."
    )
    parser.add_argument(
        "--n-triples",
        type=int,
        default=FrozenGoalRetrievalConfig.n_triples,
        help="Number of sampled state-action-goal trajectory triples.",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=FrozenGoalRetrievalConfig.kmax,
        help="Maximum temporal distance to a sampled goal.",
    )
    parser.add_argument("--seed", type=int, default=FrozenGoalRetrievalConfig.seed, help="Seed.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=FrozenGoalRetrievalConfig.checkpoint_dir,
        help="Checkpoint dir.",
    )
    parser.add_argument(
        "--critic",
        choices=CRITICS,
        type=str,
        default=FrozenGoalRetrievalConfig.critic,
        help="Critic.",
    )
    parser.add_argument("--label", type=str, default=FrozenGoalRetrievalConfig.label, help="Label.")
    parser.add_argument(
        "--training-seed",
        type=int,
        default=FrozenGoalRetrievalConfig.training_seed,
        help="Training seed recorded for the evaluated checkpoint.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="evaluate_frozen_goal_retrieval", argv=argv)
    from crlo.eval.evaluate_frozen_goal_retrieval import evaluate_frozen_goal_retrieval

    return evaluate_frozen_goal_retrieval(config)


if __name__ == "__main__":
    raise SystemExit(main())
