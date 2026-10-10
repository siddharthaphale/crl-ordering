"""Build rollout manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from crlo.eval.config_schemas import RolloutManifestConfig
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
    parser.add_argument("--val-data-path", type=Path, required=True, help="Val data path.")
    parser.add_argument("--legacy-root", type=Path, required=True, help="Legacy root.")
    parser.add_argument("--legacy-files", type=str, nargs="+", required=True, help="Legacy files.")
    parser.add_argument(
        "--expected-exclusion-digest",
        type=str,
        required=True,
        help="Expected SHA-256 of the frozen exclusion population.",
    )
    parser.add_argument(
        "--expected-pool-sha16",
        type=str,
        required=True,
        help="Expected first 16 hex characters of the candidate-pool SHA-256.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--primary-per-stratum",
        type=int,
        default=RolloutManifestConfig.primary_per_stratum,
        help="Primary per stratum.",
    )
    parser.add_argument(
        "--pilot-ranks",
        type=int,
        nargs="+",
        default=RolloutManifestConfig.pilot_ranks,
        help="Pilot ranks.",
    )
    parser.add_argument(
        "--smoke-ranks",
        type=int,
        nargs="+",
        default=RolloutManifestConfig.smoke_ranks,
        help="Smoke ranks.",
    )
    parser.add_argument(
        "--n-replicates", type=int, default=RolloutManifestConfig.n_replicates, help="N replicates."
    )
    parser.add_argument("--k", type=int, default=RolloutManifestConfig.k, help="K.")
    parser.add_argument(
        "--tie-seed", type=int, default=RolloutManifestConfig.tie_seed, help="Tie seed."
    )
    parser.add_argument(
        "--legacy-name-map-path",
        type=Path,
        default=RolloutManifestConfig.legacy_name_map_path,
        help="Legacy name map path.",
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="build_rollout_manifests", argv=argv)
    from crlo.eval.build_rollout_manifests import build_rollout_manifests

    return build_rollout_manifests(config)


if __name__ == "__main__":
    raise SystemExit(main())
