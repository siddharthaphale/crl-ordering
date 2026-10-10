"""Build candidate pools."""

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
    parser.add_argument("--n-per-stratum", type=int, default=500, help="N per stratum.")
    parser.add_argument("--n-replicates", type=int, default=5, help="N replicates.")
    parser.add_argument("--k-max", type=int, default=64, help="K max.")
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument("--diffusion-ckpt", type=Path, required=True, help="Diffusion ckpt.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir.")
    parser.add_argument(
        "--dataset-dir", type=Path, default="data/ogbench", help="Local OGBench dataset directory."
    )
    return ConfigMap(vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    """Run this experiment with the supplied arguments."""
    config = parse_args(argv)
    log_run_config(config, command="build_candidate_pools", argv=argv)
    from crlo.eval.build_candidate_pools import build_candidate_pools

    return build_candidate_pools(config)


if __name__ == "__main__":
    raise SystemExit(main())
