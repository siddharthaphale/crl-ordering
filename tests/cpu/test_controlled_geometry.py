"""Controlled training and evaluation communicate only through saved checkpoints."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.evaluate_controlled_geometry import main as evaluate
from experiments.train_controlled_critic import main as train


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled")
    checkpoints = root / "checkpoints"
    assert (
        train(
            [
                "--output-dir",
                str(checkpoints),
                "--steps",
                "1",
                "--batch-size",
                "4",
                "--n-train",
                "64",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert (
        evaluate(
            [
                "--output-dir",
                str(root),
                "--checkpoint-dir",
                str(checkpoints),
                "--grid-size",
                "4",
                "--n-candidates",
                "2",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    return root


@pytest.mark.parametrize("critic", ["raw_bilinear", "cosine", "td_q", "hybrid"])
def test_geometry_reads_checkpoint(critic: str, outputs: Path):
    assert (outputs / f"{critic}_grid.npy").exists()
    assert (outputs / f"{critic}_gradnorm.npy").exists()
    with (outputs / f"{critic}_candidates.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
