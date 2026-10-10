"""RunLogger contract: atomic event writes + flushed JSONL streams."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crlo.utils.logging import RunLogger


def test_log_event_writes_atomic_json(tmp_path: Path) -> None:
    log = RunLogger(tmp_path)
    out = log.log_event("summary", {"task": "cube", "score": 0.7})
    assert out == tmp_path / "summary.json"
    assert json.loads(out.read_text()) == {"task": "cube", "score": 0.7}
    assert not (tmp_path / "summary.json.tmp").exists()


def test_log_step_appends_one_line_per_call(tmp_path: Path) -> None:
    with RunLogger(tmp_path) as log:
        log.log_step("training", {"step": 1, "loss": 0.5})
        log.log_step("training", {"step": 2, "loss": 0.4})
    lines = (tmp_path / "training.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 1, "loss": 0.5}
    assert json.loads(lines[1]) == {"step": 2, "loss": 0.4}


def test_log_step_flushes_immediately(tmp_path: Path) -> None:
    log = RunLogger(tmp_path)
    log.log_step("training", {"step": 1})
    # Read without closing — the flush must have committed the line.
    assert (tmp_path / "training.jsonl").read_text() != ""


def test_paths_are_jsonified_via_default(tmp_path: Path) -> None:
    log = RunLogger(tmp_path)
    out = log.log_event("paths", {"out": tmp_path / "x"})
    assert json.loads(out.read_text())["out"] == str(tmp_path / "x")


def test_non_json_serialisable_raises(tmp_path: Path) -> None:
    log = RunLogger(tmp_path)
    with pytest.raises(TypeError):
        log.log_event("bad", {"x": object()})
