"""Direct experiment commands parse arguments before importing model backends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = sorted(path.stem for path in (ROOT / "experiments").glob("*.py"))


@pytest.mark.parametrize("name", COMMANDS)
def test_command_help_without_backends_or_checkout_working_directory(name, tmp_path):
    program = """
import importlib.abc
import runpy
import sys
class NoBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'flax', 'ogbench', 'mujoco'}:
            raise AssertionError('Backend imported for help: ' + fullname)
sys.meta_path.insert(0, NoBackends())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT / "experiments" / f"{name}.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert ("--config" in result.stdout) == name.startswith("train_")
    assert "--runtime-config" not in result.stdout
    assert "--cpu-threads" not in result.stdout
    assert not list(tmp_path.iterdir())


def test_direct_command_honors_custom_recipe_and_preserves_environment(tmp_path, monkeypatch):
    from crlo.training import train_critic
    from crlo.utils.config_loader import CONFIG_DIR
    from experiments import train_critic as command

    recipe = tmp_path / "custom.yaml"
    recipe.write_text((CONFIG_DIR / "critic.yaml").read_text())
    calls = []
    monkeypatch.setattr(train_critic, "train_critic", lambda cfg: calls.append(cfg) or 0)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    before = dict(os.environ)
    args = [
        f"--config={recipe}",
        "--task",
        "pointmaze-large-navigate-v0",
        "--critic",
        "cosine",
        "--steps",
        "7",
        "--output-dir",
        str(tmp_path / "output"),
    ]
    assert command.main(args) == 0
    assert len(calls) == 1
    assert calls[0].steps == 7
    record = json.loads((tmp_path / "output/resolved_config.json").read_text())
    assert record["config"]["steps"] == 7
    assert record["argv"] == args
    assert record["runtime"]["environment"]["OMP_NUM_THREADS"] == "3"
    assert dict(os.environ) == before


def test_unknown_argument_fails_before_loading_implementation(tmp_path, monkeypatch):
    from crlo.training import train_critic
    from experiments import train_critic as command

    def unexpected_call(config):
        pytest.fail("Experiment ran with an invalid CLI")

    monkeypatch.setattr(train_critic, "train_critic", unexpected_call)
    with pytest.raises(SystemExit):
        command.main(["--output-dir", str(tmp_path / "output"), "--stepps", "7"])
    assert not (tmp_path / "output").exists()
