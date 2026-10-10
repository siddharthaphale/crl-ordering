"""Real-data training commands honor YAML parameters and save independent checkpoints."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
import yaml

from crlo.agents.critic_factory import build_critic_from_metadata
from crlo.training.checkpoints import load_critic, load_diffusion, save_critic
from crlo.utils.config_loader import CONFIG_DIR
from experiments import train_critic, train_diffusion

TASK = "pointmaze-large-navigate-v0"
DATASET_DIR = CONFIG_DIR.parent / "data/ogbench"


def _reject_yaml_read(*_):
    pytest.fail("Checkpoint reconstruction attempted to read YAML")


def test_critic_training_recipe_reaches_saved_model(tmp_path, monkeypatch):
    trained = []

    def capture_checkpoint(agent, directory, meta):
        trained.append(agent)
        return save_critic(agent, directory, meta)

    monkeypatch.setattr("crlo.training.checkpoints.save_critic", capture_checkpoint)
    recipe = yaml.safe_load((CONFIG_DIR / "critic.yaml").read_text())
    recipe.update(steps=2, batch_size=8)
    recipe["model"].update(
        hidden_dims=[16], latent_dim=8, learning_rate=0.001, cosine_temperature=0.2
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(recipe))
    output = tmp_path / "critic"
    assert (
        train_critic.main(
            [
                "--config",
                str(path),
                "--task",
                TASK,
                "--critic",
                "cosine",
                "--output-dir",
                str(output),
                "--dataset-dir",
                str(DATASET_DIR),
                "--steps",
                "3",
                "--eval-interval",
                "0",
            ]
        )
        == 0
    )
    path.unlink()
    # No model construction or reconstruction is allowed to read a recipe.
    monkeypatch.setattr(
        "crlo.utils.config_loader.load_config",
        _reject_yaml_read,
    )
    agent, metadata = load_critic(output, build_critic_from_metadata, resume=True)
    for actual, expected in zip(
        jax.tree.leaves(agent.network.params),
        jax.tree.leaves(trained[0].network.params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    initial = build_critic_from_metadata(metadata)
    assert int(agent.network.step) - int(initial.network.step) == 3
    assert tuple(agent.config["value_hidden_dims"]) == (16,)
    assert agent.config["lr"] == 0.001
    assert agent.config["cosine_temperature"] == 0.2
    assert agent.config["batch_size"] == 8
    assert metadata["steps"] == 3
    assert all(
        next(iter(leaf.devices())).platform == "gpu"
        for leaf in jax.tree.leaves(trained[0].network.params)
    )


def test_diffusion_training_recipe_reaches_saved_model(tmp_path, monkeypatch):
    recipe = yaml.safe_load((CONFIG_DIR / "diffusion.yaml").read_text())
    recipe.update(
        steps=2,
        batch_size=8,
        hidden_dims=[16],
        denoising_steps=4,
        time_embed_dim=8,
        schedule="linear",
        lr=0.001,
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(recipe))
    output = tmp_path / "diffusion"
    assert (
        train_diffusion.main(
            [
                "--config",
                str(path),
                "--task",
                TASK,
                "--output-dir",
                str(output),
                "--dataset-dir",
                str(DATASET_DIR),
                "--steps",
                "3",
                "--validation-samples",
                "8",
            ]
        )
        == 0
    )
    path.unlink()
    monkeypatch.setattr(
        "crlo.utils.config_loader.load_config",
        _reject_yaml_read,
    )
    policy = load_diffusion(output, resume=True)
    assert policy.steps == 3 and policy.lr == 0.001 and policy.schedule == "linear"
    assert policy.T == 4 and policy.network.time_embed_dim == 8
    assert tuple(policy.network.hidden_dims) == (16,)
    validation = json.loads((output / "validation.json").read_text())
    assert validation["split"] == "official_validation"
    assert np.isfinite(validation["validation_denoising_loss"])
    assert all(
        next(iter(leaf.devices())).platform == "gpu" for leaf in jax.tree.leaves(policy.params)
    )
