"""Training YAML validation and explicit CLI precedence."""

from __future__ import annotations

import pytest
import yaml

from crlo.training.config_schemas import CriticModelConfig, TrainCriticConfig
from crlo.utils.config_loader import CONFIG_DIR, load_config
from experiments.train_critic import parse_args


def test_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("model:\n  width: 1\n  width: 2\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


@pytest.mark.parametrize("contents", ["[1, 2]", "42", "", "true: 3"])
def test_invalid_mapping_rejected(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def _args(tmp_path):
    return [
        "--task",
        "pointmaze-large-navigate-v0",
        "--critic",
        "cosine",
        "--output-dir",
        str(tmp_path),
    ]


def test_explicit_cli_overrides_recipe_and_model_parameters(tmp_path):
    recipe = load_config(CONFIG_DIR / "critic.yaml")
    recipe["steps"] = 13
    recipe["model"]["learning_rate"] = 0.001
    recipe["model"]["hidden_dims"] = [16, 32]
    path = tmp_path / "custom.yaml"
    # Use plain dictionaries because SafeDumper deliberately does not serialize ConfigMap.
    import json

    path.write_text(yaml.safe_dump(json.loads(json.dumps(recipe))))
    cfg = parse_args(_args(tmp_path) + ["--config", str(path), "--steps", "7"])
    assert TrainCriticConfig.from_mapping(cfg).steps == 7
    model = CriticModelConfig(**cfg.model).agent_config("cosine", batch_size=cfg.batch_size)
    assert model["lr"] == 0.001 and model["value_hidden_dims"] == (16, 32)
    assert not (tmp_path / "resolved_config.json").exists()


@pytest.mark.parametrize("field", ["stepps", "jax_platforms", "seed", "output_dir"])
def test_custom_filename_cannot_add_run_or_unknown_fields(tmp_path, field):
    path = tmp_path / "custom.yaml"
    path.write_text((CONFIG_DIR / "critic.yaml").read_text() + f"{field}: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        parse_args(_args(tmp_path) + ["--config", str(path)])


@pytest.mark.parametrize("value", ["true", "'seven'", "0"])
def test_training_budget_validation(tmp_path, value):
    path = tmp_path / "custom.yaml"
    path.write_text(
        (CONFIG_DIR / "critic.yaml").read_text().replace("steps: 1000000", f"steps: {value}")
    )
    with pytest.raises((TypeError, ValueError)):
        parse_args(_args(tmp_path) + ["--config", str(path)])


def test_schema_rejects_unknown_fields(tmp_path):
    cfg = parse_args(_args(tmp_path))
    with pytest.raises(TypeError):
        TrainCriticConfig.from_mapping(cfg | {"misspelled": 1})


def test_critic_only_training_rejects_actor_evaluation(tmp_path):
    args = _args(tmp_path)
    args[args.index("cosine")] = "td_bilinear"
    with pytest.raises(ValueError, match="no rollout actor"):
        parse_args(args)
    assert parse_args(args + ["--eval-interval", "0"]).eval_interval == 0


def test_incomplete_recipe_fails(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("steps: 4\n")
    with pytest.raises(ValueError, match="missing"):
        parse_args(_args(tmp_path) + ["--config", str(path)])
