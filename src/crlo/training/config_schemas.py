"""Validated configuration records for training commands."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

from crlo.agents.model_names import CRITICS
from crlo.utils.config_loader import ConfigMap


def _to_path(value: Any) -> Path:
    if value is None:
        raise ValueError("expected a path, got None")
    return Path(str(value))


def _container(cfg: ConfigMap) -> dict[str, Any]:
    """Copy supplied fields before applying typed-schema conversions."""
    return dict(cfg)


@dataclass(frozen=True, kw_only=True)
class TrainCriticConfig:
    """Config for ``crlo.training.train_critic``."""

    task: str
    critic: str
    output_dir: Path
    seed: int = 42
    steps: int
    eval_interval: int = 100000
    # Rollout episodes per eval. 250 = OGBench paper-grade; lower (e.g. 50)
    # trades success-rate precision for much faster evals in pilots.
    eval_episodes: int = 50
    batch_size: int
    discount: float
    bc_alpha: float

    model: dict

    dataset_dir: str = "data/ogbench"

    @classmethod
    def from_mapping(cls, cfg: ConfigMap, **overrides: Any) -> TrainCriticConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        data["output_dir"] = _to_path(data["output_dir"])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate critic name, ranges, and training budgets."""
        _validate_numeric_fields(self)
        CriticModelConfig(**self.model)
        if self.critic not in CRITICS:
            raise ValueError(f"critic must be one of {CRITICS}; got {self.critic!r}")
        critic_only = self.critic in ("td_bilinear", "contrastive_mlp") or (
            self.critic.startswith("two_head_") and not self.critic.endswith("_with_actor")
        )
        if critic_only and self.eval_interval:
            raise ValueError(
                "Critic-only models require eval_interval=0; they have no rollout actor"
            )
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1; got {self.steps}")
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval must be >= 0; got {self.eval_interval}")
        if self.eval_episodes < 1:
            raise ValueError(f"eval_episodes must be >= 1; got {self.eval_episodes}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1]; got {self.discount}")
        if self.bc_alpha < 0.0:
            raise ValueError(f"bc_alpha must be >= 0; got {self.bc_alpha}")

    RECIPE_FIELDS: ClassVar[set[str]] = {"model", "batch_size", "steps", "bc_alpha", "discount"}

    @classmethod
    def validate_recipe(cls, config: dict) -> None:
        """Reject runtime fields and incomplete training specifications."""
        _validate_recipe(config, cls.RECIPE_FIELDS)


@dataclass(frozen=True, kw_only=True)
class TrainDiffusionConfig:
    """Config for ``crlo.training.train_diffusion``."""

    task: str
    output_dir: Path
    seed: int = 42
    steps: int
    denoising_steps: int
    lr: float
    batch_size: int
    discount: float
    log_interval: int = 10000
    hidden_dims: tuple[int, ...]
    schedule: str
    validation_samples: int = 4096
    dataset_dir: str = "data/ogbench"

    time_embed_dim: int

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> TrainDiffusionConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        data["output_dir"] = _to_path(data["output_dir"])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate dim / range invariants for the DDPM run."""
        _validate_numeric_fields(self)
        if self.time_embed_dim < 2 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be a positive even width")
        if self.denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1; got {self.denoising_steps}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1; got {self.steps}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {self.lr}")
        if self.schedule not in ("cosine", "linear"):
            raise ValueError("Unsupported diffusion schedule")
        if min(self.batch_size, self.log_interval, self.validation_samples) < 1:
            raise ValueError("Batch, logging, and validation sizes must be positive")

    RECIPE_FIELDS: ClassVar[set[str]] = {
        "time_embed_dim",
        "steps",
        "discount",
        "lr",
        "schedule",
        "batch_size",
        "hidden_dims",
        "denoising_steps",
    }

    @classmethod
    def validate_recipe(cls, config: dict) -> None:
        """Reject runtime fields and incomplete training specifications."""
        _validate_recipe(config, cls.RECIPE_FIELDS)


def _validate_recipe(config: dict, expected: set[str]) -> None:
    unknown, missing = config.keys() - expected, expected - config.keys()
    if unknown or missing:
        raise ValueError(f"Training config: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _validate_numeric_fields(config) -> None:
    for field in fields(config):
        value = getattr(config, field.name)
        if field.type == "int" and type(value) is not int:
            raise TypeError(f"{field.name} must be an integer")
        if field.type == "float" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise TypeError(f"{field.name} must be a number")


@dataclass(frozen=True, kw_only=True)
class CriticModelConfig:
    """Explicit model hyperparameters supplied by either critic training recipe."""

    learning_rate: float
    hidden_dims: tuple[int, ...]
    latent_dim: int
    layer_norm: bool
    cosine_temperature: float
    td_warmup_steps: int
    tau: float
    expectile: float
    num_negatives: int
    value_p_curgoal: float
    value_p_trajgoal: float
    value_p_randomgoal: float

    def __post_init__(self):
        """Validate training hyperparameters before model construction."""
        _validate_numeric_fields(self)
        if self.learning_rate <= 0 or self.cosine_temperature <= 0:
            raise ValueError("Learning rate and cosine temperature must be positive")
        if not self.hidden_dims or any(
            type(width) is not int or width < 1 for width in self.hidden_dims
        ):
            raise ValueError("Hidden dimensions must be positive integers")
        if self.latent_dim < 1 or self.num_negatives < 1 or self.td_warmup_steps < 0:
            raise ValueError("Invalid model width, negative count, or warmup")
        if type(self.layer_norm) is not bool:
            raise TypeError("layer_norm must be boolean")
        if not 0 < self.tau <= 1 or not 0 < self.expectile < 1:
            raise ValueError("Invalid target update rate or expectile")
        probabilities = (self.value_p_curgoal, self.value_p_trajgoal, self.value_p_randomgoal)
        if any(p < 0 or p > 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-8:
            raise ValueError("Goal sampling probabilities must sum to one")

    def agent_config(self, critic: str, *, batch_size: int, controlled: bool = False) -> dict:
        """Translate training parameters to the selected agent's native fields."""
        config = dict(
            lr=self.learning_rate,
            batch_size=batch_size,
            actor_hidden_dims=tuple(self.hidden_dims),
            value_hidden_dims=tuple(self.hidden_dims),
            layer_norm=self.layer_norm,
        )
        if critic != "td_q":
            config["latent_dim"] = self.latent_dim
        if critic in ("cosine", "hybrid") or critic.startswith("two_head_"):
            config["cosine_temperature"] = self.cosine_temperature
        if critic == "hybrid":
            config["td_warmup_steps"] = self.td_warmup_steps
        if critic in ("td_q", "td_bilinear") or critic.startswith("two_head_"):
            config.update(tau=self.tau, expectile=self.expectile)
        if critic == "contrastive_mlp":
            config["num_negatives"] = self.num_negatives
        if critic == "td_q" and not controlled:
            config.update(
                value_p_curgoal=self.value_p_curgoal,
                value_p_trajgoal=self.value_p_trajgoal,
                value_p_randomgoal=self.value_p_randomgoal,
            )
        return config


@dataclass(frozen=True, kw_only=True)
class ControlledCriticConfig:
    """Controlled training hyperparameters and explicit run arguments."""

    output_dir: Path
    steps: int
    batch_size: int
    n_train: int
    bc_alpha: float
    discount: float
    model: dict
    seed: int = 42
    critics: tuple[str, ...] = ("raw_bilinear", "cosine", "td_q", "hybrid")
    seed_protocol: str = "geometry"
    critic_seed_stride: int = 1000
    model_seed_stride: int = 100
    RECIPE_FIELDS: ClassVar[set[str]] = {
        "steps",
        "batch_size",
        "n_train",
        "bc_alpha",
        "discount",
        "model",
    }

    @classmethod
    def validate_recipe(cls, config: dict) -> None:
        """Require a complete training specification without run arguments."""
        _validate_recipe(config, cls.RECIPE_FIELDS)

    @classmethod
    def from_mapping(cls, config: dict):
        """Validate explicit training and run parameters."""
        return cls(**config)

    def __post_init__(self):
        """Validate training hyperparameters before model construction."""
        _validate_numeric_fields(self)
        CriticModelConfig(**self.model)
        if min(self.steps, self.batch_size, self.n_train) < 1:
            raise ValueError("Training budgets must be positive")
        if not 0 < self.discount <= 1 or self.bc_alpha < 0:
            raise ValueError("Invalid discount or actor regularization")
        if self.seed_protocol not in ("geometry", "selection"):
            raise ValueError("Unknown controlled seed protocol")
        if not self.critics or set(self.critics) - {"raw_bilinear", "cosine", "td_q", "hybrid"}:
            raise ValueError("Unknown controlled critic")
