"""Validated configuration records for eval commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crlo.agents.model_names import CRITICS
from crlo.utils.config_loader import ConfigMap


def _to_path(value: Any) -> Path:
    if value is None:
        raise ValueError("expected a path, got None")
    return Path(str(value))


def _container(cfg: ConfigMap) -> dict[str, Any]:
    """Copy argument fields before applying typed-schema conversions."""
    return dict(cfg)


@dataclass(frozen=True, kw_only=True)
class ControlledGeometryConfig:
    """Render trained controlled critic geometry without retraining."""

    output_dir: Path
    checkpoint_dir: Path
    seed: int = 42
    grid_size: int = 60
    n_candidates: int = 64
    proposal_std: float = 0.55
    proposal_clip: float = 0.98

    @classmethod
    def from_mapping(cls, cfg: ConfigMap, **overrides: Any):
        """Validate geometry arguments."""
        data = dict(cfg) | overrides
        for key in ("output_dir", "checkpoint_dir"):
            data[key] = _to_path(data[key])
        return cls(**data)

    def __post_init__(self):
        """Check grid and candidate sizes."""
        if self.grid_size < 2 or self.n_candidates < 1:
            raise ValueError("grid_size must be >=2 and n_candidates >=1")


@dataclass(frozen=True, kw_only=True)
class RolloutManifestConfig:
    """Config for ``crlo.eval.build_rollout_manifests``."""

    task: str
    pools_dir: Path
    val_data_path: Path
    legacy_root: Path
    legacy_files: tuple[str, ...]
    expected_exclusion_digest: str
    expected_pool_sha16: str
    output_dir: Path
    primary_per_stratum: int = 100
    pilot_ranks: tuple[int, ...] = (450, 451, 452, 453, 454, 455, 456, 457, 458, 459)
    smoke_ranks: tuple[int, ...] = (498, 499)
    n_replicates: int = 5
    k: int = 64
    tie_seed: int = 7002

    legacy_name_map_path: str | None = None

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> RolloutManifestConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in ("pools_dir", "val_data_path", "legacy_root", "output_dir"):
            data[key] = _to_path(data[key])
        for key in ("legacy_files", "pilot_ranks", "smoke_ranks"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate score-blindness, rank ranges, and digest formats."""
        _refuse_score_paths(
            self.pools_dir,
            self.val_data_path,
            self.legacy_root,
            self.legacy_name_map_path,
            self.output_dir,
            *self.legacy_files,
        )
        if not self.legacy_files:
            raise ValueError("legacy_files must be non-empty")
        if len(self.expected_exclusion_digest) != 64:
            raise ValueError("expected_exclusion_digest must be sha256 hex")
        if len(self.expected_pool_sha16) != 16:
            raise ValueError("expected_pool_sha16 must be 16 hex chars")
        if self.primary_per_stratum < 1:
            raise ValueError("primary_per_stratum must be >= 1")
        if self.n_replicates < 1 or self.k < 2:
            raise ValueError("n_replicates >= 1 and k >= 2 required")
        pool = set(self.pilot_ranks) | set(self.smoke_ranks)
        if len(pool) != len(self.pilot_ranks) + len(self.smoke_ranks):
            raise ValueError("pilot_ranks and smoke_ranks must be disjoint")


@dataclass(frozen=True, kw_only=True)
class CandidateReturnsConfig:
    """Config for ``crlo.eval.evaluate_candidate_returns``."""

    task: str
    pools_dir: Path
    manifest_path: Path
    diffusion_ckpt: Path
    continuation_sha256: str
    expected_pool_sha16: str
    output_dir: Path
    val_data_path: Path | None = None
    horizon: int = 300
    discount: float = 0.99
    rollouts_per_half: int = 5
    rollout_seed: int = 7001
    n_shards: int = 12
    shard: int
    restore_atol: float = 1e-05
    restore_rtol: float = 1e-05
    continuation_mode: str = "stochastic"
    action_persist: int = 1

    dataset_dir: str = "data/ogbench"

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> CandidateReturnsConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in ("pools_dir", "manifest_path", "diffusion_ckpt", "output_dir"):
            data[key] = _to_path(data[key])
        if data.get("val_data_path") is not None:
            data["val_data_path"] = _to_path(data["val_data_path"])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate score-blindness, shard bounds, and budget ranges."""
        _refuse_score_paths(
            self.pools_dir,
            self.manifest_path,
            self.diffusion_ckpt,
            self.output_dir,
            self.val_data_path,
        )
        if len(self.continuation_sha256) != 64:
            raise ValueError("continuation_sha256 must be sha256 hex")
        if len(self.expected_pool_sha16) != 16:
            raise ValueError("expected_pool_sha16 must be 16 hex chars")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {self.horizon}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1]; got {self.discount}")
        if self.rollouts_per_half < 1:
            raise ValueError("rollouts_per_half must be >= 1")
        if self.n_shards < 1 or not 0 <= self.shard < self.n_shards:
            raise ValueError(
                f"need 0 <= shard < n_shards; got shard={self.shard}, n_shards={self.n_shards}"
            )
        if self.restore_atol <= 0 or self.restore_rtol < 0:
            raise ValueError("restore tolerances must be positive")
        if self.continuation_mode not in CONTINUATION_MODES:
            raise ValueError(
                f"continuation_mode must be one of {CONTINUATION_MODES}; "
                f"got {self.continuation_mode!r}"
            )
        if self.continuation_mode == "deterministic" and self.rollouts_per_half != 1:
            raise ValueError(
                "deterministic requires rollouts_per_half=1 (the two "
                "halves are the duplicate-verification executions)"
            )
        if not 1 <= self.action_persist <= self.horizon:
            raise ValueError(f"action_persist must be in [1, horizon]; got {self.action_persist}")


@dataclass(frozen=True, kw_only=True)
class ValueReliabilityConfig:
    """Config for ``crlo.eval.check_value_reliability``."""

    mode: str
    manifest_path: Path
    merged_path: Path
    output_dir: Path
    k: int = 64
    rollouts_per_half: int = 5
    bootstrap_seed: int = 7003
    n_boot: int = 10000
    frac_target: float = 0.5
    agreement_target: float = 0.7
    h_min: float = 0.012
    pilot_ub_q: float = 0.9
    gate_lb_q: float = 0.05

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> ValueReliabilityConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in ("manifest_path", "merged_path", "output_dir"):
            data[key] = _to_path(data[key])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate mode vocab, score-blindness, and threshold ranges."""
        if self.mode not in RELIABILITY_MODES:
            raise ValueError(f"mode must be one of {RELIABILITY_MODES}; got {self.mode!r}")
        _refuse_score_paths(self.manifest_path, self.merged_path, self.output_dir)
        min_r = 1 if self.mode == "deterministic_primary" else 2
        if self.k < 2 or self.rollouts_per_half < min_r:
            raise ValueError(f"k >= 2 and rollouts_per_half >= {min_r} required")
        if self.n_boot < 1:
            raise ValueError("n_boot must be >= 1")
        for name, v in (
            ("frac_target", self.frac_target),
            ("agreement_target", self.agreement_target),
            ("pilot_ub_q", self.pilot_ub_q),
            ("gate_lb_q", self.gate_lb_q),
        ):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1); got {v}")
        if self.h_min <= 0.0:
            raise ValueError(f"h_min must be > 0; got {self.h_min}")


@dataclass(frozen=True, kw_only=True)
class RolloutBudgetConfig:
    """Config for ``crlo.eval.calibrate_rollout_budget``."""

    manifest_path: Path
    merged_path: Path
    baseline_merged_path: Path
    output_dir: Path
    k: int = 64
    rollouts_per_half: int = 25
    r_grid: tuple[int, ...] = (5, 10, 15, 20, 25)
    r_extrap: tuple[int, ...] = (35, 50)
    r_choices: tuple[int, ...] = (25, 35, 50)
    agreement_target: float = 0.7
    frac_target: float = 0.5

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> RolloutBudgetConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in ("manifest_path", "merged_path", "baseline_merged_path", "output_dir"):
            data[key] = _to_path(data[key])
        for key in ("r_grid", "r_extrap", "r_choices"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate score-blindness, grid structure, and targets."""
        _refuse_score_paths(
            self.manifest_path, self.merged_path, self.baseline_merged_path, self.output_dir
        )
        if list(self.r_grid) != sorted(set(self.r_grid)):
            raise ValueError("r_grid must be strictly increasing")
        if max(self.r_grid) != self.rollouts_per_half:
            raise ValueError("r_grid must end at rollouts_per_half")
        if any(r <= self.rollouts_per_half for r in self.r_extrap):
            raise ValueError("r_extrap values must exceed rollouts_per_half")
        known = set(self.r_grid) | set(self.r_extrap)
        if not set(self.r_choices) <= known:
            raise ValueError("r_choices must come from r_grid or r_extrap")
        if list(self.r_choices) != sorted(set(self.r_choices)):
            raise ValueError("r_choices must be strictly increasing")
        for name, v in (
            ("agreement_target", self.agreement_target),
            ("frac_target", self.frac_target),
        ):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1); got {v}")


@dataclass(frozen=True, kw_only=True)
class ActionPersistenceConfig:
    """Config for ``crlo.eval.analyze_action_persistence``."""

    manifest_path: Path
    pools_dir: Path
    merged_root: Path
    output_dir: Path
    m_grid: tuple[int, ...] = (5, 10, 20)
    frac_target: float = 0.5
    h_min: float = 0.012
    smoothness_target: float = 0.15

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> ActionPersistenceConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in ("manifest_path", "pools_dir", "merged_root", "output_dir"):
            data[key] = _to_path(data[key])
        if "m_grid" in data:
            data["m_grid"] = tuple(data["m_grid"])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate score-blindness, grid, and targets."""
        _refuse_score_paths(self.manifest_path, self.pools_dir, self.merged_root, self.output_dir)
        if list(self.m_grid) != sorted(set(self.m_grid)) or min(self.m_grid) < 2:
            raise ValueError("m_grid must be strictly increasing, all >= 2")
        for name, v in (
            ("frac_target", self.frac_target),
            ("smoothness_target", self.smoothness_target),
        ):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1); got {v}")
        if self.h_min <= 0:
            raise ValueError("h_min must be > 0")


@dataclass(frozen=True, kw_only=True)
class ExecutedValueOrderingConfig:
    """Config for ``crlo.eval.evaluate_executed_value_ordering``."""

    mode: str
    manifest_path: Path
    merged_path: Path
    reliability_path: Path
    scores_dir: Path
    retrieval_csv: Path
    resample_indices_path: Path
    output_dir: Path
    critics: tuple[str, ...] = ("td_q", "raw_bilinear", "cosine", "hybrid")
    seeds: tuple[int, ...] = (42,)
    comparison_path: Path | None = None
    k: int = 64
    rollouts_per_half: int = 5
    k_curve: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    delta_tau: float = 0.1
    delta_v: float = 0.006
    tau_cap: float = 0.5
    auc_min: float = 0.95
    top_decile_size: int = 7

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> ExecutedValueOrderingConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in (
            "manifest_path",
            "merged_path",
            "reliability_path",
            "scores_dir",
            "retrieval_csv",
            "resample_indices_path",
            "output_dir",
        ):
            data[key] = _to_path(data[key])
        if data.get("comparison_path") is not None:
            data["comparison_path"] = _to_path(data["comparison_path"])
        for key in ("critics", "seeds", "k_curve"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate mode/critic vocab, seed sets, and thresholds."""
        if self.mode not in COMPARISON_MODES:
            raise ValueError(f"mode must be one of {COMPARISON_MODES}; got {self.mode!r}")
        unknown = set(self.critics) - set(CRITICS)
        if unknown:
            raise ValueError(f"unknown critics {sorted(unknown)}")
        for c in ("td_q", "raw_bilinear", "cosine"):
            if c not in self.critics:
                raise ValueError(f"critics must include {c!r}")
        if self.mode == "seed_replication" and tuple(self.seeds) == (42,):
            raise ValueError("seed_replication requires the fresh seeds, not only 42")
        if self.mode == "seed_replication" and self.comparison_path is None:
            raise ValueError("seed_replication requires comparison_path")
        if list(self.k_curve) != sorted(set(self.k_curve)):
            raise ValueError("k_curve must be strictly increasing")
        if max(self.k_curve) > self.k:
            raise ValueError("k_curve entries cannot exceed k")
        if not 1 <= self.top_decile_size <= self.k:
            raise ValueError("top_decile_size must be in [1, k]")
        for name, v in (("delta_tau", self.delta_tau), ("delta_v", self.delta_v)):
            if v <= 0:
                raise ValueError(f"{name} must be > 0; got {v}")
        if not 0.0 < self.auc_min <= 1.0:
            raise ValueError("auc_min must be in (0, 1]")
        if not -1.0 <= self.tau_cap <= 1.0:
            raise ValueError("tau_cap must be in [-1, 1]")


@dataclass(frozen=True, kw_only=True)
class CandidateScoringConfig:
    """Config for ``crlo.eval.score_candidate_pools``."""

    task: str
    pools_dir: Path
    manifest_path: Path
    checkpoint_dir: Path
    critic: str
    label: str
    training_seed: int
    expected_pool_sha16: str
    output_dir: Path
    chunk: int = 8192

    @classmethod
    def from_mapping(cls, cfg: ConfigMap, **overrides: Any) -> CandidateScoringConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        for key in ("pools_dir", "manifest_path", "checkpoint_dir", "output_dir"):
            data[key] = _to_path(data[key])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate critic vocab and identity fields."""
        if self.critic not in CRITICS:
            raise ValueError(f"critic must be one of {CRITICS}; got {self.critic!r}")
        if not self.label:
            raise ValueError("label must be non-empty")
        if len(self.expected_pool_sha16) != 16:
            raise ValueError("expected_pool_sha16 must be 16 hex chars")
        if self.chunk < 1:
            raise ValueError("chunk must be >= 1")


@dataclass(frozen=True, kw_only=True)
class FrozenGoalRetrievalConfig:
    """Config for ``crlo.eval.evaluate_frozen_goal_retrieval``."""

    mode: str
    task: str
    manifest_path: Path
    output_dir: Path
    discount: float = 0.99
    n_triples: int = 3000
    kmax: int = 60
    seed: int = 0
    checkpoint_dir: Path | None = None
    critic: str | None = None
    label: str | None = None
    training_seed: int | None = None

    @classmethod
    def from_mapping(
        cls,
        cfg: ConfigMap,
        **overrides: Any,
    ) -> FrozenGoalRetrievalConfig:
        """Build from a loaded ConfigMap + optional field overrides."""
        data = _container(cfg) | overrides
        data["manifest_path"] = _to_path(data["manifest_path"])
        data["output_dir"] = _to_path(data["output_dir"])
        if data.get("checkpoint_dir") is not None:
            data["checkpoint_dir"] = _to_path(data["checkpoint_dir"])
        return cls(**data)

    def __post_init__(self) -> None:
        """Validate mode vocab and score-mode identity fields."""
        if self.mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}; got {self.mode!r}")
        if self.n_triples < 2 or self.kmax < 1:
            raise ValueError("n_triples >= 2 and kmax >= 1 required")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1]; got {self.discount}")
        if self.mode == "score":
            missing = [
                name
                for name, v in (
                    ("checkpoint_dir", self.checkpoint_dir),
                    ("critic", self.critic),
                    ("label", self.label),
                    ("training_seed", self.training_seed),
                )
                if v is None
            ]
            if missing:
                raise ValueError(f"score mode requires {missing}")
            if self.critic not in CRITICS:
                raise ValueError(f"critic must be one of {CRITICS}; got {self.critic!r}")


def _refuse_score_paths(*values: Any) -> None:
    """Reject any path naming a critic score file (the score-blind protocol).

    The Frozen executed-value rollout/manifest/precheck stages may not load, join,
    inspect, or name any ``scores_*.npz`` (plan §6); configs enforce it
    structurally so a miswired argument fails before any work runs.
    """
    for value in values:
        if value is None:
            continue
        name = Path(str(value)).name
        if name.startswith("scores_") or name == "scores.npz":
            raise ValueError(f"score-blind stage received a score path: {value!s}")


RELIABILITY_MODES = ("pilot", "primary", "deterministic_primary")
CONTINUATION_MODES = ("stochastic", "deterministic")
COMPARISON_MODES = ("critic_comparison", "seed_replication")
RETRIEVAL_MODES = ("build_manifest", "score")
