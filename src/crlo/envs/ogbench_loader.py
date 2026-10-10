"""OGBench environment and dataset loading.

All vendor imports are lazy (inside function bodies) so this module is safe
to import without OGBench cloned.
"""

from __future__ import annotations

from pathlib import Path

from crlo.utils.ogbench_vendor import ensure_ogbench


def load_raw_env_and_datasets(env_name: str, dataset_dir: str | None = None) -> tuple:
    """Load compact official splits, preserving the published flat row indices."""
    from ogbench import make_env_and_datasets

    directory = dataset_dir or str(Path(__file__).resolve().parents[3] / "data/ogbench")
    env, train, validation = make_env_and_datasets(
        env_name, compact_dataset=True, dataset_dir=directory
    )
    env.reset()
    return env, train, validation


def load_crl_env_and_dataset(
    env_name: str,
    discount: float,
    split: str = "train",
    dataset_config: dict | None = None,
    dataset_dir: str | None = None,
) -> tuple:
    """Load OGBench env and GCDataset with the CRL goal-sampling config.

    Uses trajectory-only goal sampling (gc_negative=False), matching the
    published CRL baseline (OGBench hyperparameters.sh).

    Returns:
        (env, train_dataset)
    """
    ensure_ogbench()
    from utils.datasets import Dataset, GCDataset  # noqa: PLC0415

    print(f"Loading environment and dataset: {env_name}", flush=True)
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    env, train_raw, val_raw = load_raw_env_and_datasets(env_name, dataset_dir)

    sampling_config = dict(
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=False,
        p_aug=0.0,
        frame_stack=None,
        discount=discount,
        discrete=False,
    )
    if dataset_config is not None:
        sampling_config.update(dataset_config)
    raw = train_raw if split == "train" else val_raw
    if raw is None:
        raise ValueError("The requested dataset split is unavailable")
    train_dataset = GCDataset(Dataset.create(**raw), sampling_config)
    print(
        f"Dataset loaded. {split} size: {raw['observations'].shape[0]}",
        flush=True,
    )
    return env, train_dataset
