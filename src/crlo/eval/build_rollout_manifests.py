"""Frozen executed-value manifest builder: frozen primary/pilot/smoke query manifests.

Score-blind by construction (plan §5.1, §6): reads only ``pools.npz``,
pool metadata, the raw validation split (terminal boundaries), and the
frozen legacy-exclusion audit CSVs. Refuses critic score inputs at the
config layer and never opens a score file.

Outputs under ``output_dir`` (plan §10.3 manifests/ namespace):

- ``legacy_exclusions_antmaze.json`` — immutable exclusion input list,
  per-file SHA-256, aggregate digest, and the excluded query-ID set;
- ``primary_<task-prefix>.json`` / ``pilot_…`` / ``smoke_…`` — manifest
  rows: ``query_id, stratum, stratum_name, stratum_rank,
  pool_replicate, episode_id, tie_priority`` plus provenance hashes.

JSON manifests carry a ``rows_sha256`` content digest; every downstream
driver verifies it on load.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.eval.config_schemas import RolloutManifestConfig
from crlo.eval.query_manifests import (  # noqa: E402
    build_manifest_rows,
    derive_episode_ids,
    excluded_query_ids,
    exclusion_manifest_digest,
    manifest_payload,
    primary_query_ids,
    select_by_rank,
    sha256_file,
    stratum_ranks,
)


def _git_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()


def build_all(cfg: RolloutManifestConfig) -> dict[str, Path]:
    """Build and write the exclusion record + three manifests."""
    task_prefix = cfg.task.split("-")[0]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pools_npz = cfg.pools_dir / "pools.npz"
    pool_sha16 = sha256_file(pools_npz)[:16]
    if pool_sha16 != cfg.expected_pool_sha16:
        raise ValueError(f"pools.npz sha256[:16]={pool_sha16} != frozen {cfg.expected_pool_sha16}")
    z = np.load(pools_npz)
    stratum = np.asarray(z["stratum"])
    t_index = np.asarray(z["t_index"])
    if z["A"].shape[1:3] != (cfg.n_replicates, cfg.k):
        raise ValueError(f"pool shape {z['A'].shape} != (*, {cfg.n_replicates}, {cfg.k}, *)")

    val = np.load(cfg.val_data_path)
    episode_ids = derive_episode_ids(val["terminals"], t_index)
    val_sha16 = sha256_file(cfg.val_data_path)[:16]

    historical_names = (
        json.loads(Path(cfg.legacy_name_map_path).read_text()) if cfg.legacy_name_map_path else None
    )
    digest, lines = exclusion_manifest_digest(
        cfg.legacy_root, list(cfg.legacy_files), historical_names
    )
    if digest != cfg.expected_exclusion_digest:
        raise ValueError(
            f"legacy exclusion digest {digest} != frozen {cfg.expected_exclusion_digest}"
        )
    excluded = excluded_query_ids(cfg.legacy_root, list(cfg.legacy_files))
    exclusion_payload = {
        "task": cfg.task,
        "legacy_root": str(cfg.legacy_root),
        "manifest_lines": lines,
        "manifest_sha256": digest,
        "excluded_query_ids": excluded,
        "n_excluded": len(excluded),
        "git": _git_hash(),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    exclusion_path = cfg.output_dir / f"legacy_exclusions_{task_prefix}.json"
    exclusion_path.write_text(json.dumps(exclusion_payload, indent=2))

    ranks = stratum_ranks(stratum)
    excluded_set = set(excluded)
    meta_common = {
        "task": cfg.task,
        "pool_sha16": pool_sha16,
        "val_data_sha16": val_sha16,
        "exclusion_manifest_sha256": digest,
        "n_replicates": cfg.n_replicates,
        "k": cfg.k,
        "tie_seed": cfg.tie_seed,
        "git": _git_hash(),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_paths = {"exclusions": exclusion_path}

    primary_ids = primary_query_ids(stratum, ranks, excluded_set, cfg.primary_per_stratum)
    aux = {
        "pilot": {s: list(cfg.pilot_ranks) for s in range(3)},
        "smoke": {s: list(cfg.smoke_ranks) for s in range(3)},
    }
    for name, qids in (
        ("primary", primary_ids),
        ("pilot", select_by_rank(stratum, ranks, excluded_set, aux["pilot"])),
        ("smoke", select_by_rank(stratum, ranks, excluded_set, aux["smoke"])),
    ):
        overlap = set(qids) & set(primary_ids) if name != "primary" else set()
        if overlap:
            raise ValueError(f"{name} manifest overlaps primary: {overlap}")
        rows = build_manifest_rows(
            qids, stratum, ranks, episode_ids, cfg.n_replicates, cfg.tie_seed, cfg.k
        )
        payload = manifest_payload(rows, meta_common | {"manifest": name})
        path = cfg.output_dir / f"{name}_{task_prefix}.json"
        path.write_text(json.dumps(payload, indent=2))
        out_paths[name] = path
        n_ep = len({r["episode_id"] for r in rows})
        print(
            f"[query_manifestss] {name}: {len(rows)} queries, {n_ep} episode clusters -> {path}",
            flush=True,
        )
    return out_paths


def build_rollout_manifests(configuration: ConfigMap) -> int:
    """Construct hash-bound rollout manifests from frozen candidate pools."""
    cfg_raw = configuration
    cfg = RolloutManifestConfig.from_mapping(cfg_raw)
    build_all(cfg)
    return 0
