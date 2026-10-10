"""Frozen executed-value shard artifact contract and merge (plan §10.3).

Each rollout shard writes a compressed NPZ plus a JSON sidecar via an
atomic rename. The merge reconstructs dense tensors of shape
``(n_manifest, K, 2, rollouts_per_half)`` and fails on any missing or
duplicated cell; completed valid shards are immutable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CELL_FIELDS = (
    "query_id",
    "candidate",
    "half",
    "rollout",
    "success",
    "steps_to_goal",
    "disc_return",
)


def shard_paths(shard_dir: Path, shard: int) -> tuple[Path, Path]:
    """NPZ + JSON paths for one shard."""
    d = Path(shard_dir)
    return d / f"shard_{shard:04d}.npz", d / f"shard_{shard:04d}.json"


def write_shard(shard_dir: Path, shard: int, cells: dict[str, np.ndarray], sidecar: dict) -> None:
    """Atomically write one shard's NPZ + JSON sidecar."""
    npz_path, json_path = shard_paths(shard_dir, shard)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [k for k in CELL_FIELDS if k not in cells]
    if missing:
        raise ValueError(f"shard {shard}: missing cell fields {missing}")
    tmp = npz_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **cells)
    tmp.replace(npz_path)
    jtmp = json_path.with_suffix(".json.tmp")
    jtmp.write_text(
        json.dumps(
            {**sidecar, "artifact_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest()},
            indent=2,
            sort_keys=True,
        )
    )
    jtmp.replace(json_path)


def shard_complete(shard_dir: Path, shard: int, expected_sidecar: dict | None = None) -> bool:
    """Check committed artifact integrity and optional protocol identity."""
    npz_path, json_path = shard_paths(shard_dir, shard)
    if not (npz_path.exists() and json_path.exists()):
        return False
    meta = json.loads(json_path.read_text())
    recorded = meta.get("artifact_sha256")
    if recorded and hashlib.sha256(npz_path.read_bytes()).hexdigest() != recorded:
        raise ValueError(f"Corrupt shard artifact: {npz_path}")
    for key, expected in (expected_sidecar or {}).items():
        if meta.get(key) != expected:
            raise ValueError(f"Shard protocol mismatch for {key}: {json_path}")
    return True


def merge_shards(
    shard_dir: Path,
    manifest_rows: list[dict],
    k: int,
    rollouts_per_half: int,
    expected_sidecar: dict | None = None,
) -> dict[str, np.ndarray]:
    """Merge all shards into dense (n, K, 2, R) tensors.

    Fails on: no shards, a duplicated cell, a missing cell, an unknown
    query, or (when ``expected_sidecar`` is given) any sidecar whose
    listed keys disagree with the expected provenance hashes.
    """
    shard_dir = Path(shard_dir)
    npzs = sorted(shard_dir.glob("shard_*.npz"))
    if not npzs:
        raise FileNotFoundError(f"no shard NPZs under {shard_dir}")
    qids = [r["query_id"] for r in manifest_rows]
    qpos = {q: i for i, q in enumerate(qids)}
    n = len(qids)
    shape = (n, k, 2, rollouts_per_half)
    ret = np.full(shape, np.nan, np.float64)
    suc = np.full(shape, -1, np.int64)
    stp = np.full(shape, -1, np.int64)
    seen = np.zeros(shape, bool)
    for npz_path in npzs:
        sidecar_path = npz_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"missing sidecar for {npz_path}")
        sidecar = json.loads(sidecar_path.read_text())
        recorded = sidecar.get("artifact_sha256")
        if recorded and hashlib.sha256(npz_path.read_bytes()).hexdigest() != recorded:
            raise ValueError(f"Corrupt shard artifact: {npz_path}")
        if expected_sidecar:
            for key, want in expected_sidecar.items():
                got = sidecar.get(key)
                if got != want:
                    raise ValueError(
                        f"{sidecar_path}: sidecar[{key!r}]={got!r} != expected {want!r}"
                    )
        z = np.load(npz_path)
        for field in CELL_FIELDS:
            if field not in z:
                raise ValueError(f"{npz_path}: missing field {field!r}")
        for q, c, h, r, s, st, y in zip(
            z["query_id"],
            z["candidate"],
            z["half"],
            z["rollout"],
            z["success"],
            z["steps_to_goal"],
            z["disc_return"],
            strict=True,
        ):
            if int(q) not in qpos:
                raise ValueError(f"{npz_path}: query {int(q)} not in manifest")
            idx = (qpos[int(q)], int(c), int(h), int(r))
            if seen[idx]:
                raise ValueError(f"{npz_path}: duplicate cell {idx}")
            seen[idx] = True
            ret[idx] = float(y)
            suc[idx] = int(s)
            stp[idx] = int(st)
    if not seen.all():
        n_missing = int((~seen).sum())
        first = tuple(int(v) for v in np.argwhere(~seen)[0])
        raise ValueError(
            f"merge incomplete: {n_missing} missing cells "
            f"(first missing (query_pos, cand, half, rollout)={first})"
        )
    if not np.isfinite(ret).all():
        raise ValueError("merged returns contain non-finite values")
    return {"disc_return": ret, "success": suc, "steps_to_goal": stp}


def merged_path(merge_dir: Path) -> Path:
    """Canonical merged-artifact path for one manifest namespace."""
    return Path(merge_dir) / "full_pool_rollouts.npz"


def write_merged(merge_dir: Path, tensors: dict[str, np.ndarray], meta: dict) -> Path:
    """Atomically write the merged tensors + JSON meta."""
    out = merged_path(merge_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **tensors)
    tmp.replace(out)
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out
