"""Frozen executed-value frozen query-manifest construction (plan §5.1).

Deterministic, score-blind derivation of the primary / pilot / smoke
query manifests from the cached candidate pools, the validation terminal
boundaries, and the frozen legacy-exclusion audit artifacts. No function
in this module may read, join, or name a critic score file.

Frozen rules implemented here:

- legacy exclusions come from an immutable list of audit files whose
  sorted ``<relative-path> <file-sha256>`` manifest must reproduce the
  preregistered digest; the excluded IDs are the union of their
  ``query`` columns (derived from IDs, never assumed to be a range);
- stratum ranks are positions in frozen pool-index order within each
  stratum and are never reindexed after exclusions;
- pool replicate is ``original_frozen_stratum_rank mod 5`` for every
  manifest, including pilot and smoke;
- ``episode_id`` is ``searchsorted(flatnonzero(val_terminals), t_index,
  side="left")`` on the frozen validation terminal vector.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from crlo.eval.value_statistics import tie_priority

STRATUM_NAMES = ("near", "medium", "far")


def sha256_file(path: Path) -> str:
    """Full SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Full SHA-256 hex digest of a bytes payload."""
    return hashlib.sha256(data).hexdigest()


def exclusion_manifest_digest(
    root: Path, rel_paths: list[str], historical_names: dict[str, str] | None = None
) -> tuple[str, list[str]]:
    """Digest of the sorted ``<relative-path> <file-sha256>`` manifest.

    Returns ``(digest, lines)``; the digest freezes the exclusion inputs
    so later artifact additions/removals cannot change the population.
    """
    names = historical_names or {}
    if names and set(names) != set(rel_paths):
        raise ValueError("Historical-name mapping must cover exactly the exclusion inputs")
    lines = sorted(f"{names.get(rel, rel)} {sha256_file(Path(root) / rel)}" for rel in rel_paths)
    payload = ("\n".join(lines) + "\n").encode()
    return sha256_bytes(payload), lines


def excluded_query_ids(root: Path, rel_paths: list[str]) -> list[int]:
    """Union of the ``query`` columns of the frozen audit CSVs, sorted."""
    ids: set[int] = set()
    for rel in rel_paths:
        with (Path(root) / rel).open(newline="") as f:
            for row in csv.DictReader(f):
                ids.add(int(row["query"]))
    return sorted(ids)


def derive_episode_ids(val_terminals: np.ndarray, t_index: np.ndarray) -> np.ndarray:
    """Frozen episode-ID rule (plan §5.1)."""
    bounds = np.flatnonzero(np.asarray(val_terminals) > 0)
    return np.searchsorted(bounds, np.asarray(t_index), side="left")


def stratum_ranks(stratum: np.ndarray) -> np.ndarray:
    """Rank of each query within its stratum, in frozen pool-index order."""
    stratum = np.asarray(stratum)
    ranks = np.empty(len(stratum), np.int64)
    for s in np.unique(stratum):
        idx = np.flatnonzero(stratum == s)
        ranks[idx] = np.arange(len(idx))
    return ranks


def select_by_rank(
    stratum: np.ndarray,
    ranks: np.ndarray,
    excluded: set[int],
    wanted_ranks: dict[int, list[int]],
) -> list[int]:
    """Query IDs whose (stratum, rank) matches the frozen selection.

    ``wanted_ranks`` maps stratum -> explicit rank list (pilot/smoke).
    Excluded IDs may not appear; selection is by rank, never reindexed.
    """
    out: list[int] = []
    for s, rank_list in wanted_ranks.items():
        for r in rank_list:
            q = int(np.flatnonzero((np.asarray(stratum) == s) & (ranks == r))[0])
            if q in excluded:
                raise ValueError(
                    f"frozen rank {r} of stratum {s} (query {q}) collides with a legacy exclusion"
                )
            out.append(q)
    return out


def primary_query_ids(
    stratum: np.ndarray, ranks: np.ndarray, excluded: set[int], per_stratum: int
) -> list[int]:
    """First ``per_stratum`` non-excluded queries per stratum by rank."""
    out: list[int] = []
    for s in sorted(np.unique(np.asarray(stratum)).tolist()):
        idx = np.flatnonzero(np.asarray(stratum) == s)
        idx = idx[np.argsort(ranks[idx])]
        keep = [int(q) for q in idx if int(q) not in excluded][:per_stratum]
        if len(keep) < per_stratum:
            raise ValueError(
                f"stratum {s}: only {len(keep)} non-excluded queries; need {per_stratum}"
            )
        out.extend(keep)
    return out


def build_manifest_rows(
    query_ids: list[int],
    stratum: np.ndarray,
    ranks: np.ndarray,
    episode_ids: np.ndarray,
    n_replicates: int,
    tie_seed: int,
    k: int,
) -> list[dict]:
    """Manifest rows with the frozen round-robin replicate + tie priority."""
    rows = []
    for q in query_ids:
        rep = int(ranks[q]) % n_replicates
        prio = tie_priority(tie_seed, int(q), rep, k)
        rows.append(
            {
                "query_id": int(q),
                "stratum": int(np.asarray(stratum)[q]),
                "stratum_name": STRATUM_NAMES[int(np.asarray(stratum)[q])],
                "stratum_rank": int(ranks[q]),
                "pool_replicate": rep,
                "episode_id": int(episode_ids[q]),
                "tie_priority": prio.tolist(),
            }
        )
    return rows


def manifest_payload(rows: list[dict], meta: dict) -> dict:
    """Manifest JSON payload with a content digest over its rows."""
    body = json.dumps(rows, sort_keys=True).encode()
    return {
        "meta": meta,
        "n_queries": len(rows),
        "rows_sha256": sha256_bytes(body),
        "rows": rows,
    }


def load_manifest(path: Path) -> dict:
    """Load a manifest and verify its row digest."""
    payload = json.loads(Path(path).read_text())
    body = json.dumps(payload["rows"], sort_keys=True).encode()
    if sha256_bytes(body) != payload["rows_sha256"]:
        raise ValueError(f"manifest row digest mismatch: {path}")
    return payload
