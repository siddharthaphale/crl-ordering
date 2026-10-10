"""Frozen executed-value manifest construction: exclusions, ranks, round-robin (plan §10.2)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from crlo.eval.query_manifests import (
    build_manifest_rows,
    derive_episode_ids,
    excluded_query_ids,
    exclusion_manifest_digest,
    load_manifest,
    manifest_payload,
    primary_query_ids,
    select_by_rank,
    stratum_ranks,
)


def _write_audit_csvs(root, n_files=4, ids_per_file=(0, 1, 2)):
    rels = []
    for i in range(n_files):
        rel = f"shard{i}.csv"
        rows = "\n".join(f"{q + i},near,x,0,0,1,10,0.5" for q in ids_per_file)
        (root / rel).write_text(
            f"query,stratum,tags,pool_index,rollout,success,steps_to_goal,disc_return\n{rows}\n"
        )
        rels.append(rel)
    return rels


def test_exclusion_digest_deterministic_and_order_free(tmp_path):
    rels = _write_audit_csvs(tmp_path)
    d1, lines1 = exclusion_manifest_digest(tmp_path, rels)
    d2, _ = exclusion_manifest_digest(tmp_path, list(reversed(rels)))
    assert d1 == d2 and len(d1) == 64
    assert lines1 == sorted(lines1)
    # touching an input changes the digest
    (tmp_path / rels[0]).write_text((tmp_path / rels[0]).read_text().replace("0.5", "0.6"))
    d3, _ = exclusion_manifest_digest(tmp_path, rels)
    assert d3 != d1


def test_excluded_ids_from_query_columns(tmp_path):
    rels = _write_audit_csvs(tmp_path, n_files=3, ids_per_file=(0, 5))
    assert excluded_query_ids(tmp_path, rels) == [0, 1, 2, 5, 6, 7]


def test_derive_episode_ids_searchsorted_left():
    terminals = np.array([0, 0, 1, 0, 0, 1, 0, 1])
    #                   bounds at 2, 5, 7
    t_index = np.array([0, 2, 3, 5, 6, 7])
    got = derive_episode_ids(terminals, t_index)
    want = np.searchsorted(np.array([2, 5, 7]), t_index, side="left")
    assert np.array_equal(got, want)
    assert got.tolist() == [0, 0, 1, 1, 2, 2]


def test_stratum_ranks_frozen_pool_order():
    stratum = np.array([0, 0, 1, 0, 1, 2])
    assert stratum_ranks(stratum).tolist() == [0, 1, 0, 2, 1, 0]


def _population(n_per=40):
    stratum = np.repeat([0, 1, 2], n_per)
    ranks = stratum_ranks(stratum)
    return stratum, ranks


def test_primary_selection_skips_exclusions_never_reindexes():
    stratum, ranks = _population()
    excluded = set(range(3)) | {40, 41, 42} | {80, 81, 82}
    ids = primary_query_ids(stratum, ranks, excluded, 5)
    assert ids[:5] == [3, 4, 5, 6, 7]
    assert ids[5:10] == [43, 44, 45, 46, 47]
    assert len(ids) == 15
    with pytest.raises(ValueError, match="non-excluded"):
        primary_query_ids(stratum, ranks, set(range(120)), 5)


def test_select_by_rank_collision_raises():
    stratum, ranks = _population()
    with pytest.raises(ValueError, match="collides"):
        select_by_rank(stratum, ranks, {2}, {0: [2]})
    got = select_by_rank(stratum, ranks, set(), {0: [38, 39], 1: [38, 39], 2: [38, 39]})
    assert got == [38, 39, 78, 79, 118, 119]


def test_round_robin_balance_mod5():
    # 100 consecutive ranks -> exactly 20 queries per replicate per stratum.
    stratum = np.repeat([0, 1, 2], 140)
    ranks = stratum_ranks(stratum)
    excluded = {int(q) for q in np.flatnonzero(ranks < 34)}
    ids = primary_query_ids(stratum, ranks, excluded, 100)
    episode_ids = np.zeros(len(stratum), np.int64)
    rows = build_manifest_rows(ids, stratum, ranks, episode_ids, 5, 7002, 8)
    for s in range(3):
        reps = [r["pool_replicate"] for r in rows if r["stratum"] == s]
        assert sorted(np.bincount(reps).tolist()) == [20] * 5
    for r in rows:
        assert r["pool_replicate"] == r["stratum_rank"] % 5
        assert sorted(r["tie_priority"]) == list(range(8))


def test_manifest_payload_roundtrip_and_tamper(tmp_path):
    stratum, ranks = _population()
    rows = build_manifest_rows([0, 40, 80], stratum, ranks, np.zeros(120, np.int64), 5, 7002, 4)
    payload = manifest_payload(rows, {"task": "t"})
    p = tmp_path / "m.json"
    p.write_text(json.dumps(payload))
    loaded = load_manifest(p)
    assert loaded["rows"] == rows
    payload["rows"][0]["pool_replicate"] = 3
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_manifest(p)
