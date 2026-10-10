"""Frozen executed-value shard artifact contract: atomic resume + merge rejection (plan §10.2)."""

from __future__ import annotations

import numpy as np
import pytest

from crlo.eval.rollout_shards import (
    merge_shards,
    shard_complete,
    shard_paths,
    write_shard,
)

K, R = 4, 2


def _rows(n=3):
    return [{"query_id": 10 * (i + 1)} for i in range(n)]


def _cells(qids, k=K, r=R):
    q, c, h, ro = [], [], [], []
    for qq in qids:
        for cc in range(k):
            for hh in range(2):
                for rr in range(r):
                    q.append(qq), c.append(cc), h.append(hh), ro.append(rr)
    n = len(q)
    return {
        "query_id": np.array(q),
        "candidate": np.array(c),
        "half": np.array(h),
        "rollout": np.array(ro),
        "success": np.ones(n, np.int64),
        "steps_to_goal": np.full(n, 3),
        "disc_return": np.full(n, 0.97),
    }


SIDE = {"pool_sha16": "abc", "manifest_rows_sha256": "def"}


def test_write_merge_roundtrip_and_resume(tmp_path):
    rows = _rows()
    write_shard(tmp_path, 0, _cells([10, 30]), SIDE)
    write_shard(tmp_path, 1, _cells([20]), SIDE)
    assert shard_complete(tmp_path, 0) and shard_complete(tmp_path, 1)
    assert not shard_complete(tmp_path, 2)
    got = merge_shards(tmp_path, rows, K, R, expected_sidecar=SIDE)
    assert got["disc_return"].shape == (3, K, 2, R)
    assert np.allclose(got["disc_return"], 0.97)
    # no tmp litter after atomic writes
    assert not list(tmp_path.glob("*.tmp"))


def test_query_disjoint_shard_slicing_exhaustive():
    # The driver shards manifest rows as rows[shard::n_shards]; verify
    # disjoint + exhaustive for an uneven split.
    rows = list(range(11))
    parts = [rows[s::4] for s in range(4)]
    flat = sorted(x for p in parts for x in p)
    assert flat == rows
    assert sum(len(p) for p in parts) == len(rows)


def test_merge_rejects_missing_cell(tmp_path):
    cells = _cells([10])
    for f in cells:
        cells[f] = cells[f][:-1]  # drop one cell
    write_shard(tmp_path, 0, cells, SIDE)
    with pytest.raises(ValueError, match="missing cells"):
        merge_shards(tmp_path, _rows(1), K, R)


def test_merge_rejects_duplicate_cell(tmp_path):
    cells = _cells([10])
    for f in cells:
        cells[f] = np.concatenate([cells[f], cells[f][-1:]])
    write_shard(tmp_path, 0, cells, SIDE)
    with pytest.raises(ValueError, match="duplicate cell"):
        merge_shards(tmp_path, _rows(1), K, R)


def test_merge_rejects_unknown_query_and_wrong_hash(tmp_path):
    write_shard(tmp_path, 0, _cells([99]), SIDE)
    with pytest.raises(ValueError, match="not in manifest"):
        merge_shards(tmp_path, _rows(1), K, R)
    with pytest.raises(ValueError, match="sidecar"):
        merge_shards(tmp_path, _rows(1), K, R, expected_sidecar={"pool_sha16": "OTHER"})


def test_merge_rejects_wrong_shape_and_missing_field(tmp_path):
    write_shard(tmp_path, 0, _cells([10]), SIDE)
    with pytest.raises(ValueError, match="missing cells"):
        merge_shards(tmp_path, _rows(1), K, R + 1)  # wrong rollout count
    with pytest.raises(ValueError, match="missing cell fields"):
        write_shard(tmp_path, 1, {"query_id": np.array([1])}, SIDE)


def test_merge_rejects_missing_sidecar(tmp_path):
    write_shard(tmp_path, 0, _cells([10]), SIDE)
    shard_paths(tmp_path, 0)[1].unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        merge_shards(tmp_path, _rows(1), K, R)


def test_merge_rejects_nonfinite_returns(tmp_path):
    cells = _cells([10])
    cells["disc_return"] = cells["disc_return"].astype(float)
    cells["disc_return"][0] = np.nan
    write_shard(tmp_path, 0, cells, SIDE)
    with pytest.raises(ValueError, match="non-finite"):
        merge_shards(tmp_path, _rows(1), K, R)


def test_no_shards_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shard"):
        merge_shards(tmp_path, _rows(1), K, R)


def test_resume_rejects_corruption_and_changed_protocol(tmp_path):
    write_shard(tmp_path, 0, _cells([10]), SIDE)
    assert shard_complete(tmp_path, 0, SIDE)
    with pytest.raises(ValueError, match="protocol mismatch"):
        shard_complete(tmp_path, 0, {"pool_sha16": "changed"})
    artifact, _ = shard_paths(tmp_path, 0)
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        shard_complete(tmp_path, 0)
