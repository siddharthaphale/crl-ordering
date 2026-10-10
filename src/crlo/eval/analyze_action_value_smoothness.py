"""Action-space smoothness diagnostic for exact executed values (exploratory).

For each rollout-resolvable query of a deterministic-judge merged tensor,
computes the Spearman correlation between pairwise candidate action
distance and absolute value difference, and the ratio of mean |dV| over
the nearest quartile of action pairs to the all-pairs mean. Values that
are smooth in action space give a clearly positive correlation and a
ratio well below 1; chaotic values give ~0 and ~1.

Exploratory interpretation aid (labeled as such in the paper); not a
gate input. Writes ``action_smoothness.json`` + per-query CSV.

Usage::

    python experiments/analyze_action_value_smoothness.py \
        <pools_dir> <manifest_json> <merged_npz> <gate_json> <output_dir>
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.eval.query_manifests import load_manifest  # noqa: E402
from crlo.eval.value_statistics import action_value_smoothness  # noqa: E402


def query_smoothness(actions: np.ndarray, values: np.ndarray) -> dict:
    """Distance-value Spearman + near-pair ratio for one query."""
    rho, ratio = action_value_smoothness(actions, values)
    return {"rho_dist_value": rho, "near_pair_ratio": ratio}


def analyze_action_value_smoothness(configuration: ConfigMap) -> int:
    """Analyze action-distance and value differences on resolvable queries."""
    cfg = configuration
    pools_dir, man_path, merged_path, gate_path, out_dir = (
        Path(cfg[key])
        for key in ("pools_dir", "manifest_path", "merged_path", "reliability_path", "output_dir")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    man = load_manifest(man_path)
    gate = json.loads(gate_path.read_text())
    mask = np.asarray(gate["resolvable_mask"], bool)
    v = np.load(merged_path)["disc_return"][:, :, 0, 0]
    A = np.load(pools_dir / "pools.npz")["A"]

    rows_out, rhos, ratios = [], [], []
    for i, row in enumerate(man["rows"]):
        if not mask[i]:
            continue
        s = query_smoothness(
            A[row["query_id"], row["pool_replicate"]].astype(np.float64), v[i].astype(np.float64)
        )
        rows_out.append({"query_id": row["query_id"], "stratum": row["stratum_name"], **s})
        rhos.append(s["rho_dist_value"])
        ratios.append(s["near_pair_ratio"])

    with (out_dir / "action_smoothness_queries.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    rhos = np.asarray(rhos)
    ratios = np.asarray(ratios)
    summary = {
        "kind": "exploratory action-space smoothness diagnostic",
        "n_resolvable": int(mask.sum()),
        "rho_dist_value_median": float(np.median(rhos)),
        "rho_dist_value_iqr": [float(np.quantile(rhos, 0.25)), float(np.quantile(rhos, 0.75))],
        "near_pair_ratio_median": float(np.nanmedian(ratios)),
        "merged": str(merged_path),
        "gate": str(gate_path),
    }
    (out_dir / "action_smoothness.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[smoothness] median rho={summary['rho_dist_value_median']:.3f} "
        f"near-ratio={summary['near_pair_ratio_median']:.3f} "
        f"-> {out_dir}",
        flush=True,
    )
    return 0
