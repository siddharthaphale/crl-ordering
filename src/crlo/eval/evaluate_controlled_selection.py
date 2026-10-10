"""Toy candidate-selection decomposition: extrapolation vs objective mismatch.

At s=0 with goal delta drawn *inside* the support disk (optimum a*=delta is
in-support), candidates span in- and off-support. For each critic's selected
action we record regret and whether it is off-support (||a|| > R), splitting the
failure into extrapolation (off-support selection) vs in-support mismatch.

The same evaluation applies to both primary and two-head critic checkpoints.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function
from crlo.envs.toy_2d import Toy2DConfig  # noqa: E402
from crlo.eval.critic_scoring import critic_score  # noqa: E402
from crlo.training.checkpoints import load_critic


def evaluate_controlled_selection(configuration: ConfigMap) -> int:
    """Measure controlled regret and off-support selection on shared pools.

    Loads saved primary or two-head critics and compares their selections with
    the exact zero-noise target. Query goals lie inside the support disk, and
    ``query_seed`` determines goals and candidate pools for every critic.

    Args:
        configuration: Checkpoint/output paths, critics, support/proposal settings,
            query count, candidate count, and query seed from the matching command.
            Its ``output_dir`` field is converted to a ``Path`` in place.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    args = configuration
    args["output_dir"] = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    critics = args.critics
    K = args.candidates

    cfg = (
        Toy2DConfig(support_radius=args.support_radius)
        if args.support_radius is not None
        else Toy2DConfig()
    )
    R = cfg.support_radius
    qr = np.random.default_rng(args.query_seed)
    deltas = []
    while len(deltas) < args.n_queries:
        d = qr.uniform(-R, R, size=2)
        if np.linalg.norm(d) <= args.goal_radius_fraction * R:
            deltas.append(d)
    deltas = np.array(deltas, np.float32)
    pools = [
        np.clip(
            qr.normal(scale=args.proposal_radius_multiplier * R, size=(K, 2)),
            cfg.action_low,
            cfg.action_high,
        )
        for _ in range(args.n_queries)
    ]
    s0 = np.zeros(2, np.float32)

    rows = []
    for critic in critics:
        agent, metadata = load_critic(
            Path(args.checkpoint_dir) / critic, critic_rebuild_function(critic)
        )
        if metadata["environment"]["noise_std"] != 0:
            raise ValueError("Exact values require deterministic controlled training")
        if metadata["environment"]["support_radius"] != R:
            raise ValueError("Evaluation support radius differs from the training environment")
        regrets, off_sel, sel_norms = [], [], []
        for delta, cand in zip(deltas, pools, strict=True):
            g = (s0 + delta).astype(np.float32)
            sb = np.broadcast_to(s0, (K, 2)).astype(np.float32)
            gb = np.broadcast_to(g, (K, 2)).astype(np.float32)
            sc = critic_score(agent, critic, sb, gb, cand.astype(np.float32))
            qstar = -np.linalg.norm(s0 + cand - g, axis=1)
            idx = int(np.argmax(sc))
            regrets.append(float(qstar.max() - qstar[idx]))
            off_sel.append(np.linalg.norm(cand[idx]) > R)
            sel_norms.append(float(np.linalg.norm(cand[idx])))
        regrets = np.asarray(regrets)
        off_mask = np.asarray(off_sel, bool)
        # Decompose regret by whether the selected action was off/in support.
        regret_off = float(regrets[off_mask].mean()) if off_mask.any() else float("nan")
        regret_in = float(regrets[~off_mask].mean()) if (~off_mask).any() else float("nan")
        row = {
            "critic": critic,
            "regret": float(regrets.mean()),
            "off_support_pct": 100.0 * float(off_mask.mean()),
            "mean_sel_norm": float(np.mean(sel_norms)),
            "regret_off": regret_off,
            "regret_in": regret_in,
            "support_radius": R,
        }
        rows.append(row)
        print(
            f"[controlled_selection] {critic:12s} regret={row['regret']:.3f} "
            f"off-support={row['off_support_pct']:.0f}% "
            f"mean||sel||={row['mean_sel_norm']:.3f} "
            f"reg|off={regret_off:.3f} reg|in={regret_in:.3f} (R={R:.2f})",
            flush=True,
        )

    out = args.output_dir / "controlled_selection.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[controlled_selection] wrote {out} ({len(rows)} rows)")
    return 0
