"""Measure selection of actions outside the local dataset support."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.envs.ogbench_loader import load_crl_env_and_dataset  # noqa: E402
from crlo.eval.critic_scoring import critic_score  # noqa: E402
from crlo.eval.trajectory_queries import sample_trajectory_queries  # noqa: E402
from crlo.training.checkpoints import load_critic  # noqa: E402

CRITICS = ("td_q", "raw_bilinear", "cosine", "hybrid")


def _score(agent, critic, S, G, A, batch=4096):
    out = []
    for i in range(0, len(S), batch):
        out.append(
            critic_score(agent, critic, S[i : i + batch], G[i : i + batch], A[i : i + batch])
        )
    return np.concatenate(out).reshape(-1)


def evaluate_local_support(configuration: ConfigMap) -> int:
    """Measure selection from local versus globally sampled dataset actions.

    For each query, mixes actions from nearby states with actions drawn from
    the full dataset. The off-support label denotes the global proposal source,
    not a measured distance threshold. Shared nested pools use first-index
    argmax tie breaking. Writes ``local_support_<task>_kcurve.csv``.

    Args:
        configuration: Task, critics, triple sampling settings, neighbor count,
            candidate counts, seed, and checkpoint/output paths from the command.

    Returns:
        Zero after successful completion. Input and checkpoint errors propagate
        to the caller; output directories are created as needed.
    """
    cfg = configuration
    task = cfg.task
    critics = list(cfg.get("critics", CRITICS))
    gamma = float(cfg["discount"])
    n = int(cfg["n_triples"])
    kmax = int(cfg["kmax"])
    seed = int(cfg["seed"])
    n_near = int(cfg["n_near"])
    Ks = list(cfg["k_grid"])
    evals_root = Path(cfg["evals_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Kmax = max(Ks)

    _, ds = load_crl_env_and_dataset(env_name=task, discount=gamma)
    obs = np.asarray(ds.dataset["observations"], np.float32)
    act = np.asarray(ds.dataset["actions"], np.float32)
    S, A, G, _ = sample_trajectory_queries(ds, gamma, n, kmax, seed)
    N, adim = len(S), act.shape[1]

    # Labeled candidate pools: on-support = action at a near state; off-support =
    # action at a uniformly random state. Both real dataset actions.
    rng = np.random.default_rng(seed)
    tree = cKDTree(obs)
    _, nn = tree.query(S, k=n_near)
    on = act[nn[np.arange(N)[:, None], rng.integers(0, n_near, (N, Kmax))]]
    off = act[rng.integers(0, len(act), (N, Kmax))]
    lbl = rng.integers(0, 2, (N, Kmax))  # 1 = off-support slot
    cand = np.where(lbl[..., None] == 1, off, on).astype(np.float32)
    print(f"[drift] {task}: {N} ref triples, adim={adim}, base off-rate=0.5", flush=True)

    Sb = np.repeat(S[:, None, :], Kmax, 1).reshape(N * Kmax, -1)
    Gb = np.repeat(G[:, None, :], Kmax, 1).reshape(N * Kmax, -1)
    rows = []
    for critic in critics:
        d = evals_root / f"{critic}_{task}"
        if not (d / "params.msgpack").exists():
            raise FileNotFoundError(d / "params.msgpack")
        agent, _ = load_critic(d, critic_rebuild_function(critic))
        sc = _score(agent, critic, Sb, Gb, cand.reshape(N * Kmax, adim)).reshape(N, Kmax)
        crow = []
        for K in Ks:
            pick = np.argmax(sc[:, :K], axis=1)
            off_rate = float(lbl[np.arange(N), pick].mean())
            base = float(lbl[:, :K].mean())
            rows.append(
                {
                    "task": task,
                    "critic": critic,
                    "K": K,
                    "off_rate": off_rate,
                    "base_rate": base,
                    "excess_off_rate": off_rate - base,
                }
            )
            crow.append(off_rate - base)
        print(
            f"[drift]   {critic:12s} excess off-rate: "
            + " ".join(f"K{k}={v:+.3f}" for k, v in zip(Ks, crow, strict=True)),
            flush=True,
        )

    out = output_dir / f"local_support_{task}_kcurve.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["task", "critic", "K", "off_rate", "base_rate", "excess_off_rate"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[drift] wrote {out}", flush=True)
    return 0
