"""OGBench optimizer's curse: best-of-K candidate maximization vs gamma^d value.

The OGBench analogue of the closed-form-Q* toy (Fig 5). Off-support Q* is not
estimable on OGBench, and closed-loop best-of-K is confounded (one-step Q is
nearly action-flat for far goals -> selection is noise-dominated). We therefore
test the thesis with an *in-support* value signal that has resolving power:
gamma^d return-to-go on dataset triples.

For each critic we sample N in-distribution triples (s, a, g=future state) with
known value V = gamma^d, score them with the deployed scalar, then simulate
best-of-K candidate maximization by drawing random K-subsets, taking the
score-argmax member, and recording BOTH:
  selected value  = mean gamma^d of the score-argmax pick   (the objective)
  selected score  = mean critic score of the pick           (the proxy)
vs K. A value-calibrated critic's selected value rises toward the oracle
(pool-max gamma^d) as K grows; a miscalibrated one's selected *score* rises
(proxy improves) while its selected *value* falls (objective degrades) -- the
optimizer's curse. Regret_K = oracle_value - selected_value.

Candidates are in-support triples (mixed goals), so this measures the deployed
*scalar* under argmax over a proposal, the OGBench-feasible form of candidate
maximization; the literal fixed-state/vary-action test is confounded by flat
one-step value (see closed-loop note). Writes cross_query_selection_{task}.csv (per-triple
score+value) and cross_query_selection_{task}_kcurve.csv (the K-curves).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from crlo.utils.config_loader import ConfigMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from crlo.agents.critic_factory import critic_rebuild_function  # noqa: E402
from crlo.envs.ogbench_loader import load_crl_env_and_dataset  # noqa: E402
from crlo.eval.critic_scoring import critic_score  # noqa: E402
from crlo.eval.trajectory_queries import sample_trajectory_queries  # noqa: E402
from crlo.training.checkpoints import load_critic  # noqa: E402

CRITICS = ("td_q", "raw_bilinear", "cosine", "hybrid")


def _score(agent, critic, S, G, A, batch=512):
    out = []
    for i in range(0, len(S), batch):
        out.append(
            critic_score(agent, critic, S[i : i + batch], G[i : i + batch], A[i : i + batch])
        )
    return np.concatenate(out).reshape(-1)


def _kcurve(score, value, Ks, n_pools, seed):
    """Best-of-K selected value/score + oracle, by bootstrap over K-subsets."""
    rng = np.random.default_rng(seed)
    n = len(value)
    rows = []
    for K in Ks:
        idx = rng.integers(0, n, size=(n_pools, K))
        winner = idx[np.arange(n_pools), np.argmax(score[idx], axis=1)]
        sel_val = float(value[winner].mean())
        sel_score = float(score[winner].mean())
        oracle = float(value[idx].max(axis=1).mean())
        rows.append(
            {
                "K": K,
                "selected_value": sel_val,
                "selected_score": sel_score,
                "oracle_value": oracle,
                "regret": oracle - sel_val,
                "random_value": float(value.mean()),
            }
        )
    return rows


def evaluate_cross_query_selection(configuration: ConfigMap) -> int:
    """Compare best-of-K selection across trajectory queries.

    Samples logged state-action-future-goal triples once and compares selected
    scores with their discounted temporal-progress proxies. Candidates have
    mixed states and goals; this is not fixed-query action-value ordering.
    Writes per-triple scores and candidate-count curves under ``output_dir``.

    Args:
        configuration: Task, critics, checkpoint/output paths, triple sampling
            settings, candidate counts, repeat count, and seed from the command.

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
    n_pools = int(cfg["n_pools"])
    Ks = list(cfg["k_grid"])
    evals_root = Path(cfg["evals_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, ds = load_crl_env_and_dataset(env_name=task, discount=gamma)
    S, A, G, V = sample_trajectory_queries(ds, gamma, n, kmax, seed)
    print(
        f"[curse] {task}: {len(S)} triples, gamma^d in [{V.min():.3f}, {V.max():.3f}]", flush=True
    )

    per_triple, kcurve = [], []
    for critic in critics:
        d = evals_root / f"{critic}_{task}"
        if not (d / "params.msgpack").exists():
            raise FileNotFoundError(d / "params.msgpack")
        agent, _ = load_critic(d, critic_rebuild_function(critic))
        sc = _score(agent, critic, S, G, A)
        for i in range(len(sc)):
            per_triple.append({"critic": critic, "score": float(sc[i]), "gamma_d": float(V[i])})
        rows = _kcurve(sc, V, Ks, n_pools, seed)
        for r in rows:
            r["critic"] = critic
            kcurve.append(r)
        print(
            f"[curse]   {critic:12s} selected gamma^d: "
            + " ".join(f"K{r['K']}={r['selected_value']:.3f}" for r in rows),
            flush=True,
        )
    print(
        f"[curse]   {'oracle':12s} "
        + " ".join(
            f"K{r['K']}={r['oracle_value']:.3f}" for r in kcurve if r["critic"] == critics[0]
        ),
        flush=True,
    )

    p1 = output_dir / f"cross_query_selection_{task}.csv"
    with p1.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["critic", "score", "gamma_d"])
        w.writeheader()
        w.writerows(per_triple)
    p2 = output_dir / f"cross_query_selection_{task}_kcurve.csv"
    with p2.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "critic",
                "K",
                "selected_value",
                "selected_score",
                "oracle_value",
                "regret",
                "random_value",
            ],
        )
        w.writeheader()
        w.writerows(kcurve)
    print(f"[curse] wrote {p1} and {p2}", flush=True)
    return 0
