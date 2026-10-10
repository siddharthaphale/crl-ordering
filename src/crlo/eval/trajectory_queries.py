"""Sample trajectory-linked queries and score their progress proxies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import numpy as np

from crlo.eval.critic_scoring import critic_score


def score_triples(
    agent: Any, critic: str, S: np.ndarray, G: np.ndarray, A: np.ndarray, batch: int = 512
) -> np.ndarray:
    """Score aligned state-action-goal triples in bounded inference batches.

    Args:
        agent: Constructed critic agent accepted by ``critic_score``.
        critic: Deployed critic readout name accepted by ``critic_score``.
        S: States, shape ``(n, observation_dim)``, with ``n > 0``.
        G: Goals, shape ``(n, goal_dim)``, in the same order as states.
        A: Actions, shape ``(n, action_dim)``, in the same order as states.
        batch: Positive maximum number of triples per inference call.

    Returns:
        NumPy scores of shape ``(n,)`` in input order, averaging critic
        ensembles according to ``critic_score``. No sampling or sorting occurs.
    """
    out = []
    for i in range(0, len(S), batch):
        out.append(
            critic_score(agent, critic, S[i : i + batch], G[i : i + batch], A[i : i + batch])
        )
    return np.concatenate(out).reshape(-1)


class TrajectoryDataset(Protocol):
    """Dataset arrays and inclusive episode ends required by the query sampler."""

    @property
    def dataset(self) -> Mapping[str, np.ndarray]:
        """Observation and action arrays indexed by flat transition position."""
        ...

    @property
    def terminal_locs(self) -> np.ndarray:
        """Flat indices of inclusive episode ends."""
        ...


class TrajectoryQueries(NamedTuple):
    """Aligned sampled triples, progress proxies, and their dataset identities.

    Attributes:
        states: Float32 observations, shape ``(n, observation_dim)``.
        actions: Float32 logged actions, shape ``(n, action_dim)``.
        goals: Float32 future observations, shape ``(n, observation_dim)``.
        values: Float64 discounted temporal progress ``gamma**offsets``, shape ``(n,)``.
        indices: Flat anchor indices, shape ``(n,)``.
        offsets: Positive temporal distances to goals, shape ``(n,)``.
        episode_ends: Inclusive terminal indices for anchors, shape ``(n,)``.
        episode_ids: Zero-based ordinals in the sorted terminal array, shape ``(n,)``.
    """

    states: np.ndarray
    actions: np.ndarray
    goals: np.ndarray
    values: np.ndarray
    indices: np.ndarray
    offsets: np.ndarray
    episode_ends: np.ndarray
    episode_ids: np.ndarray


def sample_trajectory_queries(
    ds: TrajectoryDataset, gamma: float, n: int, kmax: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample same-episode future goals and discounted temporal progress.

    Args:
        ds: Dataset containing aligned observations, actions, and episode ends.
            At least one anchor must have two remaining transitions.
        gamma: Discount factor used in the progress proxy ``gamma**k``.
        n: Positive number of triples to sample, with replacement.
        kmax: Positive maximum temporal offset, capped by each episode end.
        seed: Seed for a local NumPy generator; global RNG state is unchanged.

    Returns:
        States, actions, goals, and progress values in sampled order. Shapes
        are ``(n, observation_dim)``, ``(n, action_dim)``,
        ``(n, observation_dim)``, and ``(n,)``. The first three arrays are
        float32; values are float64. Values are temporal proxies, not measured
        candidate-action returns. Sampling follows
        ``sample_trajectory_queries_with_metadata`` exactly.
    """
    queries = sample_trajectory_queries_with_metadata(ds, gamma, n, kmax, seed)
    return queries.states, queries.actions, queries.goals, queries.values


def sample_trajectory_queries_with_metadata(
    ds: TrajectoryDataset, gamma: float, n: int, kmax: int, seed: int
) -> TrajectoryQueries:
    """Sample trajectory queries with the frozen retrieval RNG sequence.

    Args:
        ds: Aligned observations/actions and inclusive episode ends. At least
            one anchor must have two remaining transitions.
        gamma: Discount factor used in ``gamma**k``.
        n: Positive number of triples to sample, with replacement.
        kmax: Positive maximum temporal distance to a goal.
        seed: Seed for a local NumPy generator; global RNG state is unchanged.

    Returns:
        Named arrays of triples and metadata, all in the same sampled order.
        Anchors are drawn from ``[0, len(observations) - 2)`` and rejected when
        fewer than two transitions remain. Each accepted anchor draws one
        offset uniformly from ``1`` through ``min(kmax, episode_end - index)``.
        Repeated queries are retained; no draw is consumed for metadata.
    """
    raw = ds.dataset
    obs = np.asarray(raw["observations"])
    act = np.asarray(raw["actions"])
    term = np.sort(np.asarray(ds.terminal_locs))
    rng = np.random.default_rng(seed)
    S, A, G, V, T, K, END, EP = [], [], [], [], [], [], [], []
    while len(S) < n:
        t = int(rng.integers(0, len(obs) - 2))
        idx = np.searchsorted(term, t)
        if idx >= len(term):
            continue
        end = int(term[idx])
        if end - t < 2:
            continue
        k = int(rng.integers(1, min(kmax, end - t) + 1))
        S.append(obs[t])
        A.append(act[t])
        G.append(obs[t + k])
        V.append(gamma**k)
        T.append(t)
        K.append(k)
        END.append(end)
        EP.append(idx)
    return TrajectoryQueries(
        np.asarray(S, np.float32),
        np.asarray(A, np.float32),
        np.asarray(G, np.float32),
        np.asarray(V),
        np.asarray(T),
        np.asarray(K),
        np.asarray(END),
        np.asarray(EP),
    )
