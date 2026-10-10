"""Export frozen PointMaze ranks and representative-pool data for rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CRITICS = ("td_q", "raw_bilinear", "cosine", "hybrid")


def _load(
    data_dir: Path,
    expected_queries: int,
    reference_labels: list[str],
    critic_labels: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    pools = np.load(data_dir / "pools.npz")
    actions = pools["A"]
    references = np.stack(
        [np.load(data_dir / f"scores_{label}.npz")["scores"] for label in reference_labels]
    )
    ref_mean = references.mean(axis=0)
    ref_se = references.std(axis=0, ddof=1) / np.sqrt(references.shape[0])
    spread = np.percentile(ref_mean, 90, axis=-1) - np.percentile(ref_mean, 10, axis=-1)
    pool_resolvable = spread > 2 * np.median(ref_se, axis=-1)
    query_resolvable = pool_resolvable.sum(axis=1) >= 3
    if int(query_resolvable.sum()) != expected_queries:
        raise RuntimeError(
            "Expected the frozen PointMaze estimand to match the declared resolvable query count"
        )
    scores = {
        critic: np.load(data_dir / f"scores_{critic_labels[critic]}.npz")["scores"]
        for critic in CRITICS
    }
    return actions, ref_mean, query_resolvable, scores


def _selected_percentiles(
    ref_mean: np.ndarray, query_mask: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for critic in CRITICS:
        chosen = scores[critic].argmax(axis=-1)
        values: list[float] = []
        for query in np.flatnonzero(query_mask):
            for replicate in range(ref_mean.shape[1]):
                pool = ref_mean[query, replicate]
                selected = pool[chosen[query, replicate]]
                values.append(100 * float(np.mean(pool <= selected)))
        output[critic] = np.asarray(values)
    return output


def _representative_pool(
    ref_mean: np.ndarray,
    query_mask: np.ndarray,
    scores: dict[str, np.ndarray],
    expected_pool: tuple[int, int],
) -> tuple[int, int]:
    regrets = []
    for critic in CRITICS:
        chosen = scores[critic].argmax(axis=-1)
        selected = np.take_along_axis(ref_mean, chosen[..., None], axis=-1)[..., 0]
        best = ref_mean.max(axis=-1)
        denominator = best - ref_mean.mean(axis=-1) + 1e-9
        regrets.append((best - selected) / denominator)
    matrix = np.stack(regrets, axis=-1)
    eligible = matrix[query_mask].reshape(-1, len(CRITICS))
    median = np.median(eligible, axis=0)
    scale = np.percentile(eligible, 75, axis=0) - np.percentile(eligible, 25, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    distance = np.square((matrix - median) / scale).sum(axis=-1)
    distance[~query_mask] = np.inf
    query, replicate = np.unravel_index(np.argmin(distance), distance.shape)
    if (int(query), int(replicate)) != expected_pool:
        raise RuntimeError("Frozen representative-pool rule changed unexpectedly")
    return int(query), int(replicate)


def export_reference_figure_data(
    data_dir: Path,
    output: Path,
    expected_queries: int,
    expected_pool: tuple[int, int],
    reference_labels: list[str],
    critic_labels: dict[str, str],
) -> None:
    """Write derived ranks once, with the original frozen population assertions."""
    if set(critic_labels) != set(CRITICS) or len(reference_labels) != 3:
        raise ValueError("The frozen plot requires four critic readouts and three references")
    actions, reference, mask, scores = _load(
        data_dir, expected_queries, reference_labels, critic_labels
    )
    percentiles = _selected_percentiles(reference, mask, scores)
    query, replicate = _representative_pool(reference, mask, scores, expected_pool)
    pool_values = reference[query, replicate]
    payload = {
        "pool_actions": actions[query, replicate],
        "pool_percentile": 100 * np.mean(pool_values[:, None] <= pool_values[None, :], axis=0),
        "reference_winner": np.asarray(pool_values.argmax()),
        "query_id": np.asarray(query),
        "pool_replicate": np.asarray(replicate),
        "n_resolvable": np.asarray(mask.sum()),
    }
    for critic in CRITICS:
        payload[f"percentiles_{critic}"] = percentiles[critic]
        payload[f"winner_{critic}"] = np.asarray(scores[critic][query, replicate].argmax())
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
