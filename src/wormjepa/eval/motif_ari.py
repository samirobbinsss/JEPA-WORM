"""Hungarian-matched ARI vs Flavell labels (Story 6.3 / FR26 / row 2).

The PRD's motif-recovery metric: cluster the JEPA latent with k-means, match
cluster IDs to Flavell behavioral-state labels via the Hungarian assignment,
then compute Adjusted Rand Index on held-out worms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True, slots=True)
class MotifARIResult:
    """Aggregate ARI + per-worm details."""

    ari: float
    """Mean Hungarian-matched ARI across held-out worms."""

    per_worm_ari: list[float]
    """Per-worm ARI on the corresponding held-out fold."""

    worm_ids: list[str]


def _hungarian_assign_labels(predicted: np.ndarray, truth: np.ndarray, n_states: int) -> np.ndarray:
    """Map cluster ids to label ids that maximize agreement, then return remapped ``predicted``."""
    n_clusters = int(np.max(predicted)) + 1 if predicted.size else 0
    size = max(n_clusters, n_states)
    cost = np.zeros((size, size), dtype=np.int64)
    for c in range(n_clusters):
        for s in range(n_states):
            cost[c, s] = int(np.sum((predicted == c) & (truth == s)))
    # linear_sum_assignment minimizes; we want to maximize agreement, so negate.
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind, strict=True)}
    return np.asarray([mapping.get(int(p), int(p)) for p in predicted])


def motif_ari(
    latent: np.ndarray,
    labels: np.ndarray,
    worm_ids: Sequence[str],
    *,
    n_states: int = 5,
    n_clusters: int | None = None,
    kmeans_random_state: int = 42,
) -> MotifARIResult:
    """Leave-one-worm-out Hungarian-matched ARI.

    Args:
        latent: ``(N, D)`` per-frame latents.
        labels: ``(N,)`` integer Flavell-style behavioral-state labels.
        worm_ids: ``(N,)`` worm-id strings.
        n_states: Number of distinct behavioral states (e.g., 5).
        n_clusters: K for k-means. Defaults to ``n_states``.
        kmeans_random_state: Seed for k-means initialization.

    Returns:
        :class:`MotifARIResult` with per-worm and aggregate scores.
    """
    n_clusters = n_clusters or n_states
    unique_worms = list(dict.fromkeys(worm_ids))
    if len(unique_worms) < 2:
        msg = "motif_ari requires >=2 distinct worms"
        raise ValueError(msg)
    worm_array = np.asarray(worm_ids)

    per_worm: list[float] = []
    for held_out in unique_worms:
        train_mask = worm_array != held_out
        test_mask = worm_array == held_out
        if not train_mask.any() or not test_mask.any():
            continue
        # Fit k-means on the training subset, transfer to test.
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=kmeans_random_state,
        )
        kmeans.fit(latent[train_mask])
        test_clusters = np.asarray(kmeans.predict(latent[test_mask]))
        test_truth = labels[test_mask].astype(np.int64)
        remapped = _hungarian_assign_labels(test_clusters, test_truth, n_states)
        per_worm.append(float(adjusted_rand_score(test_truth, remapped)))

    mean_ari = float(np.mean(per_worm)) if per_worm else 0.0
    return MotifARIResult(ari=mean_ari, per_worm_ari=per_worm, worm_ids=unique_worms)
