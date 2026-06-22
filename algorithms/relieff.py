from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

from .common import SelectionResult, column_ranges, resolve_top_k, top_k_mask, validate_X_y


def select_relieff(
    X_train: Any,
    y_train: Any,
    top_k: int,
    random_state: int | None = 42,
    *,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> SelectionResult:
    """ReliefF filter using every provided training row as an anchor instance."""
    X, y = validate_X_y(X_train, y_train)
    n_samples, n_features = X.shape
    k = resolve_top_k(n_features, top_k)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer.")

    ranges = column_ranges(X)
    inv_ranges = 1.0 / ranges
    if sparse.issparse(X):
        X_scaled = X.multiply(inv_ranges).tocsr()
    else:
        X_scaled = X * inv_ranges

    distances = pairwise_distances(X_scaled, metric=metric)
    np.fill_diagonal(distances, np.inf)

    labels, counts = np.unique(y, return_counts=True)
    class_priors = {label: count / n_samples for label, count in zip(labels, counts)}
    weights = np.zeros(n_features, dtype=np.float64)
    used_hit_counts = np.zeros(n_samples, dtype=int)
    used_miss_counts = np.zeros(n_samples, dtype=int)

    for row_index in range(n_samples):
        row_label = y[row_index]
        hit_indices = _nearest_indices(
            distances[row_index],
            np.flatnonzero(y == row_label),
            n_neighbors,
            exclude=row_index,
        )
        if hit_indices.size:
            weights -= _mean_feature_difference(X, row_index, hit_indices, inv_ranges) / n_samples
            used_hit_counts[row_index] = hit_indices.size

        miss_denominator = max(1.0 - class_priors[row_label], np.finfo(float).eps)
        for label in labels:
            if label == row_label:
                continue
            miss_indices = _nearest_indices(
                distances[row_index],
                np.flatnonzero(y == label),
                n_neighbors,
            )
            if not miss_indices.size:
                continue
            class_weight = class_priors[label] / miss_denominator
            weights += (
                class_weight
                * _mean_feature_difference(X, row_index, miss_indices, inv_ranges)
                / n_samples
            )
            used_miss_counts[row_index] += miss_indices.size

    mask, selected = top_k_mask(weights, k)
    metadata = {
        "method": "ReliefF",
        "selected_indices": selected,
        "feature_weights": weights,
        "top_k": k,
        "n_neighbors": int(n_neighbors),
        "metric": metric,
        "random_state": random_state,
        "rows_used": n_samples,
        "mean_hits_used": float(np.mean(used_hit_counts)),
        "mean_misses_used": float(np.mean(used_miss_counts)),
    }
    return SelectionResult(
        mask=mask,
        scores=weights,
        selected_indices=selected,
        metadata=metadata,
    )


def _nearest_indices(
    distances: np.ndarray,
    candidates: np.ndarray,
    n_neighbors: int,
    *,
    exclude: int | None = None,
) -> np.ndarray:
    if exclude is not None:
        candidates = candidates[candidates != exclude]
    if candidates.size == 0:
        return np.asarray([], dtype=int)
    order = np.lexsort((candidates, distances[candidates]))
    return candidates[order[: min(n_neighbors, candidates.size)]]


def _mean_feature_difference(
    X: np.ndarray | sparse.csr_matrix,
    row_index: int,
    neighbor_indices: np.ndarray,
    inv_ranges: np.ndarray,
) -> np.ndarray:
    if neighbor_indices.size == 0:
        return np.zeros(X.shape[1], dtype=np.float64)

    if sparse.issparse(X):
        anchor = X[row_index]
        diffs = np.zeros(X.shape[1], dtype=np.float64)
        for neighbor_index in neighbor_indices:
            diffs += np.abs((anchor - X[neighbor_index]).toarray()).ravel() * inv_ranges
        return diffs / neighbor_indices.size

    anchor = X[row_index]
    return np.mean(np.abs(X[neighbor_indices] - anchor) * inv_ranges, axis=0)


def relieff(
    X_train: Any,
    y_train: Any,
    top_k: int,
    random_state: int | None = 42,
    *,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> SelectionResult:
    return select_relieff(
        X_train,
        y_train,
        top_k,
        random_state=random_state,
        n_neighbors=n_neighbors,
        metric=metric,
    )


relieff_filter = select_relieff
ReliefF = select_relieff
