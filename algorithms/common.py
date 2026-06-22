from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SelectionResult:
    mask: np.ndarray
    scores: np.ndarray
    selected_indices: np.ndarray
    metadata: dict[str, object]


def validate_X_y(X_train: Any, y_train: Any) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    """Return a numeric 2D feature matrix and 1D target array."""
    if sparse.issparse(X_train):
        X = X_train.tocsr().astype(np.float64)
    else:
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X_train must be a 2D matrix.")

    y = np.asarray(y_train).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X_train must contain at least one row and one feature.")

    return X, y


def resolve_top_k(
    n_features: int,
    top_k: int | None = None,
    *,
    default_fraction: float = 0.10,
) -> int:
    """Resolve a feature cap, defaulting to 10 percent of available features."""
    if top_k is None:
        return max(1, int(ceil(n_features * default_fraction)))
    if int(top_k) != top_k or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    return min(int(top_k), n_features)


def make_mask(n_features: int, selected_indices: list[int]) -> np.ndarray:
    mask = np.zeros(n_features, dtype=bool)
    mask[np.asarray(selected_indices, dtype=int)] = True
    return mask


def top_k_mask(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the top scoring features with deterministic index tie-breaking."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(scores.size), -np.nan_to_num(scores, nan=-np.inf)))
    selected = order[:top_k]
    return make_mask(scores.size, selected.tolist()), selected


def column_means(X: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X.mean(axis=0), dtype=np.float64)


def column_variances(
    X: np.ndarray | sparse.csr_matrix,
    means: np.ndarray | None = None,
) -> np.ndarray:
    if means is None:
        means = column_means(X)
    if sparse.issparse(X):
        second_moment = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        second_moment = np.asarray(np.mean(X * X, axis=0), dtype=np.float64)
    return np.maximum(second_moment - means * means, 0.0)


def column_ranges(X: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    if sparse.issparse(X):
        mins = np.asarray(X.min(axis=0).toarray()).ravel()
        maxs = np.asarray(X.max(axis=0).toarray()).ravel()
    else:
        mins = np.asarray(np.min(X, axis=0), dtype=np.float64)
        maxs = np.asarray(np.max(X, axis=0), dtype=np.float64)
    ranges = maxs - mins
    ranges[ranges == 0.0] = 1.0
    return ranges


def target_correlations(X: np.ndarray | sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Estimate absolute feature-target association without imposing label order."""
    y = np.asarray(y).ravel()
    numeric_y = np.issubdtype(y.dtype, np.number)
    unique_values = np.unique(y)

    if numeric_y and unique_values.size > max(20, int(np.sqrt(max(y.size, 1)))):
        return absolute_correlations_with_vector(X, y.astype(np.float64))

    return correlation_ratio_by_class(X, y)


def absolute_correlations_with_vector(
    X: np.ndarray | sparse.csr_matrix,
    values: np.ndarray,
    *,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    centered = values - values.mean()
    value_std = float(np.sqrt(np.mean(centered * centered)))
    if value_std == 0.0:
        return np.zeros(X.shape[1], dtype=np.float64)

    if means is None:
        means = column_means(X)
    if stds is None:
        stds = np.sqrt(column_variances(X, means))

    if sparse.issparse(X):
        covariance = np.asarray(X.T @ centered).ravel() / X.shape[0]
    else:
        covariance = np.asarray(X.T @ centered, dtype=np.float64).ravel() / X.shape[0]

    correlations = np.zeros(X.shape[1], dtype=np.float64)
    valid = stds > 0.0
    correlations[valid] = covariance[valid] / (stds[valid] * value_std)
    return np.abs(np.nan_to_num(correlations))


def correlation_ratio_by_class(X: np.ndarray | sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    means = column_means(X)
    variances = column_variances(X, means)
    numerator = np.zeros(X.shape[1], dtype=np.float64)

    for label in np.unique(y):
        class_mask = y == label
        class_count = int(np.sum(class_mask))
        if class_count == 0:
            continue
        class_means = column_means(X[class_mask])
        centered = class_means - means
        numerator += class_count * centered * centered

    denominator = X.shape[0] * variances
    eta_squared = np.zeros(X.shape[1], dtype=np.float64)
    valid = denominator > 0.0
    eta_squared[valid] = numerator[valid] / denominator[valid]
    return np.sqrt(np.clip(np.nan_to_num(eta_squared), 0.0, 1.0))


def feature_correlations_with_column(
    X: np.ndarray | sparse.csr_matrix,
    column_index: int,
    *,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> np.ndarray:
    """Return absolute Pearson correlations between all features and one feature."""
    if means is None:
        means = column_means(X)
    if stds is None:
        stds = np.sqrt(column_variances(X, means))

    selected_std = stds[column_index]
    correlations = np.zeros(X.shape[1], dtype=np.float64)
    if selected_std == 0.0:
        return correlations

    if sparse.issparse(X):
        selected_column = X[:, column_index]
        cross_moment = np.asarray((X.T @ selected_column).toarray()).ravel() / X.shape[0]
    else:
        selected_column = X[:, column_index]
        cross_moment = np.asarray(X.T @ selected_column, dtype=np.float64).ravel() / X.shape[0]

    covariance = cross_moment - means * means[column_index]
    valid = stds > 0.0
    correlations[valid] = covariance[valid] / (stds[valid] * selected_std)
    correlations[column_index] = 0.0
    return np.abs(np.nan_to_num(correlations))
