from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .common import SelectionResult, make_mask, resolve_top_k, target_correlations, validate_X_y


def select_cfs_greedy(
    X_train: Any,
    y_train: Any,
    max_features: int,
    *,
    min_improvement: float = 1e-12,
    relevance_pool_multiplier: int = 4,
) -> SelectionResult:
    """Correlation-based Feature Selection with bounded greedy forward search.

    The first stage keeps the strongest features by class relevance. The second
    stage performs greedy CFS on that candidate pool using a precomputed
    feature-feature correlation matrix. This keeps Gisette-sized runs practical
    while preserving the CFS objective: high class relevance and low redundancy.
    """
    X, y = validate_X_y(X_train, y_train)
    n_features = X.shape[1]
    cap = resolve_top_k(n_features, max_features)
    if relevance_pool_multiplier <= 0:
        raise ValueError("relevance_pool_multiplier must be positive.")

    relevance = target_correlations(X, y)
    pool_size = min(n_features, max(cap, cap * relevance_pool_multiplier))
    candidate_pool = np.lexsort((np.arange(n_features), -relevance))[:pool_size]
    pool_relevance = relevance[candidate_pool]
    pool_correlations = _candidate_correlation_matrix(X, candidate_pool)

    selected_local: list[int] = []
    selected_original: list[int] = []
    remaining = np.ones(pool_size, dtype=bool)
    selected_relevance_sum = 0.0
    selected_pair_corr_sum = 0.0
    best_merit = 0.0
    feature_merits = np.full(n_features, np.nan, dtype=np.float64)
    history: list[dict[str, float | int]] = []

    while len(selected_local) < cap and np.any(remaining):
        candidate_locals = np.flatnonzero(remaining)
        if selected_local:
            redundancy_to_selected = pool_correlations[np.ix_(candidate_locals, selected_local)].sum(axis=1)
        else:
            redundancy_to_selected = np.zeros(candidate_locals.size, dtype=np.float64)

        candidate_merits = _cfs_merits(
            selected_count=len(selected_local) + 1,
            relevance_sums=selected_relevance_sum + pool_relevance[candidate_locals],
            pair_corr_sums=selected_pair_corr_sum + redundancy_to_selected,
        )
        best_position = int(np.argmax(candidate_merits))
        best_local = int(candidate_locals[best_position])
        best_candidate_merit = float(candidate_merits[best_position])

        if best_candidate_merit <= best_merit + min_improvement:
            break

        selected_local.append(best_local)
        selected_feature = int(candidate_pool[best_local])
        selected_original.append(selected_feature)
        remaining[best_local] = False
        selected_relevance_sum += float(pool_relevance[best_local])
        selected_pair_corr_sum += float(redundancy_to_selected[best_position])
        best_merit = best_candidate_merit
        feature_merits[selected_feature] = best_merit
        history.append(
            {
                "step": len(selected_original),
                "feature": selected_feature,
                "merit": best_merit,
                "relevance": float(relevance[selected_feature]),
                "mean_feature_correlation": float(
                    0.0
                    if len(selected_local) == 1
                    else (2.0 * selected_pair_corr_sum)
                    / (len(selected_local) * (len(selected_local) - 1))
                ),
            }
        )

    if not selected_original:
        fallback_feature = int(candidate_pool[0])
        selected_original.append(fallback_feature)
        feature_merits[fallback_feature] = float(relevance[fallback_feature])
        best_merit = float(relevance[fallback_feature])

    selected_indices = np.asarray(selected_original, dtype=int)
    metadata = {
        "method": "CFS-Greedy",
        "selected_indices": selected_indices,
        "class_correlations": relevance,
        "feature_merits": feature_merits,
        "best_merit": float(best_merit),
        "history": history,
        "top_k": cap,
        "candidate_pool_size": int(pool_size),
        "relevance_pool_multiplier": int(relevance_pool_multiplier),
        "stopped_reason": "top_k_reached" if len(selected_original) >= cap else "no_merit_improvement",
        "min_improvement": float(min_improvement),
    }
    return SelectionResult(
        mask=make_mask(n_features, selected_original),
        scores=feature_merits,
        selected_indices=selected_indices,
        metadata=metadata,
    )


def _candidate_correlation_matrix(X: Any, candidate_pool: np.ndarray) -> np.ndarray:
    if sparse.issparse(X):
        candidate_data = X[:, candidate_pool].toarray().astype(np.float64, copy=False)
    else:
        candidate_data = np.asarray(X[:, candidate_pool], dtype=np.float64)

    candidate_data = candidate_data - candidate_data.mean(axis=0)
    stds = candidate_data.std(axis=0)
    valid = stds > 0.0
    standardized = np.zeros_like(candidate_data, dtype=np.float64)
    standardized[:, valid] = candidate_data[:, valid] / stds[valid]
    correlations = np.abs((standardized.T @ standardized) / max(standardized.shape[0], 1))
    np.fill_diagonal(correlations, 0.0)
    return np.nan_to_num(correlations)


def _cfs_merits(
    selected_count: int,
    relevance_sums: np.ndarray,
    pair_corr_sums: np.ndarray,
) -> np.ndarray:
    if selected_count <= 0:
        return np.zeros_like(relevance_sums, dtype=np.float64)
    mean_relevance = relevance_sums / selected_count
    if selected_count == 1:
        return mean_relevance
    mean_feature_corr = (2.0 * pair_corr_sums) / (selected_count * (selected_count - 1))
    denominator = np.sqrt(selected_count + selected_count * (selected_count - 1) * mean_feature_corr)
    merits = np.zeros_like(relevance_sums, dtype=np.float64)
    valid = denominator > 0.0
    merits[valid] = (selected_count * mean_relevance[valid]) / denominator[valid]
    return np.nan_to_num(merits)


def cfs_greedy(
    X_train: Any,
    y_train: Any,
    max_features: int,
    *,
    min_improvement: float = 1e-12,
    relevance_pool_multiplier: int = 4,
) -> SelectionResult:
    return select_cfs_greedy(
        X_train,
        y_train,
        max_features,
        min_improvement=min_improvement,
        relevance_pool_multiplier=relevance_pool_multiplier,
    )


cfs_greedy_filter = select_cfs_greedy
CFS_Greedy = select_cfs_greedy
