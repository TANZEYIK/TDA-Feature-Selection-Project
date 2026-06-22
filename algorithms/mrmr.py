from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    SelectionResult,
    column_means,
    column_variances,
    feature_correlations_with_column,
    make_mask,
    resolve_top_k,
    target_correlations,
    validate_X_y,
)


def select_mrmr(
    X_train: Any,
    y_train: Any,
    top_k: int,
    *,
    redundancy_weight: float = 1.0,
) -> SelectionResult:
    """Minimum Redundancy Maximum Relevance filter selection.

    Scores are computed greedily as feature-target relevance minus the average
    absolute correlation to already selected features.
    """
    X, y = validate_X_y(X_train, y_train)
    n_features = X.shape[1]
    k = resolve_top_k(n_features, top_k)

    relevance = target_correlations(X, y)
    means = column_means(X)
    stds = np.sqrt(column_variances(X, means))
    selected: list[int] = []
    remaining: set[int] = set(range(n_features))
    selection_scores = np.full(n_features, np.nan, dtype=np.float64)
    redundancy_sums = np.zeros(n_features, dtype=np.float64)
    corr_cache: dict[int, np.ndarray] = {}
    history: list[dict[str, float | int]] = []

    for step in range(k):
        if selected:
            denominator = float(len(selected))
            candidate_scores = {
                feature: float(
                    relevance[feature]
                    - redundancy_weight * (redundancy_sums[feature] / denominator)
                )
                for feature in remaining
            }
        else:
            candidate_scores = {feature: float(relevance[feature]) for feature in remaining}

        best_feature = min(
            candidate_scores,
            key=lambda feature: (-candidate_scores[feature], feature),
        )
        best_score = candidate_scores[best_feature]

        selected.append(best_feature)
        remaining.remove(best_feature)
        selection_scores[best_feature] = best_score
        history.append(
            {
                "step": step + 1,
                "feature": best_feature,
                "score": best_score,
                "relevance": float(relevance[best_feature]),
                "redundancy": float(
                    0.0 if step == 0 else redundancy_sums[best_feature] / len(selected[:-1])
                ),
            }
        )

        if not remaining:
            break

        if best_feature not in corr_cache:
            corr_cache[best_feature] = feature_correlations_with_column(
                X,
                best_feature,
                means=means,
                stds=stds,
            )
        redundancy_sums += corr_cache[best_feature]

    metadata = {
        "method": "mRMR",
        "selected_indices": np.asarray(selected, dtype=int),
        "relevance_scores": relevance,
        "selection_scores": selection_scores,
        "history": history,
        "top_k": k,
        "redundancy_weight": float(redundancy_weight),
    }
    selected_indices = np.asarray(selected, dtype=int)
    return SelectionResult(
        mask=make_mask(n_features, selected),
        scores=selection_scores,
        selected_indices=selected_indices,
        metadata=metadata,
    )


def mrmr(
    X_train: Any,
    y_train: Any,
    top_k: int,
    *,
    redundancy_weight: float = 1.0,
) -> SelectionResult:
    return select_mrmr(
        X_train,
        y_train,
        top_k,
        redundancy_weight=redundancy_weight,
    )


mrmr_filter = select_mrmr
mRMR = select_mrmr
