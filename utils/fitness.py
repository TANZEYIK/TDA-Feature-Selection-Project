from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from utils.helpers import ensure_non_empty


FitnessFunction = Callable[[np.ndarray], float]


def make_accuracy_function(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    rng: np.random.Generator,
) -> Callable[[np.ndarray], float]:
    """Create an accuracy evaluator for a selected feature subset."""

    def evaluate_accuracy(mask: np.ndarray) -> float:
        mask = ensure_non_empty(mask, rng).astype(bool)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train[:, mask], y_train)
        prediction = model.predict(x_evaluation[:, mask])
        return accuracy_score(y_evaluation, prediction)

    return evaluate_accuracy


def make_fitness_function(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.90,
    beta: float = 0.10,
) -> tuple[FitnessFunction, Callable[[np.ndarray], float]]:
    """
    Create the fitness function for feature selection.

    The validation set is used during feature subset search. The test set should
    only be used after the best feature subset has been selected.

    Lower fitness is better:
    fitness = alpha * classification_error + beta * selected_feature_ratio
    """
    total_features = x_train.shape[1]
    evaluate_validation_accuracy = make_accuracy_function(
        x_train,
        y_train,
        x_validation,
        y_validation,
        rng,
    )

    def fitness(mask: np.ndarray) -> float:
        mask = ensure_non_empty(mask, rng)
        accuracy = evaluate_validation_accuracy(mask)
        error = 1.0 - accuracy
        selected_ratio = mask.sum() / total_features
        return alpha * error + beta * selected_ratio

    return fitness, evaluate_validation_accuracy
