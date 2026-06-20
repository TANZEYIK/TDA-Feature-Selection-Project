from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from utils.helpers import ensure_non_empty


FitnessFunction = Callable[[np.ndarray], float]


def make_fitness_function(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.90,
    beta: float = 0.10,
) -> tuple[FitnessFunction, Callable[[np.ndarray], float]]:
    """
    Create the fitness function for feature selection.

    Lower fitness is better:
    fitness = alpha * classification_error + beta * selected_feature_ratio
    """
    total_features = x_train.shape[1]

    def evaluate_accuracy(mask: np.ndarray) -> float:
        mask = ensure_non_empty(mask, rng).astype(bool)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train[:, mask], y_train)
        prediction = model.predict(x_test[:, mask])
        return accuracy_score(y_test, prediction)

    def fitness(mask: np.ndarray) -> float:
        mask = ensure_non_empty(mask, rng)
        accuracy = evaluate_accuracy(mask)
        error = 1.0 - accuracy
        selected_ratio = mask.sum() / total_features
        return alpha * error + beta * selected_ratio

    return fitness, evaluate_accuracy
