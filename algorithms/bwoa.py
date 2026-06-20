from __future__ import annotations

import math
from typing import Callable

import numpy as np

from utils.helpers import ensure_non_empty, sigmoid


FitnessFunction = Callable[[np.ndarray], float]


def binary_whale_optimization_algorithm(
    n_features: int,
    fitness_fn: FitnessFunction,
    rng: np.random.Generator,
    population_size: int,
    iterations: int,
) -> tuple[np.ndarray, float]:
    """Binary Whale Optimization Algorithm for feature subset selection."""
    positions = rng.uniform(0, 1, size=(population_size, n_features))
    binary_positions = (positions > 0.5).astype(int)
    for i in range(population_size):
        binary_positions[i] = ensure_non_empty(binary_positions[i], rng)

    scores = np.array([fitness_fn(mask) for mask in binary_positions])
    best_index = int(np.argmin(scores))
    best_position = positions[best_index].copy()
    best_mask = binary_positions[best_index].copy()
    best_score = float(scores[best_index])

    for iteration in range(iterations):
        a = 2 - 2 * (iteration / max(iterations - 1, 1))
        for i in range(population_size):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)
            a_vector = 2 * a * r1 - a
            c_vector = 2 * r2
            p = rng.random()

            if p < 0.5:
                if np.linalg.norm(a_vector) < 1:
                    distance = np.abs(c_vector * best_position - positions[i])
                    positions[i] = best_position - a_vector * distance
                else:
                    random_index = rng.integers(0, population_size)
                    random_position = positions[random_index]
                    distance = np.abs(c_vector * random_position - positions[i])
                    positions[i] = random_position - a_vector * distance
            else:
                b = 1.0
                l_value = rng.uniform(-1, 1, size=n_features)
                distance = np.abs(best_position - positions[i])
                positions[i] = distance * np.exp(b * l_value) * np.cos(2 * math.pi * l_value) + best_position

        probabilities = sigmoid(positions)
        binary_positions = (rng.random((population_size, n_features)) < probabilities).astype(int)
        for i in range(population_size):
            binary_positions[i] = ensure_non_empty(binary_positions[i], rng)

        scores = np.array([fitness_fn(mask) for mask in binary_positions])
        current_best_index = int(np.argmin(scores))
        current_best_score = float(scores[current_best_index])
        if current_best_score < best_score:
            best_score = current_best_score
            best_position = positions[current_best_index].copy()
            best_mask = binary_positions[current_best_index].copy()

    return best_mask, best_score
