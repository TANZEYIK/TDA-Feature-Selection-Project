from __future__ import annotations

from typing import Callable

import numpy as np

from utils.helpers import ensure_non_empty, sigmoid


FitnessFunction = Callable[[np.ndarray], float]


def binary_grey_wolf_optimizer(
    n_features: int,
    fitness_fn: FitnessFunction,
    rng: np.random.Generator,
    population_size: int,
    iterations: int,
    progress_callback: Callable[[int, float], None] | None = None,
    early_stopping_patience: int = 5,
    min_delta: float = 0.001,
) -> tuple[np.ndarray, float, list[float], bool]:
    """Binary Grey Wolf Optimizer for feature subset selection."""
    positions = rng.uniform(0, 1, size=(population_size, n_features))
    binary_positions = (positions > 0.5).astype(int)
    for i in range(population_size):
        binary_positions[i] = ensure_non_empty(binary_positions[i], rng)

    def rank_wolves():
        scores = np.array([fitness_fn(mask) for mask in binary_positions])
        order = np.argsort(scores)
        return order, scores

    order, scores = rank_wolves()
    alpha_pos = positions[order[0]].copy()
    beta_pos = positions[order[1]].copy()
    delta_pos = positions[order[2]].copy()
    best_mask = binary_positions[order[0]].copy()
    best_score = float(scores[order[0]])
    best_fitness_history = [best_score]
    no_improvement_count = 0
    stopped_early = False

    for iteration in range(iterations):
        a = 2 - 2 * (iteration / max(iterations - 1, 1))

        for i in range(population_size):
            updated_positions = []
            for leader_pos in (alpha_pos, beta_pos, delta_pos):
                r1 = rng.random(n_features)
                r2 = rng.random(n_features)
                a_vector = 2 * a * r1 - a
                c_vector = 2 * r2
                distance = np.abs(c_vector * leader_pos - positions[i])
                updated_positions.append(leader_pos - a_vector * distance)
            positions[i] = np.mean(updated_positions, axis=0)

        probabilities = sigmoid(positions)
        binary_positions = (rng.random((population_size, n_features)) < probabilities).astype(int)
        for i in range(population_size):
            binary_positions[i] = ensure_non_empty(binary_positions[i], rng)

        order, scores = rank_wolves()
        alpha_pos = positions[order[0]].copy()
        beta_pos = positions[order[1]].copy()
        delta_pos = positions[order[2]].copy()

        current_best_score = float(scores[order[0]])
        if best_score - current_best_score >= min_delta:
            best_score = current_best_score
            best_mask = binary_positions[order[0]].copy()
            no_improvement_count = 0
        elif current_best_score < best_score:
            best_score = current_best_score
            best_mask = binary_positions[order[0]].copy()
            no_improvement_count += 1
        else:
            no_improvement_count += 1
        best_fitness_history.append(best_score)
        if progress_callback is not None:
            progress_callback(iteration + 1, best_score)
        if no_improvement_count >= early_stopping_patience:
            stopped_early = True
            break

    return best_mask, best_score, best_fitness_history, stopped_early
