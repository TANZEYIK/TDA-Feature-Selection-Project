from __future__ import annotations

from typing import Callable

import numpy as np

from utils.helpers import ensure_non_empty


FitnessFunction = Callable[[np.ndarray], float]


def binary_genetic_algorithm(
    n_features: int,
    fitness_fn: FitnessFunction,
    rng: np.random.Generator,
    population_size: int,
    iterations: int,
    crossover_rate: float = 0.80,
    mutation_rate: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Binary Genetic Algorithm for feature subset selection."""
    population = rng.integers(0, 2, size=(population_size, n_features))
    for i in range(population_size):
        population[i] = ensure_non_empty(population[i], rng)

    scores = np.array([fitness_fn(individual) for individual in population])
    best_index = int(np.argmin(scores))
    best_mask = population[best_index].copy()
    best_score = float(scores[best_index])

    def tournament_select() -> np.ndarray:
        candidates = rng.choice(population_size, size=3, replace=False)
        winner = candidates[int(np.argmin(scores[candidates]))]
        return population[winner].copy()

    for _ in range(iterations):
        new_population = []
        while len(new_population) < population_size:
            parent_1 = tournament_select()
            parent_2 = tournament_select()

            if rng.random() < crossover_rate and n_features > 1:
                point = rng.integers(1, n_features)
                child_1 = np.concatenate([parent_1[:point], parent_2[point:]])
                child_2 = np.concatenate([parent_2[:point], parent_1[point:]])
            else:
                child_1, child_2 = parent_1.copy(), parent_2.copy()

            for child in (child_1, child_2):
                mutation_mask = rng.random(n_features) < mutation_rate
                child[mutation_mask] = 1 - child[mutation_mask]
                child = ensure_non_empty(child, rng)
                new_population.append(child)
                if len(new_population) == population_size:
                    break

        population = np.array(new_population)
        scores = np.array([fitness_fn(individual) for individual in population])
        current_best_index = int(np.argmin(scores))
        current_best_score = float(scores[current_best_index])
        if current_best_score < best_score:
            best_score = current_best_score
            best_mask = population[current_best_index].copy()

    return best_mask, best_score
