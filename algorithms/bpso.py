from __future__ import annotations

from typing import Callable

import numpy as np

from utils.helpers import ensure_non_empty, sigmoid


FitnessFunction = Callable[[np.ndarray], float]


def binary_particle_swarm_optimization(
    n_features: int,
    fitness_fn: FitnessFunction,
    rng: np.random.Generator,
    population_size: int,
    iterations: int,
    progress_callback: Callable[[int, float], None] | None = None,
    early_stopping_patience: int = 5,
    min_delta: float = 0.001,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
) -> tuple[np.ndarray, float, list[float], bool]:
    """Binary Particle Swarm Optimization for feature subset selection."""
    positions = rng.integers(0, 2, size=(population_size, n_features))
    velocities = rng.uniform(-1, 1, size=(population_size, n_features))
    for i in range(population_size):
        positions[i] = ensure_non_empty(positions[i], rng)

    personal_best = positions.copy()
    personal_scores = np.array([fitness_fn(p) for p in personal_best])
    global_index = int(np.argmin(personal_scores))
    global_best = personal_best[global_index].copy()
    global_score = float(personal_scores[global_index])
    best_fitness_history = [global_score]
    no_improvement_count = 0
    stopped_early = False

    for iteration in range(1, iterations + 1):
        r1 = rng.random((population_size, n_features))
        r2 = rng.random((population_size, n_features))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best - positions)
            + social * r2 * (global_best - positions)
        )
        probabilities = sigmoid(velocities)
        positions = (rng.random((population_size, n_features)) < probabilities).astype(int)

        for i in range(population_size):
            positions[i] = ensure_non_empty(positions[i], rng)
            score = fitness_fn(positions[i])
            if score < personal_scores[i]:
                personal_scores[i] = score
                personal_best[i] = positions[i].copy()

        current_global_index = int(np.argmin(personal_scores))
        current_global_score = float(personal_scores[current_global_index])
        if global_score - current_global_score >= min_delta:
            global_score = current_global_score
            global_best = personal_best[current_global_index].copy()
            no_improvement_count = 0
        elif current_global_score < global_score:
            global_score = current_global_score
            global_best = personal_best[current_global_index].copy()
            no_improvement_count += 1
        else:
            no_improvement_count += 1
        best_fitness_history.append(global_score)
        if progress_callback is not None:
            progress_callback(iteration, global_score)
        if no_improvement_count >= early_stopping_patience:
            stopped_early = True
            break

    return global_best, global_score, best_fitness_history, stopped_early
