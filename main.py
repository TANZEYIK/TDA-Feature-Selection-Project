"""
TDA6323 Part 2: Metaheuristic Algorithms for Feature Selection

This is the main entry point for the experiment. The implementation is separated
into algorithm modules and utility modules to make the project easier to read.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from algorithms.bga import binary_genetic_algorithm
from algorithms.bgwo import binary_grey_wolf_optimizer
from algorithms.bpso import binary_particle_swarm_optimization
from algorithms.bwoa import binary_whale_optimization_algorithm
from utils.dataset import prepare_dataset
from utils.fitness import make_fitness_function
from utils.plotting import plot_accuracy, plot_execution_time, plot_selected_feature_count


@dataclass
class RunResult:
    algorithm: str
    input_features: int
    runtime_seconds: float
    best_fitness: float
    accuracy: float
    selected_count: int
    selected_features: str


def run_single_algorithm(
    algorithm_name: str,
    algorithm_fn: Callable[..., tuple[np.ndarray, float]],
    input_features: int,
    population_size: int,
    iterations: int,
    dataset_random_state: int,
    algorithm_random_state: int,
) -> RunResult:
    rng = np.random.default_rng(algorithm_random_state)
    x_train, x_test, y_train, y_test, feature_names = prepare_dataset(
        input_features,
        dataset_random_state,
    )
    fitness_fn, accuracy_fn = make_fitness_function(x_train, x_test, y_train, y_test, rng)

    start = time.perf_counter()
    best_mask, best_fitness = algorithm_fn(
        input_features,
        fitness_fn,
        rng,
        population_size,
        iterations,
    )
    runtime = time.perf_counter() - start

    accuracy = accuracy_fn(best_mask)
    selected_names = feature_names[best_mask.astype(bool)]
    return RunResult(
        algorithm=algorithm_name,
        input_features=input_features,
        runtime_seconds=runtime,
        best_fitness=best_fitness,
        accuracy=accuracy,
        selected_count=int(best_mask.sum()),
        selected_features=", ".join(selected_names),
    )


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    algorithms = {
        "BGA": binary_genetic_algorithm,
        "BPSO": binary_particle_swarm_optimization,
        "BGWO": binary_grey_wolf_optimizer,
        "BWOA": binary_whale_optimization_algorithm,
    }

    all_results: list[RunResult] = []
    for input_features in args.input_sizes:
        dataset_seed = args.random_state + input_features * 100
        for algorithm_index, (name, function) in enumerate(algorithms.items()):
            algorithm_seed = args.random_state + input_features * 1000 + algorithm_index
            print(f"Running {name} with {input_features} features...")
            result = run_single_algorithm(
                name,
                function,
                input_features,
                args.population_size,
                args.iterations,
                dataset_seed,
                algorithm_seed,
            )
            all_results.append(result)
            print(
                f"  accuracy={result.accuracy:.4f}, "
                f"selected={result.selected_count}, "
                f"time={result.runtime_seconds:.4f}s"
            )

    return pd.DataFrame([result.__dict__ for result in all_results])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDA6323 feature selection experiment.")
    parser.add_argument(
        "--input-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="Different numbers of features used as input sizes.",
    )
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.population_size < 3:
        raise ValueError("population-size must be at least 3 for BGA tournament selection and BGWO leaders.")
    if args.iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if any(input_size < 1 for input_size in args.input_sizes):
        raise ValueError("All input sizes must be at least 1.")


def save_results_csv(results: pd.DataFrame, csv_path: Path) -> Path:
    """Save the CSV result file, using a fallback name if the target file is open."""
    try:
        results.to_csv(csv_path, index=False)
        return csv_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = csv_path.with_name(f"{csv_path.stem}_{timestamp}{csv_path.suffix}")
        results.to_csv(fallback_path, index=False)
        print(f"\nWarning: {csv_path} is currently locked or open.")
        print(f"Results were saved as: {fallback_path}")
        return fallback_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(args)
    csv_path = args.output_dir / "execution_time_results.csv"
    runtime_graph_path = args.output_dir / "figure_1_execution_time_vs_input_size.png"
    accuracy_graph_path = args.output_dir / "figure_2_accuracy_vs_input_size.png"
    selected_count_graph_path = args.output_dir / "figure_3_selected_features_vs_input_size.png"

    saved_csv_path = save_results_csv(results, csv_path)
    plot_execution_time(results, runtime_graph_path)
    plot_accuracy(results, accuracy_graph_path)
    plot_selected_feature_count(results, selected_count_graph_path)

    print("\nExperiment complete.")
    print(f"Results CSV: {saved_csv_path}")
    print(f"Figure 1: {runtime_graph_path}")
    print(f"Figure 2: {accuracy_graph_path}")
    print(f"Figure 3: {selected_count_graph_path}")
    print("\nAverage runtime by algorithm:")
    ordered_algorithms = ["BGA", "BPSO", "BGWO", "BWOA"]
    average_runtime = results.groupby("algorithm")["runtime_seconds"].mean().reindex(ordered_algorithms)
    print(average_runtime)


if __name__ == "__main__":
    main()
