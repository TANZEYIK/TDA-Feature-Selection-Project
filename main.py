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
from utils.data_loader import prepare_dataset
from utils.fitness import make_accuracy_function, make_fitness_function
from utils.plotting import (
    plot_accuracy,
    plot_convergence,
    plot_execution_time,
    plot_selected_feature_count,
)


@dataclass
class RunResult:
    dataset: str
    algorithm: str
    input_features: int
    runtime_seconds: float
    best_fitness: float
    validation_accuracy: float
    test_accuracy: float
    selected_count: int
    selected_features: str
    convergence_history: str


def run_single_algorithm(
    dataset_name: str,
    algorithm_name: str,
    algorithm_fn: Callable[..., tuple[np.ndarray, float, list[float]]],
    input_features: int,
    population_size: int,
    iterations: int,
    dataset_random_state: int,
    algorithm_random_state: int,
) -> RunResult:
    rng = np.random.default_rng(algorithm_random_state)
    dataset = prepare_dataset(
        dataset_name,
        input_features,
        dataset_random_state,
    )
    fitness_fn, validation_accuracy_fn = make_fitness_function(
        dataset.x_train,
        dataset.x_validation,
        dataset.y_train,
        dataset.y_validation,
        rng,
    )

    start = time.perf_counter()
    best_mask, best_fitness, convergence_history = algorithm_fn(
        input_features,
        fitness_fn,
        rng,
        population_size,
        iterations,
        progress_callback=lambda iteration, score: print(
            f"  iteration {iteration}/{iterations}, best_fitness={score:.6f}",
            flush=True,
        ),
    )
    runtime = time.perf_counter() - start

    validation_accuracy = validation_accuracy_fn(best_mask)
    test_accuracy_fn = make_accuracy_function(
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
        rng,
    )
    test_accuracy = test_accuracy_fn(best_mask)
    selected_names = dataset.feature_names[best_mask.astype(bool)]
    return RunResult(
        dataset=dataset_name,
        algorithm=algorithm_name,
        input_features=input_features,
        runtime_seconds=runtime,
        best_fitness=best_fitness,
        validation_accuracy=validation_accuracy,
        test_accuracy=test_accuracy,
        selected_count=int(best_mask.sum()),
        selected_features=", ".join(selected_names),
        convergence_history=";".join(f"{value:.10g}" for value in convergence_history),
    )


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    algorithms = {
        "BGA": binary_genetic_algorithm,
        "BPSO": binary_particle_swarm_optimization,
        "BGWO": binary_grey_wolf_optimizer,
        "BWOA": binary_whale_optimization_algorithm,
    }

    all_results: list[RunResult] = []
    for dataset_index, dataset_name in enumerate(args.datasets):
        input_sizes = args.input_sizes or default_input_sizes(dataset_name)
        for input_features in input_sizes:
            dataset_seed = args.random_state + dataset_index * 100000 + input_features * 100
            for algorithm_index, (name, function) in enumerate(algorithms.items()):
                algorithm_seed = args.random_state + dataset_index * 100000 + input_features * 1000 + algorithm_index
                print(f"Running {name} on {dataset_name} with {input_features} features...", flush=True)
                result = run_single_algorithm(
                    dataset_name,
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
                    f"  validation_accuracy={result.validation_accuracy:.4f}, "
                    f"test_accuracy={result.test_accuracy:.4f}, "
                    f"selected={result.selected_count}, "
                    f"time={result.runtime_seconds:.4f}s",
                    flush=True,
                )

    return pd.DataFrame([result.__dict__ for result in all_results])


def default_input_sizes(dataset_name: str) -> list[int]:
    if dataset_name in {"arcene", "gisette", "dexter", "dorothea", "madelon"}:
        return [100, 200, 300, 400, 500]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDA6323 feature selection experiment.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["arcene", "gisette", "dexter", "dorothea", "madelon"],
        choices=["arcene", "gisette", "dexter", "dorothea", "madelon"],
        help="Datasets to use in the experiment.",
    )
    parser.add_argument(
        "--input-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Different numbers of features used as input sizes. If omitted, each dataset uses its default sizes.",
    )
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.population_size < 3:
        raise ValueError("population-size must be at least 3 for BGA tournament selection and BGWO leaders.")
    if args.iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if args.input_sizes is not None and any(input_size < 1 for input_size in args.input_sizes):
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

    saved_csv_path = save_results_csv(results, csv_path)
    generated_figures: list[Path] = []
    for dataset_name in results["dataset"].drop_duplicates():
        dataset_results = results[results["dataset"] == dataset_name]
        prefix = "" if len(results["dataset"].unique()) == 1 else f"{dataset_name}_"
        runtime_graph_path = args.output_dir / f"{prefix}figure_1_execution_time_vs_input_size.png"
        accuracy_graph_path = args.output_dir / f"{prefix}figure_2_test_accuracy_vs_input_size.png"
        selected_count_graph_path = args.output_dir / f"{prefix}figure_3_selected_features_vs_input_size.png"
        convergence_graph_path = args.output_dir / f"{prefix}figure_4_convergence_curve.png"
        plot_execution_time(dataset_results, runtime_graph_path, dataset_name)
        plot_accuracy(dataset_results, accuracy_graph_path, dataset_name)
        plot_selected_feature_count(dataset_results, selected_count_graph_path, dataset_name)
        plot_convergence(dataset_results, convergence_graph_path, dataset_name)
        generated_figures.extend(
            [
                runtime_graph_path,
                accuracy_graph_path,
                selected_count_graph_path,
                convergence_graph_path,
            ]
        )

    print("\nExperiment complete.")
    print(f"Results CSV: {saved_csv_path}")
    for index, figure_path in enumerate(generated_figures, start=1):
        print(f"Figure {index}: {figure_path}")
    print("\nAverage runtime by algorithm:")
    ordered_algorithms = ["BGA", "BPSO", "BGWO", "BWOA"]
    average_runtime = results.groupby(["dataset", "algorithm"])["runtime_seconds"].mean()
    average_runtime = average_runtime.reindex(
        pd.MultiIndex.from_product(
            [results["dataset"].drop_duplicates(), ordered_algorithms],
            names=["dataset", "algorithm"],
        )
    )
    print(average_runtime)


if __name__ == "__main__":
    main()
