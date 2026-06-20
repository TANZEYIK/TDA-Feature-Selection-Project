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
from utils.data_loader import PreparedDataset, prepare_dataset
from utils.fitness import make_accuracy_function, make_fitness_function
from utils.plotting import plot_dataset_summary


EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001
SELECTED_FEATURE_PREVIEW_LIMIT = 30


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
    selected_feature_preview: str
    convergence_history: str
    iterations_run: int
    stopped_early: bool


def selected_feature_preview(feature_names: np.ndarray, mask: np.ndarray) -> str:
    """Return a compact preview of selected feature names for readable CSV output."""
    selected_names = feature_names[mask.astype(bool)]
    preview_names = selected_names[:SELECTED_FEATURE_PREVIEW_LIMIT]
    preview = ", ".join(preview_names)
    remaining_count = len(selected_names) - len(preview_names)
    if remaining_count > 0:
        preview = f"{preview}, ... (+{remaining_count} more)"
    return preview


def run_single_algorithm(
    dataset: PreparedDataset,
    algorithm_name: str,
    algorithm_fn: Callable[..., tuple[np.ndarray, float, list[float], bool]],
    population_size: int,
    iterations: int,
    algorithm_random_state: int,
) -> RunResult:
    rng = np.random.default_rng(algorithm_random_state)
    fitness_fn, validation_accuracy_fn = make_fitness_function(
        dataset.x_train,
        dataset.x_validation,
        dataset.y_train,
        dataset.y_validation,
        rng,
    )

    start = time.perf_counter()
    best_mask, best_fitness, convergence_history, stopped_early = algorithm_fn(
        dataset.input_features,
        fitness_fn,
        rng,
        population_size,
        iterations,
        progress_callback=lambda iteration, score: print(
            f"  iteration {iteration}/{iterations}, best_fitness={score:.6f}",
            flush=True,
        ),
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
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
    return RunResult(
        dataset=dataset.dataset_name,
        algorithm=algorithm_name,
        input_features=dataset.input_features,
        runtime_seconds=runtime,
        best_fitness=best_fitness,
        validation_accuracy=validation_accuracy,
        test_accuracy=test_accuracy,
        selected_count=int(best_mask.sum()),
        selected_feature_preview=selected_feature_preview(dataset.feature_names, best_mask),
        convergence_history=";".join(f"{value:.10g}" for value in convergence_history),
        iterations_run=max(len(convergence_history) - 1, 0),
        stopped_early=stopped_early,
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
        dataset_seed = args.random_state + dataset_index * 100000
        dataset = prepare_dataset(dataset_name, dataset_seed)
        for algorithm_index, (name, function) in enumerate(algorithms.items()):
            algorithm_seed = args.random_state + dataset_index * 100000 + algorithm_index
            print(
                f"Running {name} on {dataset_name} with {dataset.input_features} features...",
                flush=True,
            )
            result = run_single_algorithm(
                dataset,
                name,
                function,
                args.population_size,
                args.iterations,
                algorithm_seed,
            )
            all_results.append(result)
            print(
                f"  validation_accuracy={result.validation_accuracy:.4f}, "
                f"test_accuracy={result.test_accuracy:.4f}, "
                f"selected={result.selected_count}, "
                f"iterations_run={result.iterations_run}, "
                f"stopped_early={result.stopped_early}, "
                f"time={result.runtime_seconds:.4f}s",
                flush=True,
            )

    return pd.DataFrame([result.__dict__ for result in all_results])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDA6323 feature selection experiment.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["arcene", "gisette", "dexter", "dorothea", "madelon"],
        choices=["arcene", "gisette", "dexter", "dorothea", "madelon"],
        help="Datasets to use in the experiment.",
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
        summary_graph_path = args.output_dir / f"{dataset_name}_summary.png"
        plot_dataset_summary(dataset_results, summary_graph_path, dataset_name)
        generated_figures.append(summary_graph_path)

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
