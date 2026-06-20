from __future__ import annotations

from pathlib import Path

import pandas as pd


def _plot_metric(
    results: pd.DataFrame,
    output_path: Path,
    metric_column: str,
    y_label: str,
    title: str,
    dataset_name: str | None = None,
) -> None:
    """Plot one metric against input size for all algorithms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    algorithm_order = ["BGA", "BPSO", "BGWO", "BWOA"]
    for algorithm in algorithm_order:
        group = results[results["algorithm"] == algorithm]
        if group.empty:
            continue
        ordered = group.sort_values("input_features")
        plt.plot(
            ordered["input_features"],
            ordered[metric_column],
            marker="o",
            linewidth=2,
            label=algorithm,
        )
    plt.xlabel("Input size (number of features)")
    plt.ylabel(y_label)
    if dataset_name:
        title = f"{title} ({dataset_name})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_execution_time(results: pd.DataFrame, output_path: Path, dataset_name: str | None = None) -> None:
    """Plot execution time against input size for all algorithms."""
    _plot_metric(
        results=results,
        output_path=output_path,
        metric_column="runtime_seconds",
        y_label="Execution time (seconds)",
        title="Execution Time of Metaheuristic Feature Selection Algorithms",
        dataset_name=dataset_name,
    )


def plot_accuracy(results: pd.DataFrame, output_path: Path, dataset_name: str | None = None) -> None:
    """Plot final test classification accuracy against input size for all algorithms."""
    _plot_metric(
        results=results,
        output_path=output_path,
        metric_column="test_accuracy",
        y_label="Test accuracy",
        title="Final Test Accuracy of Selected Feature Subsets",
        dataset_name=dataset_name,
    )


def plot_selected_feature_count(results: pd.DataFrame, output_path: Path, dataset_name: str | None = None) -> None:
    """Plot selected feature count against input size for all algorithms."""
    _plot_metric(
        results=results,
        output_path=output_path,
        metric_column="selected_count",
        y_label="Number of selected features",
        title="Selected Feature Count of Metaheuristic Algorithms",
        dataset_name=dataset_name,
    )


def _parse_convergence_history(history: str) -> list[float]:
    return [float(value) for value in str(history).split(";") if value]


def plot_convergence(results: pd.DataFrame, output_path: Path, dataset_name: str | None = None) -> None:
    """Plot best-fitness convergence for the largest input size of one dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    largest_input_size = int(results["input_features"].max())
    convergence_rows = results[results["input_features"] == largest_input_size]

    plt.figure(figsize=(9, 6))
    algorithm_order = ["BGA", "BPSO", "BGWO", "BWOA"]
    for algorithm in algorithm_order:
        group = convergence_rows[convergence_rows["algorithm"] == algorithm]
        if group.empty:
            continue
        history = _parse_convergence_history(group.iloc[0]["convergence_history"])
        plt.plot(
            range(len(history)),
            history,
            marker="o",
            linewidth=2,
            label=algorithm,
        )

    title = f"Convergence Curve at {largest_input_size} Input Features"
    if dataset_name:
        title = f"{title} ({dataset_name})"
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
