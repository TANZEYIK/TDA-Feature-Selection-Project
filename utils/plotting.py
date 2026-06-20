from __future__ import annotations

from pathlib import Path

import pandas as pd


def _plot_metric(
    results: pd.DataFrame,
    output_path: Path,
    metric_column: str,
    y_label: str,
    title: str,
) -> None:
    """Plot one metric against input size for all algorithms."""
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
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_execution_time(results: pd.DataFrame, output_path: Path) -> None:
    """Plot execution time against input size for all algorithms."""
    _plot_metric(
        results=results,
        output_path=output_path,
        metric_column="runtime_seconds",
        y_label="Execution time (seconds)",
        title="Execution Time of Metaheuristic Feature Selection Algorithms",
    )


def plot_accuracy(results: pd.DataFrame, output_path: Path) -> None:
    """Plot classification accuracy against input size for all algorithms."""
    _plot_metric(
        results=results,
        output_path=output_path,
        metric_column="accuracy",
        y_label="Accuracy",
        title="Classification Accuracy of Selected Feature Subsets",
    )


def plot_selected_feature_count(results: pd.DataFrame, output_path: Path) -> None:
    """Plot selected feature count against input size for all algorithms."""
    _plot_metric(
        results=results,
        output_path=output_path,
        metric_column="selected_count",
        y_label="Number of selected features",
        title="Selected Feature Count of Metaheuristic Algorithms",
    )
