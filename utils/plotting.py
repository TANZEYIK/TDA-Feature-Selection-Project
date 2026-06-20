from __future__ import annotations

from pathlib import Path

import pandas as pd


ALGORITHM_ORDER = ["BGA", "BPSO", "BGWO", "BWOA"]


def _plot_metric_on_axis(
    axis,
    results: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
) -> None:
    """Plot one metric against input size on an existing subplot axis."""
    for algorithm in ALGORITHM_ORDER:
        group = results[results["algorithm"] == algorithm]
        if group.empty:
            continue
        ordered = group.sort_values("input_features")
        axis.plot(
            ordered["input_features"],
            ordered[metric_column],
            marker="o",
            linewidth=2,
            label=algorithm,
        )
    axis.set_xlabel("Input size (number of features)")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()


def _parse_convergence_history(history: str) -> list[float]:
    return [float(value) for value in str(history).split(";") if value]


def _plot_convergence_on_axis(axis, results: pd.DataFrame) -> None:
    """Plot best-fitness convergence on an existing subplot axis."""
    largest_input_size = int(results["input_features"].max())
    convergence_rows = results[results["input_features"] == largest_input_size]

    for algorithm in ALGORITHM_ORDER:
        group = convergence_rows[convergence_rows["algorithm"] == algorithm]
        if group.empty:
            continue
        history = _parse_convergence_history(group.iloc[0]["convergence_history"])
        axis.plot(
            range(len(history)),
            history,
            marker="o",
            linewidth=2,
            label=algorithm,
        )

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best fitness")
    axis.set_title(f"Convergence Curve at {largest_input_size} Input Features")
    axis.grid(True, alpha=0.3)
    axis.legend()


def plot_dataset_summary(results: pd.DataFrame, output_path: Path, dataset_name: str) -> None:
    """Create one 2x2 summary image for a dataset's four experiment graphs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle(f"Feature Selection Results ({dataset_name})", fontsize=16)

    _plot_metric_on_axis(
        axes[0, 0],
        results,
        "runtime_seconds",
        "Execution time (seconds)",
        "Execution Time",
    )
    _plot_metric_on_axis(
        axes[0, 1],
        results,
        "test_accuracy",
        "Test accuracy",
        "Final Test Accuracy",
    )
    _plot_metric_on_axis(
        axes[1, 0],
        results,
        "selected_count",
        "Number of selected features",
        "Selected Feature Count",
    )
    _plot_convergence_on_axis(axes[1, 1], results)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
