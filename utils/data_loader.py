from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"


DATASET_FILES = {
    "arcene": "arcene.arff",
    "gisette": "gisette.arff",
    "dexter": "Dexter.arff",
    "dorothea": "Dorothea.arff",
    "madelon": "madelon.arff",
}


@dataclass
class PreparedDataset:
    dataset_name: str
    input_features: int
    x_train: np.ndarray | csr_matrix
    x_validation: np.ndarray | csr_matrix
    x_test: np.ndarray | csr_matrix
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: np.ndarray


def load_dataset(dataset_name: str) -> tuple[np.ndarray | csr_matrix, np.ndarray, np.ndarray]:
    """Load one locally downloaded NIPS 2003 ARFF dataset."""
    normalized_name = dataset_name.lower()
    if normalized_name not in DATASET_FILES:
        supported = ", ".join(DATASET_FILES)
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}")

    file_path = DATASET_DIR / DATASET_FILES[normalized_name]
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {file_path}\n"
            "Download the ARFF file from OpenML and place it in the dataset folder."
        )
    return _load_arff(file_path)


def _load_arff(file_path: Path) -> tuple[np.ndarray | csr_matrix, np.ndarray, np.ndarray]:
    attributes: list[str] = []
    data_lines: list[str] = []
    in_data_section = False

    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lower_line = line.lower()
            if lower_line.startswith("@attribute"):
                attributes.append(line)
            elif lower_line.startswith("@data"):
                in_data_section = True
            elif in_data_section:
                data_lines.append(line)

    if len(attributes) < 2:
        raise ValueError(f"Could not find attributes in {file_path}")
    if not data_lines:
        raise ValueError(f"Could not find data rows in {file_path}")

    n_features = len(attributes) - 1
    feature_names = np.array([f"feature_{index + 1}" for index in range(n_features)])

    if data_lines[0].startswith("{"):
        x, y = _parse_sparse_arff_rows(data_lines, n_features)
    else:
        x, y = _parse_dense_arff_rows(data_lines, n_features)
    return x, y, feature_names


def _parse_dense_arff_rows(data_lines: list[str], n_features: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[str] = []
    expected_values = n_features + 1

    for line_number, line in enumerate(data_lines, start=1):
        values = [value.strip() for value in line.split(",")]
        if len(values) != expected_values:
            raise ValueError(
                f"Dense ARFF row {line_number} has {len(values)} values; expected {expected_values}."
            )
        rows.append([float(value) for value in values[:n_features]])
        labels.append(values[-1])

    return np.asarray(rows, dtype=float), np.asarray(labels)


def _parse_sparse_arff_rows(data_lines: list[str], n_features: int) -> tuple[csr_matrix, np.ndarray]:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    labels: list[str] = []

    for row_number, line in enumerate(data_lines):
        content = line.strip()
        if not content:
            continue
        if not (content.startswith("{") and content.endswith("}")):
            raise ValueError(f"Sparse ARFF row {row_number + 1} is not in sparse format.")

        label = "0"
        entries = content[1:-1].split(",")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            index_text, value_text = entry.split(maxsplit=1)
            column_index = int(index_text)
            if column_index == n_features:
                label = value_text.strip()
            elif 0 <= column_index < n_features:
                row_indices.append(row_number)
                column_indices.append(column_index)
                values.append(float(value_text))
            else:
                raise ValueError(
                    f"Sparse ARFF row {row_number + 1} contains invalid column index {column_index}."
                )
        labels.append(label)

    x = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(labels), n_features),
        dtype=float,
    )
    return x, np.asarray(labels)


def prepare_dataset(dataset_name: str, input_features: int, random_state: int) -> PreparedDataset:
    """Load, split, and scale a fixed-size feature pool for the experiment."""
    x, y, all_feature_names = load_dataset(dataset_name)
    total_features = x.shape[1]
    if input_features < 1 or input_features > total_features:
        raise ValueError(
            f"input_features must be between 1 and {total_features}, got {input_features}."
        )

    rng = np.random.default_rng(random_state)
    if input_features == total_features:
        feature_indices = np.arange(total_features)
    else:
        feature_indices = np.sort(rng.choice(total_features, size=input_features, replace=False))

    x = x[:, feature_indices]
    if issparse(x):
        x = x.tocsr()
    feature_names = all_feature_names[feature_indices]

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=random_state + 1,
        stratify=y_train_validation,
    )

    scaler = StandardScaler(with_mean=not issparse(x_train))
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_test = scaler.transform(x_test)

    return PreparedDataset(
        dataset_name=dataset_name,
        input_features=input_features,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        feature_names=feature_names,
    )
