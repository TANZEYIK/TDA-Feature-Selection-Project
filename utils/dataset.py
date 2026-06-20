from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_dataset(input_features: int, random_state: int):
    """Load and prepare a fixed-size feature matrix for the experiment."""
    dataset = load_breast_cancer()
    total_features = dataset.data.shape[1]
    if input_features < 1 or input_features > total_features:
        raise ValueError(
            f"input_features must be between 1 and {total_features}, got {input_features}."
        )

    x = dataset.data[:, :input_features]
    y = dataset.target
    feature_names = dataset.feature_names[:input_features]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, feature_names
