from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid function used to convert continuous values into probabilities."""
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def ensure_non_empty(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Avoid invalid empty feature subsets by forcing one random feature to be selected."""
    fixed = mask.astype(int).copy()
    if fixed.sum() == 0:
        fixed[rng.integers(0, len(fixed))] = 1
    return fixed
