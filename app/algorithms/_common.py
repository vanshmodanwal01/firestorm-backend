"""Small numeric helpers shared by the trend algorithms."""
import numpy as np


def clean_1d(x, name: str = "x") -> np.ndarray:
    """Return a float array; refuse NaNs (missing data are never imputed)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN; remove missing values before running a trend test")
    return arr


def pairwise_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(y_j - y_i) / (x_j - x_i) for all i < j with x_j != x_i."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    dy = y[j] - y[i]
    keep = dx != 0
    return dy[keep] / dx[keep]
