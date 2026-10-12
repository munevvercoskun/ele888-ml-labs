"""Distance helpers shared across the clustering algorithms.

Everything here works on NumPy arrays. The whole point of keeping the
distance in one place is that k-means, k-NN and DBSCAN all lean on the
exact same "how far apart are these two points" question (ELE 888 Lab 3),
so it should be defined once and reused.
"""
from __future__ import annotations

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance between two points of equal length."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def euclidean_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Distance from every row of X to every center.

    Returns an (m x k) matrix where entry [i, j] is the distance from
    example i to center j. This is the vectorised version -- no Python
    loop over the examples -- which is what keeps k-means fast enough to
    re-run many times.
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    # (m, 1, n) - (1, k, n) -> (m, k, n), square, sum over features, sqrt.
    diff = X[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))
