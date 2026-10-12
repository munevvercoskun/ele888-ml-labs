"""k-means clustering, written from scratch.

This mirrors the algorithm from ELE 888 Lab 3 / Lab 4:

  1. pick k centres (at random, from the data)
  2. assign every point to its nearest centre
  3. move each centre to the mean of its points
  4. repeat until the assignments stop changing

Lab 4 added one more idea on top: because a bad random start can give a
bad clustering, you run the whole thing N times and keep the run with the
lowest *dispersion cost* (the total squared distance of points to their
centre). That is what `fit` does when `n_init > 1`.
"""
from __future__ import annotations

import numpy as np

from ..distances import euclidean_matrix


class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 n_init: int = 10, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        # Filled in by fit():
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None   # the dispersion cost we kept
        self.n_iter_: int | None = None
        self.history_: list[np.ndarray] = []  # centres after each iteration

    # --- one full run of k-means -------------------------------------
    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        m = X.shape[0]
        # Start from k distinct data points (Lab 4's initialisation).
        idx = rng.permutation(m)[:self.n_clusters]
        centroids = X[idx].copy()
        history = [centroids.copy()]
        labels = np.full(m, -1)

        for it in range(1, self.max_iter + 1):
            # 1. assignment step
            new_labels = np.argmin(euclidean_matrix(X, centroids), axis=1)
            # 2. update step
            new_centroids = np.array([
                X[new_labels == j].mean(axis=0) if np.any(new_labels == j)
                else centroids[j]            # keep an empty cluster where it was
                for j in range(self.n_clusters)
            ])
            history.append(new_centroids.copy())
            # 3. converged? (assignments unchanged)
            converged = np.array_equal(new_labels, labels)
            labels = new_labels
            centroids = new_centroids
            if converged:
                break

        inertia = self._dispersion_cost(X, labels, centroids)
        return centroids, labels, inertia, it, history

    @staticmethod
    def _dispersion_cost(X, labels, centroids) -> float:
        """Total squared distance of each point to its own centre."""
        assigned = centroids[labels]
        return float(np.sum((X - assigned) ** 2))

    # --- public API --------------------------------------------------
    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(self.n_init):
            run = self._single_run(X, rng)
            if best is None or run[2] < best[2]:   # smaller dispersion wins
                best = run

        self.centroids_, self.labels_, self.inertia_, self.n_iter_, self.history_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest fitted centre."""
        X = np.asarray(X, dtype=float)
        return np.argmin(euclidean_matrix(X, self.centroids_), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
