"""DBSCAN, written from scratch (ELE 888 Lab 3, Part C).

The idea that makes DBSCAN different from k-means: you don't tell it how
many clusters there are. Instead you say how close counts as "neighbours"
(eps) and how many neighbours make a dense region (min_points). It grows a
cluster outward from any dense point, and anything that never lands in a
dense region is labelled noise.

Labels follow the scikit-learn convention: 0, 1, 2, ... for clusters and
-1 for noise.
"""
from __future__ import annotations

import numpy as np

from ..distances import euclidean


class DBSCAN:
    def __init__(self, eps: float = 2.0, min_points: int = 2):
        self.eps = eps
        self.min_points = min_points
        self.labels_: np.ndarray | None = None

    def _region_query(self, X, i):
        """Indices of every point within eps of point i (including i)."""
        return [j for j in range(len(X)) if euclidean(X[i], X[j]) <= self.eps]

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=float)
        n = len(X)
        labels = np.full(n, -2)   # -2 = not yet visited
        cluster_id = -1

        for i in range(n):
            if labels[i] != -2:
                continue                      # already handled
            neighbours = self._region_query(X, i)
            if len(neighbours) < self.min_points:
                labels[i] = -1                # mark as noise (may be claimed later)
                continue

            cluster_id += 1                   # start a new cluster
            labels[i] = cluster_id
            seeds = list(neighbours)
            k = 0
            while k < len(seeds):
                j = seeds[k]
                if labels[j] == -1:
                    labels[j] = cluster_id    # noise point on a cluster border
                elif labels[j] == -2:
                    labels[j] = cluster_id
                    j_neighbours = self._region_query(X, j)
                    if len(j_neighbours) >= self.min_points:
                        seeds.extend(j_neighbours)   # j is also a core point
                k += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
