"""Principal Component Analysis, written from scratch (ELE 888 Lab 4).

PCA finds the directions the data varies along the most and lets you keep
only the top few, projecting everything onto them. Two uses from the lab:
compression (fewer numbers describe each example) and visualisation
(squash to 2-D so you can plot it).

The maths is the lab's: centre the data, build the covariance matrix,
take its eigenvectors via SVD, and keep the top k. `transform` projects
down to k dimensions; `inverse_transform` reconstructs an approximation
back in the original space (that is the image-compression step in Lab 4).
"""
from __future__ import annotations

import numpy as np


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None      # (n_components, n_features)
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X) -> "PCA":
        X = np.asarray(X, dtype=float)
        # 1. Centre the data (PCA is about variance around the mean).
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # 2. SVD of the centred data -- more stable than eig(cov) and is
        #    exactly what the lecture's "eigenvectors of Sigma via SVD" means.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        # 3. How much variance each kept direction explains.
        var = S ** 2 / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (var / var.sum())[:self.n_components]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        """Rebuild an approximation of the original data from k components."""
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.mean_
