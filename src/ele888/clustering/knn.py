"""k-nearest-neighbours classifier, written from scratch (ELE 888 Lab 3).

k-NN is the simplest "learning" there is: it doesn't train at all. To
classify a new point it finds the k closest labelled points and takes a
majority vote. All the work happens at prediction time, which is exactly
why it gets slow on big datasets -- the lab's own observation.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..distances import euclidean_matrix


class KNNClassifier:
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        # "Training" is just remembering the data.
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dists = euclidean_matrix(X, self.X_train_)       # (m_test, m_train)
        preds = []
        for row in dists:
            nearest = np.argsort(row)[:self.k]           # indices of k closest
            votes = Counter(self.y_train_[nearest].tolist())
            preds.append(votes.most_common(1)[0][0])     # majority label
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
