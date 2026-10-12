"""A linear Support Vector Machine, written from scratch.

ELE 888 Lab 3 used MATLAB's SVM toolbox; here we implement the core idea
by hand so it isn't a black box. An SVM looks for the boundary with the
widest margin between two classes. We minimise the standard soft-margin
objective by sub-gradient descent:

    (1/2) * ||w||^2  +  C * mean( max(0, 1 - y*(w.x + b)) )

The max(0, 1 - y*f(x)) term is the hinge loss -- zero once a point is on
the correct side of the margin, growing linearly as it crosses over. C is
exactly the regularisation dial from the lectures (large C = punish every
mistake, narrow margin; small C = wider, calmer margin). Labels are +1/-1.

For non-linear boundaries you reach for a kernel, which is what the
scikit-learn SVC in scripts/05 demonstrates -- see the guide for why the
kernel trick is hard to hand-roll and easy to call.
"""
from __future__ import annotations

import numpy as np


class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000, random_state=None):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y) <= 0, -1, 1).astype(float)   # force +1/-1
        m, n = X.shape
        rng = np.random.default_rng(self.random_state)
        self.w_ = np.zeros(n)
        self.b_ = 0.0

        for _ in range(self.epochs):
            for i in rng.permutation(m):
                margin = y[i] * (X[i] @ self.w_ + self.b_)
                if margin >= 1:
                    # Correctly classified & outside margin: only shrink w.
                    self.w_ -= self.lr * self.w_
                else:
                    # Inside margin / misclassified: push toward correct side.
                    self.w_ -= self.lr * (self.w_ - self.C * y[i] * X[i])
                    self.b_ += self.lr * self.C * y[i]
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
