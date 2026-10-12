"""A small feed-forward neural network trained by backprop + SGD.

This is the from-scratch network from ELE 888 Lab 3, Part 1 and 2:

  * forward propagation through one or more hidden layers
  * the error at the output, pushed back layer by layer (backprop)
  * weights nudged downhill one example at a time (stochastic gradient
    descent), with optional L2 regularisation

It handles two cases the lab cares about:
  * binary classification (XOR) -- one output neuron, sigmoid
  * multi-class classification (car data / digits) -- one output neuron
    per class, one-hot targets, pick the highest output (one-vs-all style)

The maths is kept deliberately close to the lab's notation: s = W^T x is
the weighted input into a layer, x = g(s) is its output, and the output
sensitivity is delta = (x - y) * g'(s).
"""
from __future__ import annotations

import numpy as np

from .activations import ACTIVATIONS


class NeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid", lr=0.5,
                 epochs=2000, l2=0.0, random_state=None):
        """layer_sizes e.g. [2, 2, 1] = 2 inputs, one hidden layer of 2, 1 output."""
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.random_state = random_state
        self.g, self.dg = ACTIVATIONS[activation]
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.loss_history_: list[float] = []

    # --- setup -------------------------------------------------------
    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        self.weights_, self.biases_ = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Small random weights so neurons start out different.
            self.weights_.append(rng.normal(0, 1.0, size=(n_in, n_out)) * np.sqrt(1.0 / n_in))
            self.biases_.append(np.zeros(n_out))

    # --- forward pass ------------------------------------------------
    def _forward(self, x):
        """Return the list of pre-activations (s) and activations (x) per layer."""
        activations = [x]
        pre = []
        a = x
        for W, b in zip(self.weights_, self.biases_):
            s = a @ W + b
            pre.append(s)
            a = self.g(s)
            activations.append(a)
        return pre, activations

    # --- training ----------------------------------------------------
    def fit(self, X, Y):
        """X: (m, n_in).  Y: (m, n_out) -- one-hot for multi-class, column for binary."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self._init_params()
        m = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            order = rng.permutation(m)      # shuffle: the "stochastic" part
            epoch_loss = 0.0
            for i in order:
                x = X[i:i + 1]              # keep it 2-D: (1, n_in)
                y = Y[i:i + 1]
                pre, acts = self._forward(x)
                y_hat = acts[-1]
                epoch_loss += 0.5 * np.sum((y_hat - y) ** 2)

                # Backprop: start with the output sensitivity.
                delta = (y_hat - y) * self.dg(pre[-1])
                grads_W = [None] * len(self.weights_)
                grads_b = [None] * len(self.biases_)
                for l in range(len(self.weights_) - 1, -1, -1):
                    grads_W[l] = acts[l].T @ delta + self.l2 * self.weights_[l]
                    grads_b[l] = delta.sum(axis=0)
                    if l > 0:               # push the error back one layer
                        delta = (delta @ self.weights_[l].T) * self.dg(pre[l - 1])
                # Gradient-descent update.
                for l in range(len(self.weights_)):
                    self.weights_[l] -= self.lr * grads_W[l]
                    self.biases_[l] -= self.lr * grads_b[l]

            self.loss_history_.append(epoch_loss / m)
        return self

    # --- prediction --------------------------------------------------
    def predict_scores(self, X):
        """Raw output-neuron values -- the lab's 'score matrix'."""
        X = np.asarray(X, dtype=float)
        _, acts = self._forward(X)
        return acts[-1]

    def predict(self, X):
        scores = self.predict_scores(X)
        if scores.shape[1] == 1:
            return (scores[:, 0] >= 0.5).astype(int)      # binary threshold
        return np.argmax(scores, axis=1)                  # multi-class: highest neuron

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
