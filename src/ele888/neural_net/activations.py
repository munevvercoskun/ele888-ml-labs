"""Activation functions and their derivatives (ELE 888 Lab 3, Part 1).

Each activation comes with its derivative because backpropagation needs
g'(z) to push the error back through a layer.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    """g'(z) = g(z) * (1 - g(z)) -- the lab's own formula."""
    g = sigmoid(z)
    return g * (1.0 - g)


def tanh(z):
    return np.tanh(z)


def dtanh(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def drelu(z):
    return (z > 0).astype(float)


ACTIVATIONS = {
    "sigmoid": (sigmoid, dsigmoid),
    "tanh": (tanh, dtanh),
    "relu": (relu, drelu),
}
