"""Experiment 3 -- the XOR problem with a from-scratch neural network.

ELE 888 Lab 3, Part 1. XOR is the classic example of data that *no*
straight line can separate, so logistic regression fails on it. A network
with one hidden layer bends the boundary and solves it. We train with
stochastic gradient descent and plot the loss falling and the decision
regions it learns.
"""
import _common

import matplotlib.pyplot as plt
import numpy as np

from ele888.neural_net.network import NeuralNetwork

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Y = np.array([0, 1, 1, 0])          # XOR


def main():
    print("Experiment 3: XOR with a from-scratch neural network")
    net = NeuralNetwork([2, 4, 1], activation="sigmoid", lr=0.5,
                        epochs=3000, random_state=1).fit(X, Y)

    preds = net.predict(X)
    print("  inputs -> prediction (target):")
    for xi, p, t in zip(X, preds, Y):
        print(f"    {xi.astype(int)} -> {p} ({t})")
    print(f"  accuracy: {net.score(X, Y):.2f}  (1.00 means XOR solved)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(net.loss_history_)
    ax[0].set(title="Training loss (SGD)", xlabel="epoch", ylabel="mean squared error")
    ax[0].grid(True, alpha=0.3)

    # Decision regions: colour the plane by the network's prediction.
    xs = np.linspace(-0.2, 1.2, 200)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = net.predict(grid).reshape(gx.shape)
    ax[1].contourf(gx, gy, zz, alpha=0.3, cmap="coolwarm")
    ax[1].scatter(X[:, 0], X[:, 1], c=Y, cmap="coolwarm", s=200, edgecolors="k")
    ax[1].set(title="Learned decision regions for XOR", xlabel="x1", ylabel="x2")
    _common.savefig(fig, "03_xor.png")


if __name__ == "__main__":
    main()
