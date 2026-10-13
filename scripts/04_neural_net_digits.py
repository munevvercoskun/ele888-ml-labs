"""Experiment 4 -- multi-class classification with the from-scratch network.

ELE 888 Lab 3, Part 2 used a car-acceptability dataset and a one-vs-all
neural network. Here we use the 8x8 handwritten-digits dataset (10 classes,
64 features) -- same shape of problem, and it doubles as the input for the
PCA and image experiments later.

The from-scratch network (one hidden layer, one output neuron per class,
pick the highest) is trained and then compared to scikit-learn's
MLPClassifier, which is the same kind of model a Keras `Dense` network
would be.
"""
import _common

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ele888.neural_net.network import NeuralNetwork


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def main():
    print("Experiment 4: multi-class neural network on handwritten digits")
    digits = load_digits()
    X, y = digits.data, digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # From-scratch network.
    net = NeuralNetwork([64, 32, 10], activation="sigmoid", lr=0.3,
                        epochs=120, l2=1e-4, random_state=0)
    net.fit(Xtr_s, one_hot(ytr, 10))
    ours = net.score(Xte_s, yte)
    print(f"  from-scratch NN  test accuracy: {ours:.3f}")

    # scikit-learn's MLP (the Keras-equivalent).
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0)
    mlp.fit(Xtr_s, ytr)
    theirs = mlp.score(Xte_s, yte)
    print(f"  scikit-learn MLP test accuracy: {theirs:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(net.loss_history_)
    ax[0].set(title="From-scratch NN: training loss", xlabel="epoch", ylabel="MSE")
    ax[0].grid(True, alpha=0.3)
    # Show a few test digits with our predictions.
    preds = net.predict(Xte_s)
    for i in range(10):
        sub = fig.add_subplot(2, 10, 10 + i + 1)
        sub.imshow(Xte[i].reshape(8, 8), cmap="gray_r")
        sub.set_title(str(preds[i]), fontsize=9)
        sub.axis("off")
    ax[1].axis("off")
    ax[1].set_title("Sample predictions (bottom row)")
    _common.savefig(fig, "04_digits_nn.png")

    print("  classification report (from-scratch NN):")
    print(classification_report(yte, preds, digits=3))


if __name__ == "__main__":
    main()
