"""Experiment 5 -- SVM vs neural network (ELE 888 Lab 3, Part 3 & 4).

The lab re-solved its classification problems with SVMs and compared them
to the neural networks. We do the same two things:

  1. A 2-D synthetic set so you can actually see the large-margin boundary
     the from-scratch LinearSVM finds.
  2. A real binary task -- telling the digit 3 apart from the digit 8 --
     solved three ways: our from-scratch linear SVM, scikit-learn's SVC
     (linear and RBF kernels), and the from-scratch neural network.
"""
import _common

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ele888.neural_net.network import NeuralNetwork
from ele888.svm.linear_svm import LinearSVM


def margin_plot():
    X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.1, random_state=6)
    svm = LinearSVM(C=1.0, lr=0.01, epochs=500, random_state=0).fit(X, y)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=40, edgecolors="k")
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    ys = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    gx, gy = np.meshgrid(xs, ys)
    zz = svm.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contour(gx, gy, zz, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"])       # margin + boundary
    ax.set(title="From-scratch linear SVM: boundary (solid) and margins (dashed)")
    _common.savefig(fig, "05_svm_margin.png")
    print(f"  2-D synthetic: LinearSVM train accuracy {svm.score(X, y):.2f}")


def digits_3_vs_8():
    digits = load_digits()
    mask = np.isin(digits.target, [3, 8])
    X = digits.data[mask]
    y = (digits.target[mask] == 8).astype(int)      # 1 = eight, 0 = three
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    print("  telling digit 3 apart from digit 8:")
    ours = LinearSVM(C=1.0, lr=0.001, epochs=300, random_state=0).fit(Xtr, ytr)
    print(f"    from-scratch linear SVM : {ours.score(Xte, yte):.3f}")
    print(f"    scikit-learn SVC linear : {SVC(kernel='linear').fit(Xtr, ytr).score(Xte, yte):.3f}")
    print(f"    scikit-learn SVC rbf    : {SVC(kernel='rbf').fit(Xtr, ytr).score(Xte, yte):.3f}")
    net = NeuralNetwork([64, 16, 1], lr=0.3, epochs=150, l2=1e-4, random_state=0).fit(Xtr, ytr)
    print(f"    from-scratch neural net : {net.score(Xte, yte):.3f}")


def main():
    print("Experiment 5: SVM vs neural network")
    margin_plot()
    digits_3_vs_8()


if __name__ == "__main__":
    main()
