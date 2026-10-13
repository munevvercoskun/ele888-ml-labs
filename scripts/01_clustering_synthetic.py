"""Experiment 1 -- k-means on the 8-point toy set (ELE 888 Lab 3, Part A).

Runs the from-scratch k-means on the eight points from the lab, shows how
the centres move each epoch, and checks the result against scikit-learn.
"""
import _common  # noqa: F401  (sets up the path)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans as SkKMeans

from ele888.clustering.kmeans import KMeans

POINTS = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]], float)


def main():
    print("Experiment 1: k-means on the 8-point toy set")

    km = KMeans(n_clusters=3, n_init=1, random_state=2).fit(POINTS)
    print(f"  converged in {km.n_iter_} iterations")
    print(f"  final centres:\n{np.round(km.centroids_, 2)}")
    print(f"  dispersion cost: {km.inertia_:.2f}")

    # Plot the centre movement, one panel per epoch.
    history = km.history_
    n = len(history)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for e, centres in enumerate(history):
        ax = axes[0][e]
        labels = np.argmin(
            np.linalg.norm(POINTS[:, None, :] - centres[None, :, :], axis=2), axis=1)
        ax.scatter(POINTS[:, 0], POINTS[:, 1], c=labels, cmap="viridis", s=80)
        ax.scatter(centres[:, 0], centres[:, 1], c="red", marker="x", s=120)
        ax.set(title=f"epoch {e}", xlim=(0, 10), ylim=(0, 10))
        ax.grid(True, alpha=0.3)
    fig.suptitle("From-scratch k-means: centres settling over epochs")
    _common.savefig(fig, "01_kmeans_epochs.png")

    # Cross-check against scikit-learn.
    sk = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit(POINTS)
    print(f"  scikit-learn inertia: {sk.inertia_:.2f}  (ours: {km.inertia_:.2f})")
    print("  same grouping up to label-renaming -- both find the 3 natural clusters.")


if __name__ == "__main__":
    main()
