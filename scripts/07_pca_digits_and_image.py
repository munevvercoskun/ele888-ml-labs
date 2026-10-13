"""Experiment 7 -- PCA for visualisation and compression (ELE 888 Lab 4).

Two uses of the same from-scratch PCA:

  1. Visualisation: squash the 64-feature digits down to 2 components so
     all ten classes can be drawn on one scatter plot.
  2. Compression: keep only the top k principal components of an image and
     reconstruct it -- the lab's image-compression experiment. We also
     plot how much variance is retained as k grows.
"""
import _common

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_sample_image

from ele888.decomposition.pca import PCA


def digits_2d():
    digits = load_digits()
    Z = PCA(n_components=2).fit_transform(digits.data)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=digits.target, cmap="tab10", s=15)
    ax.set(title="Digits (64-D) projected to 2-D by PCA",
           xlabel="component 1", ylabel="component 2")
    fig.colorbar(sc, label="digit")
    _common.savefig(fig, "07_pca_digits_2d.png")


def image_compression():
    img = load_sample_image("china.jpg").mean(axis=2)   # grayscale: treat rows as samples
    h, w = img.shape
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set(title=f"original ({w} cols)")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], (50, 20, 5)):
        pca = PCA(n_components=k).fit(img)
        recon = pca.inverse_transform(pca.transform(img))
        retained = 100 * pca.explained_variance_ratio_.sum()
        ax.imshow(np.clip(recon, 0, 255), cmap="gray")
        ax.set(title=f"k={k} comps ({retained:.0f}% variance)")
        ax.axis("off")
        print(f"  k={k:>3}: {retained:.1f}% of variance kept")
    _common.savefig(fig, "07_pca_image_compression.png")


def main():
    print("Experiment 7: PCA for visualisation and image compression")
    digits_2d()
    image_compression()


if __name__ == "__main__":
    main()
