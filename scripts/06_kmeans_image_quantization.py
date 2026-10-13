"""Experiment 6 -- colour quantization with k-means (ELE 888 Lab 4, Part 2).

Every pixel is a point in 3-D colour space (R, G, B). Run k-means with
k=16 and you get the 16 "average" colours of the image; repaint each pixel
as its nearest centre and you have the same picture drawn with only 16
colours. That is image compression by clustering -- exactly the lab's
jellybeans experiment, here on a sample photo that ships with scikit-learn.
"""
import _common

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image

from ele888.clustering.kmeans import KMeans


def quantize(img, k, rng_seed=0):
    h, w, _ = img.shape
    pixels = img.reshape(-1, 3).astype(float)
    # Fit on a subsample for speed, then map every pixel to its centre.
    rng = np.random.default_rng(rng_seed)
    sample = pixels[rng.choice(len(pixels), size=3000, replace=False)]
    km = KMeans(n_clusters=k, n_init=3, max_iter=50, random_state=rng_seed).fit(sample)
    labels = km.predict(pixels)
    out = km.centroids_[labels].reshape(h, w, 3)
    return np.clip(out / 255.0, 0, 1), km.centroids_


def main():
    print("Experiment 6: k-means image colour quantization")
    img = load_sample_image("flower.jpg")
    print(f"  original image: {img.shape}, {len(np.unique(img.reshape(-1, 3), axis=0))} colours")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img)
    axes[0].set(title="original")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], (16, 8, 4)):
        out, _ = quantize(img, k)
        ax.imshow(out)
        ax.set(title=f"k = {k} colours")
        ax.axis("off")
        print(f"  quantized to {k} colours")
    _common.savefig(fig, "06_image_quantization.png")


if __name__ == "__main__":
    main()
