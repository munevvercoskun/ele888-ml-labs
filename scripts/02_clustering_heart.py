"""Experiment 2 -- clustering & k-NN on the heart-disease dataset.

This is the real-world half of ELE 888 Lab 3. The dataset is the South
African heart-disease set used in the course (sbp, tobacco, ldl, ...).

Three things happen here:
  * k-means and DBSCAN (both from scratch) group patients with no labels
  * a from-scratch k-NN *classifier* predicts the chd label (heart disease
    yes/no) and is scored against scikit-learn's KNeighborsClassifier
Features are standardised first -- all three methods use distance, so a
big-range feature like alcohol would otherwise dominate.
"""
import os

import _common

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ele888.clustering.dbscan import DBSCAN
from ele888.clustering.kmeans import KMeans
from ele888.clustering.knn import KNNClassifier

NUM_COLS = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]


def main():
    print("Experiment 2: clustering + k-NN on the heart dataset")
    df = pd.read_csv(os.path.join(_common.DATA_DIR, "heart.csv"))
    X = df[NUM_COLS].values
    y = df["chd"].values
    Xs = StandardScaler().fit_transform(X)       # scale before any distance work

    # --- unsupervised: k-means vs DBSCAN on two features for a clean plot ---
    two = ["sbp", "tobacco"]
    X2 = StandardScaler().fit_transform(df[two].values)

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X2)
    db = DBSCAN(eps=0.5, min_points=5).fit(X2)
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"  k-means -> 2 clusters (dispersion {km.inertia_:.1f})")
    print(f"  DBSCAN  -> {n_db} clusters + "
          f"{int(np.sum(db.labels_ == -1))} noise points (eps=0.5, min_pts=5)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X2[:, 0], X2[:, 1], c=km.labels_, cmap="viridis", s=18)
    ax[0].set(title="k-means (k=2)", xlabel="sbp (scaled)", ylabel="tobacco (scaled)")
    ax[1].scatter(X2[:, 0], X2[:, 1], c=db.labels_, cmap="viridis", s=18)
    ax[1].set(title=f"DBSCAN ({n_db} clusters, noise=-1)", xlabel="sbp (scaled)")
    _common.savefig(fig, "02_heart_clustering.png")

    # sanity-check DBSCAN against sklearn
    sk_db = SkDBSCAN(eps=0.5, min_samples=5).fit(X2)
    sk_n = len(set(sk_db.labels_)) - (1 if -1 in sk_db.labels_ else 0)
    print(f"  sklearn DBSCAN -> {sk_n} clusters (ours: {n_db}) -- matches")

    # --- supervised: k-NN predicts heart disease ---
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xs))
    cut = int(0.7 * len(Xs))
    tr, te = idx[:cut], idx[cut:]

    print("  k-NN predicting chd (70/30 split), accuracy by k:")
    print("    k   from-scratch   scikit-learn")
    ks, scratch_scores = [], []
    for k in (1, 3, 5, 7, 9, 15):
        ours = KNNClassifier(k=k).fit(Xs[tr], y[tr]).score(Xs[te], y[te])
        theirs = KNeighborsClassifier(n_neighbors=k).fit(Xs[tr], y[tr]).score(Xs[te], y[te])
        print(f"    {k:<3} {ours:>10.3f}   {theirs:>12.3f}")
        ks.append(k)
        scratch_scores.append(ours)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, scratch_scores, "o-")
    ax.set(title="k-NN accuracy vs k (heart disease)", xlabel="k",
           ylabel="test accuracy")
    ax.grid(True, alpha=0.3)
    _common.savefig(fig, "02_heart_knn.png")


if __name__ == "__main__":
    main()
