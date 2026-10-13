# ELE 888 — Machine Learning, from scratch

The core algorithms from my ELE 888 (Intelligent Systems) labs, written by hand in
Python — then checked against scikit-learn to prove the hand-written versions actually
work. Clustering, a neural network trained with backpropagation, a support vector
machine, and PCA, each built up from NumPy and then run on real data.

> ELE 888 — Intelligent Systems (Machine Learning). The labs were done in MATLAB with a
> lot of the scaffolding handed to us. I re-did them in Python from the algorithm up,
> because being able to write k-means or backprop myself — not just call `.fit()` — is
> the part worth showing.

```
make setup      # one-time: install numpy, scikit-learn, matplotlib, pandas
make run        # run all 7 experiments, write the figures
```

---

## The headline: my code vs the library

Every number here is printed by the scripts in this repo. The point of each experiment
is that the from-scratch implementation lands on the same answer as scikit-learn.

| Experiment | From scratch | scikit-learn | Notes |
|---|---|---|---|
| k-means on the 8-point toy set | dispersion **14.33** | **14.33** | identical clustering |
| k-NN predicting heart disease | **0.63** | **0.63** | same accuracy at every k |
| XOR with a neural network | **100%** | — | the problem no straight line can solve |
| Digit recognition, neural net | **98.0%** | 98.0% (MLP) | 10 classes, one hidden layer, backprop |
| Digit 3 vs 8 | SVM **97%** / NN **99%** | SVC linear 99%, RBF **100%** | kernel wins, as expected |
| PCA image compression | **95%** variance in 50 comps | — | reconstructs the image from a fraction of the data |

---

## What's in here

```
src/ele888/
  distances.py              euclidean distance, shared by all three clustering methods
  clustering/
    kmeans.py               k-means with the dispersion-cost restart trick (Lab 4)
    dbscan.py               DBSCAN — density-based, finds its own number of clusters
    knn.py                  k-nearest-neighbours classifier
  neural_net/
    activations.py          sigmoid/tanh/relu and their derivatives
    network.py              forward prop + backprop + SGD, binary and multi-class
  svm/
    linear_svm.py           soft-margin linear SVM by sub-gradient descent (hinge loss)
  decomposition/
    pca.py                  PCA via SVD: transform, inverse_transform (compression)

scripts/                    7 experiments, each saves a figure and prints its results
figures/                    generated plots (made by `make run`)
data/heart.csv              the South African heart-disease dataset used in Lab 3
```

---

## The four topics

**Clustering (Lab 3).** k-means, DBSCAN and k-NN, all from scratch. k-means and k-NN
match scikit-learn to the digit; DBSCAN agrees on the cluster count. The thing that
clicked re-writing these: k-means, k-NN and DBSCAN are all just *"how far apart are two
points"* asked three different ways, which is why they share one `distances.py`.

**Neural networks (Lab 3).** A feed-forward net with one hidden layer, trained by
backpropagation + stochastic gradient descent. It solves XOR (the standard "no linear
boundary exists" problem) and then reaches 98% on 10-class handwritten digits — the same
score scikit-learn's `MLPClassifier` gets, which is the model a Keras `Dense` network
would be too.

**SVM (Lab 3).** A linear SVM minimising the soft-margin (hinge-loss) objective, so the
`C` dial and the large-margin idea aren't a black box. On telling a 3 from an 8 it gets
97%; swapping in scikit-learn's RBF-kernel SVC pushes it to 100%, which is the whole
argument for kernels in one number.

**PCA & image compression (Lab 4).** PCA by SVD of the centred data. Two uses: squashing
64-dimensional digits to 2-D so all ten classes fit on one plot, and compressing an
image by keeping only its top principal components — 50 components hold 95% of the
variance and the picture is still clearly readable.

---

## What I got right, and where I'd be honest about the limits

- **The from-scratch versions genuinely match the library.** That was the goal and it's
  the part I'd actually stand behind — k-means hits the same dispersion cost, k-NN the
  same accuracy at every k, the network the same 98%. When they match, I know I
  implemented the algorithm and not just a lookalike.

- **The heart-disease numbers are modest, and that's the real result.** k-NN lands around
  63% and DBSCAN mostly finds one blob with some noise. That isn't a bug — the two
  features I plot genuinely overlap, and heart-disease risk isn't cleanly separable from
  blood pressure and tobacco alone. I left the honest numbers in rather than cherry-pick
  features to make the plot look tidy.

- **My linear SVM trails scikit-learn by a couple of points** (97% vs 99%). That's the
  gap between a hand-rolled sub-gradient loop and a proper QP solver — same idea, less
  polished optimiser. I didn't hand-roll the kernel version: the kernel trick is where
  doing it yourself stops teaching you much and a library call starts being the right
  answer, so I used `SVC` there and said so.

- **Scaling matters and I almost learned that the hard way.** Every distance-based method
  here (k-means, k-NN, DBSCAN) breaks if you skip `StandardScaler`, because one
  large-range feature like `alcohol` quietly dominates the distance. Same rule for PCA —
  it chases variance, so an unscaled feature hijacks the components.

---

## What I left out (scope, stated plainly)

- **No unit tests.** This is a learning/portfolio repo; the "test" is that each algorithm
  reproduces scikit-learn's result, which the scripts print every run.
- **No kernel SVM from scratch.** Covered above.
- **The networks use plain SGD, not Adam or mini-batches.** Enough to make the point and
  match the lab; a production net would use an optimiser and batching (see the
  large-scale-ML section of my course notes).

## Built with
Python · NumPy · scikit-learn (for the side-by-side checks) · pandas · matplotlib ·
no deep-learning framework — the networks are hand-written
