# Setup & run — ELE 888 ML labs

Step-by-step, assuming nothing is installed yet. Everything runs on plain Python with
NumPy/scikit-learn/matplotlib — no GPU, no deep-learning framework.

---

## 1. Get Python (3.10 or newer)

Check what you have:

```bash
python3 --version
```

If that's missing or older than 3.10, install Python 3 from python.org (or `sudo apt
install python3 python3-pip` on Ubuntu/WSL).

---

## 2. Install the dependencies

From the project folder:

```bash
make setup
# which is just:  python3 -m pip install -r requirements.txt
```

That installs `numpy`, `pandas`, `scikit-learn`, and `matplotlib`.

> **Tip:** to keep this isolated from the rest of your system, make a virtual
> environment first:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate        # Windows: .venv\Scripts\activate
> pip install -r requirements.txt
> ```

---

## 3. Run everything

```bash
make run
```

This runs all seven experiments in order (~20 seconds total) and writes the plots into
`figures/`. You'll see the results printed as it goes — accuracies, the from-scratch vs
scikit-learn comparisons, variance retained, and so on.

To run just one group:

```bash
make clustering   # experiments 1-2: k-means, DBSCAN, k-NN
make neuralnet    # experiments 3-4: XOR, then digit recognition
make svm          # experiment 5: SVM vs neural network
make images       # experiment 6: k-means colour quantization
make pca          # experiment 7: PCA visualization + image compression
```

Or a single script directly:

```bash
python3 scripts/03_neural_net_xor.py
```

---

## 4. What each experiment produces

| Script | What it does | Figure |
|---|---|---|
| `01_clustering_synthetic.py` | k-means on the 8-point toy set, centres moving per epoch | `01_kmeans_epochs.png` |
| `02_clustering_heart.py` | k-means + DBSCAN + k-NN on the heart-disease data | `02_heart_clustering.png`, `02_heart_knn.png` |
| `03_neural_net_xor.py` | neural net solving XOR, loss curve + decision regions | `03_xor.png` |
| `04_neural_net_digits.py` | 10-class digit recognition, from-scratch vs sklearn MLP | `04_digits_nn.png` |
| `05_svm_vs_nn.py` | SVM margins + digit 3-vs-8 solved four ways | `05_svm_margin.png` |
| `06_kmeans_image_quantization.py` | reduce an image to 16/8/4 colours with k-means | `06_image_quantization.png` |
| `07_pca_digits_and_image.py` | PCA 2-D projection + image compression | `07_pca_digits_2d.png`, `07_pca_image_compression.png` |

---

## 5. Using the algorithms yourself

Everything under `src/ele888/` is importable. The classes follow scikit-learn's
`fit` / `predict` shape on purpose, so they're easy to swap in:

```python
import sys; sys.path.insert(0, "src")      # or: pip install -e .
from ele888.clustering.kmeans import KMeans
from ele888.neural_net.network import NeuralNetwork
from ele888.decomposition.pca import PCA

labels = KMeans(n_clusters=3).fit_predict(X)
net    = NeuralNetwork([64, 32, 10]).fit(X, Y_onehot)
Z      = PCA(n_components=2).fit_transform(X)
```

`pip install -e .` (uses `pyproject.toml`) installs the package so you can
`from ele888...` from anywhere without the `sys.path` line.

---

## 6. Pushing it to GitHub

```bash
cd ele888-ml-labs
git init
git add .
git commit -m "ELE 888: machine-learning algorithms from scratch"
git branch -M main
git remote add origin https://github.com/munevvercoskun/ele888-ml-labs.git
git push -u origin main
```

(Create the empty `ele888-ml-labs` repo on GitHub first, without a README so there's
nothing to merge.) The `figures/` images are git-ignored because `make run` regenerates
them; if you'd rather show them on the repo page, delete the last line of `.gitignore`
and commit them.

---

## Troubleshooting

- **`ModuleNotFoundError: ele888`** — run the scripts from the project root
  (`python3 scripts/01_...py`), not from inside `scripts/`. The scripts add `src/` to the
  path themselves; running from the wrong folder breaks that.
- **A window doesn't pop up** — that's intended. The scripts save PNGs to `figures/`
  instead of opening windows (they force the non-interactive `Agg` backend), so they work
  the same over SSH/WSL with no display.
- **`heart.csv` missing** — it's bundled in `data/`. If it ever goes missing, it's the
  South African heart dataset from
  `https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv`.
