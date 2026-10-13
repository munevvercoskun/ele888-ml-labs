"""Run every ELE 888 experiment in order and write all figures to figures/.

    python run_all.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

EXPERIMENTS = [
    "01_clustering_synthetic",
    "02_clustering_heart",
    "03_neural_net_xor",
    "04_neural_net_digits",
    "05_svm_vs_nn",
    "06_kmeans_image_quantization",
    "07_pca_digits_and_image",
]


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    for name in EXPERIMENTS:
        print("\n" + "=" * 70)
        mod = importlib.import_module(name)
        mod.main()
    print("\nAll experiments finished. See the figures/ folder.")


if __name__ == "__main__":
    main()
