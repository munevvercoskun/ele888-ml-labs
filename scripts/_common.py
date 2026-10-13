"""Shared setup for the experiment scripts.

Importing this first puts the `src/` folder on the path (so the scripts
run with a plain `python scripts/xx.py`, no install needed) and gives
everyone the same figures/ output directory.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    import matplotlib.pyplot as plt
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")
    return path
