# ELE 888 - Machine Learning labs, from scratch
# ------------------------------------------------
# Common targets. `make setup` once, then `make run`.

PYTHON ?= python3

.PHONY: help setup run clustering neuralnet svm images pca clean

help:
	@echo "make setup      - install the dependencies (numpy, sklearn, matplotlib, pandas)"
	@echo "make run        - run all 7 experiments, write figures to figures/"
	@echo "make clustering - experiments 1-2 (k-means, DBSCAN, k-NN)"
	@echo "make neuralnet  - experiments 3-4 (XOR + digits)"
	@echo "make svm        - experiment 5 (SVM vs neural net)"
	@echo "make images     - experiment 6 (k-means colour quantization)"
	@echo "make pca        - experiment 7 (PCA viz + image compression)"
	@echo "make clean      - delete generated figures"

setup:
	$(PYTHON) -m pip install -r requirements.txt

run:
	MPLBACKEND=Agg $(PYTHON) run_all.py

clustering:
	MPLBACKEND=Agg $(PYTHON) scripts/01_clustering_synthetic.py
	MPLBACKEND=Agg $(PYTHON) scripts/02_clustering_heart.py

neuralnet:
	MPLBACKEND=Agg $(PYTHON) scripts/03_neural_net_xor.py
	MPLBACKEND=Agg $(PYTHON) scripts/04_neural_net_digits.py

svm:
	MPLBACKEND=Agg $(PYTHON) scripts/05_svm_vs_nn.py

images:
	MPLBACKEND=Agg $(PYTHON) scripts/06_kmeans_image_quantization.py

pca:
	MPLBACKEND=Agg $(PYTHON) scripts/07_pca_digits_and_image.py

clean:
	rm -f figures/*.png
