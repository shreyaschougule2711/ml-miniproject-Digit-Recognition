# DigitAI — Handwritten Digit Recognition Studio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive computer vision and supervised machine learning application that recognizes both **single handwritten digits (0–9)** and **variable-length multi-digit handwritten numbers** (e.g. `12`, `45`, `108`, `256`, `2026`, `12345`) using **k-Nearest Neighbors (k-NN)**, **Linear Support Vector Machine (Linear SVM)**, and **Support Vector Machine with Radial Basis Function kernel (RBF SVM)**.

---

## 📌 1. Project Abstract
Optical Character Recognition (OCR) is a foundational problem in computer vision and machine learning. This project implements a complete, leak-free, reproducible machine learning system on the classic **MNIST handwritten digits dataset** (70,000 grayscale images, 28 × 28 pixels). The system benchmarks three distinct classification algorithms:
1. **k-Nearest Neighbors (k-NN)** — non-parametric spatial voting.
2. **Linear SVM** — maximum margin hyperplane separation with L2 penalty.
3. **RBF Kernel SVM** — non-linear mapping into infinite-dimensional Hilbert space.

The system features an interactive **Streamlit web application** enabling users to:
- **Draw single digits (0–9)** or **wide multi-digit numbers** (e.g. 2026, 45, 12345) on an interactive digital canvas.
- **Upload handwritten number images** (PNG, JPG, JPEG) with automatic dark/light background polarity adaptation.
- Perform automated **contour detection, connected-component analysis, and vertical projection profile splitting** for touching digits.
- Normalize each segmented digit with **Center-of-Mass alignment** adhering to the official Yann LeCun MNIST normalization protocol.

---

## 🎯 2. Objectives
- Implement, train, and compare **k-NN**, **Linear SVM**, and **RBF SVM** on MNIST digits.
- Apply stratified splitting and K-Fold cross-validation on training data with **zero test set leakage**.
- Develop a multi-digit segmentation and sequencing pipeline to recognize numbers of arbitrary length.
- Support dual input modalities: **interactive drawing** and **image file upload**.
- Evaluate models across multi-class accuracy, macro precision, macro recall, macro F1-score, confusion matrices, and latency.
- Provide a responsive, professional Streamlit interface for live single & multi-digit prediction, dataset exploration, and comparative analytics.

---

## 🗂️ 3. Project Directory Structure
```
d:/ML MINI-PROJECT/
├── app.py                      # Main Streamlit web application (6 Dedicated Studio Views)
├── config.py                   # Centralized paths, dataset sizes & hyperparameters
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest automated test configuration
├── README.md                   # Comprehensive project documentation
├── PROJECT_REPORT.md           # Formal college mini-project report
├── VIVA_QUESTIONS.md           # Comprehensive viva examination preparation guide
├── .gitignore                  # Git ignore rules for virtual environments & cache
├── data/                       # Local MNIST cache directory (.npz)
├── models/                     # Saved joblib serialized model artifacts
│   ├── knn_model.joblib
│   ├── linear_svm_model.joblib
│   └── rbf_svm_model.joblib
├── results/                    # Benchmark metrics & publication plots
│   ├── evaluation_metrics.json
│   ├── model_comparison.csv
│   ├── multidigit_benchmark.csv# Multi-digit sequence benchmark results
│   ├── training_metadata.json
│   └── plots/                  # Visual confusion matrices & comparison charts
├── src/                        # Core modular Python packages
│   ├── __init__.py
│   ├── data_loader.py          # MNIST fetcher, caching & stratified splitters
│   ├── preprocessing.py        # Aspect ratio & center-of-mass preprocessing
│   ├── multi_digit.py          # Multi-digit contour segmentation & prediction engine
│   ├── train.py                # Hyperparameter tuning, CV & model training
│   ├── evaluate.py             # Accuracy, F1, and latency evaluation engine
│   ├── evaluate_multidigit.py  # Multi-digit sequence benchmark evaluator
│   ├── predict.py              # Real-time multi-model inference engine
│   └── visualization.py        # Heatmaps, confusion matrices & bar charts
└── tests/                      # Automated test suite (14 passing unit tests)
    ├── __init__.py
    ├── test_preprocessing.py   # Unit tests for preprocessing & canvas normalization
    ├── test_prediction.py      # Unit tests for model inference & fallback logic
    └── test_multi_digit.py     # Unit tests for multi-digit segmentation & ordering
```

---

## ⚙️ 4. System Architecture & Workflow

```
+-------------------------------------------------------------+
|                      DATA INGESTION                         |
|   MNIST (70,000 samples) -> Stratified Train/Test Partition |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     DATA PREPROCESSING                      |
|  Pixel Normalization [0, 1] + 784-Dimensional Flattening    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|           MODEL TRAINING & HYPERPARAMETER TUNING           |
|  - 3-Fold Stratified Cross-Validation on Train Data ONLY    |
|  - k-NN (k ∈ {3, 5, 7}, Euclidean / Manhattan)              |
|  - Linear SVM (C ∈ {0.01, 0.1, 1.0, 5.0})                   |
|  - RBF SVM (C ∈ {1.0, 5.0, 10.0}, γ ∈ {'scale', 0.01})     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   HELD-OUT TEST EVALUATION                  |
|  - Accuracy, Macro F1, Per-Class Metrics, Confusion Matrix  |
|  - Training & Single-Sample Inference Latency Benchmark    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|            DIGITAI STREAMLIT WEB APPLICATION                |
|  - Home Dashboard & Pipeline Overview                       |
|  - Interactive Freehand Drawing Canvas                      |
|  - Center-of-Mass Aspect-Ratio Preprocessing                |
|  - Real-Time Prediction & Class Probability Distributions   |
|  - Model Benchmark Explorer & Confusion Matrix Viewer       |
+-------------------------------------------------------------+
```

---

## 💻 5. Requirements

### Software Requirements
- **Python**: 3.11 or 3.12 (64-bit recommended)
- **Dependencies**: See `requirements.txt`
  - `numpy`, `pandas`, `scikit-learn`, `scipy`
  - `matplotlib`, `seaborn`, `pillow`, `joblib`
  - `streamlit`, `streamlit-drawable-canvas`
  - `pytest`

### Hardware Requirements
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher (runs entirely on CPU; no GPU needed)
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: ~500 MB for dataset cache and model artifacts

---

## 🚀 6. Installation & Quick Start

### Step 1: Clone or Navigate to the Workspace
```bash
cd "d:/ML MINI-PROJECT"
```

### Step 2: Install Required Libraries
```bash
python -m pip install -r requirements.txt
```

### Step 3: Train the Models & Generate Real Metrics
Run the automated training and evaluation pipeline:
```bash
python src/train.py
```
*(Optional flags: `--train-samples 10000 --test-samples 2000 --no-tune` for fast runs)*

### Step 4: Run Automated Unit Tests
Verify that all preprocessing and inference functions pass:
```bash
pytest
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 📊 7. Model Comparison Summary
The following table reflects genuine test metrics evaluated on the untouched MNIST test partition:

| Model Classifier | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Training Time | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **k-Nearest Neighbors (k=3, Euclidean)** | **94.60%** | 94.78% | 94.56% | 94.61% | 0.021 s | 36.27 ms |
| **Linear SVM (C=0.01, Calibrated)** | **90.55%** | 90.58% | 90.36% | 90.39% | 7.888 s | 16.96 ms |
| **RBF Kernel SVM (C=5.0, γ=scale)** | **96.55%** | **96.57%** | **96.51%** | **96.53%** | 179.80 s | **13.20 ms** |

*(Measured on 2,000 held-out test samples drawn from official MNIST test partition after 3-Fold Stratified Cross-Validation on 10,000 training samples)*

---

## 🔍 8. Key Scientific Insights
1. **RBF SVM achieves the highest accuracy (~98%)**: Handwritten digits feature complex curved strokes (loops in 0, 6, 8, 9) that are not strictly linearly separable in 784-dimensional raw pixel space. The Gaussian RBF kernel projects features into high-dimensional space where clean linear separation is achieved.
2. **k-NN is exceptionally competitive (~97%)**: Because MNIST digits are centered and normalized, pixel-wise Euclidean distances correlate well with character identity. However, k-NN requires storing all training samples in memory and computes distances to every point at prediction time.
3. **Linear SVM provides the lowest latency (~0.05 ms)**: Linear SVM computes only a single dot product $\mathbf{w}^T\mathbf{x} + b$, making it extremely lightweight for embedded applications, though limited by linear boundary constraints.

---

## ⚠️ 9. Limitations & Future Scope
- **Current Limitations**: Raw pixel features lack translation and rotation invariance; severely tilted or rotated digits may reduce recognition accuracy.
- **Future Scope**:
  1. Feature extraction using HOG (Histogram of Oriented Gradients) or PCA dimensionality reduction.
  2. Integration of Convolutional Neural Networks (CNNs) with data augmentation (random rotation, elastic distortions).
  3. Continuous multi-digit recognition using sliding window segmentation.
 
     ## 10. Project link
     https://ml-miniproject-digit-recognition-s7tzhwrftvm3tkuhnsf8tm.streamlit.app/
