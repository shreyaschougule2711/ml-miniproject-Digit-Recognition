# DigitAI — Ensemble-Based Handwritten Digit & Multi-Digit Number Recognition System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive computer vision and supervised machine learning platform that recognizes both **isolated handwritten digits (0–9)** and **variable-length multi-digit handwritten numbers** (e.g. `12`, `45`, `108`, `256`, `2026`, `12345`). The system features an **Ensemble Voting Classifier** using democratic hard majority voting across three diverse classical learning paradigms:
1. **k-Nearest Neighbors (k-NN)** — non-parametric instance-based spatial voting
2. **Linear Support Vector Machine (Linear SVM)** — calibrated maximum-margin hyperplane separation
3. **Radial Basis Function Support Vector Machine (RBF SVM)** — non-linear mapping into infinite-dimensional Hilbert space

---

## 📌 1. Project Abstract
Optical Character Recognition (OCR) remains a cornerstone problem in computer vision and machine learning. This project implements a rigorous, leak-free, reproducible machine learning system on the landmark **MNIST handwritten digits dataset** (70,000 grayscale images, 28 × 28 pixels). 

The platform benchmarks each individual baseline classifier and implements a **Hard Majority Voting Classifier** combining their predictions. Users can seamlessly choose between any of the three individual baseline classifiers or the unified Ensemble Voting Classifier.

The system features an interactive **Streamlit web application** enabling users to:
- **Draw single digits (0–9)** or **wide multi-digit numbers** (e.g. 2026, 45, 12345) on an interactive digital canvas.
- **Upload handwritten number images** (PNG, JPG, JPEG) with automatic dark/light background polarity adaptation.
- Perform automated **contour detection, connected-component analysis, and vertical projection profile splitting** for touching digits.
- Normalize each segmented digit with **Center-of-Mass alignment** adhering strictly to the official Yann LeCun MNIST normalization protocol.
- Inspect detailed **ensemble voting breakdowns** (unanimous 3/3 consensus, majority 2/3 votes, or split decision tie-breaking) with real-time class score distributions.

---

## 🎯 2. Objectives
- Implement, train, and benchmark **k-NN**, **Linear SVM**, **RBF SVM**, and a **Hard Voting Ensemble**.
- Guarantee **zero test set leakage** by enforcing strict separation between training hyperparameter selection (Stratified 3-Fold Cross-Validation) and untouched test evaluation.
- Provide full flexibility to use individual models as baselines or the Ensemble Voting Classifier.
- Develop an automated multi-digit segmentation and sequencing pipeline to recognize numbers of arbitrary length.
- Support dual input modalities: **interactive digital canvas drawing** and **image file upload**.
- Rigorously evaluate all models using identical held-out test samples across accuracy, precision, recall, macro F1-score, confusion matrices, and latency.
- Deliver a responsive Streamlit interface compatible with Streamlit Community Cloud.

---

## 🗂️ 3. Project Directory Structure
```
d:/ML MINI-PROJECT/
├── app.py                      # Main Streamlit web application (6 Dedicated Studio Views)
├── config.py                   # Centralized paths, dataset sizes, model paths & hyperparameters
├── requirements.txt            # Python dependencies (Streamlit Cloud compatible)
├── packages.txt                # Linux system libraries for Streamlit Community Cloud (libgl1, libglib)
├── pytest.ini                  # Pytest automated test configuration
├── README.md                   # Comprehensive project documentation
├── PROJECT_REPORT.md           # Academic mini-project report
├── VIVA_QUESTIONS.md           # Comprehensive viva examination preparation guide
├── .gitignore                  # Git ignore rules for virtual environments & caches
├── data/                       # Local MNIST cache directory (.npz)
│   └── mnist_784.npz           # Cached 70,000 MNIST dataset samples
├── models/                     # Serialized joblib model artifacts
│   ├── knn_model.joblib
│   ├── linear_svm_model.joblib
│   ├── rbf_svm_model.joblib
│   └── ensemble_voting_model.joblib # Trained hard majority voting ensemble
├── results/                    # Benchmark metrics & publication plots
│   ├── evaluation_metrics.json # Full per-class metrics & confusion matrices
│   ├── model_comparison.csv   # Standardized comparative metrics table
│   ├── multidigit_benchmark.csv# Multi-digit sequence benchmark results
│   ├── training_metadata.json  # Pipeline execution metadata
│   └── plots/                  # Visual confusion matrices & comparison charts
│       ├── confusion_matrix_knn.png
│       ├── confusion_matrix_linear_svm.png
│       ├── confusion_matrix_rbf_svm.png
│       ├── confusion_matrix_ensemble.png
│       ├── model_metrics_comparison.png
│       ├── runtime_comparison.png
│       └── sample_digits_grid.png
├── src/                        # Core modular Python packages
│   ├── __init__.py
│   ├── data_loader.py          # MNIST fetcher, caching & stratified splitters
│   ├── preprocessing.py        # Aspect ratio & center-of-mass preprocessing
│   ├── ensemble.py             # EnsembleVotingClassifier implementation & majority voting
│   ├── multi_digit.py          # Multi-digit contour segmentation & prediction engine
│   ├── train.py                # Hyperparameter tuning, CV & model training pipeline
│   ├── evaluate.py             # Accuracy, F1, and latency evaluation engine
│   ├── evaluate_ensemble.py    # Zero-retrain evaluation & plot generation script
│   ├── evaluate_multidigit.py  # Multi-digit sequence benchmark evaluator
│   ├── predict.py              # Real-time multi-model & ensemble inference engine
│   └── visualization.py        # Heatmaps, confusion matrices & bar charts
└── tests/                      # Automated test suite (21 passing unit tests)
    ├── __init__.py
    ├── test_preprocessing.py   # Unit tests for preprocessing & canvas normalization
    ├── test_prediction.py      # Unit tests for model inference & fallback logic
    ├── test_multi_digit.py     # Unit tests for multi-digit segmentation & ordering
    └── test_ensemble.py        # Unit tests for ensemble voting, tie-breaking & consensus
```

---

## ⚙️ 4. System Architecture & Methodology

```
+--------------------------------------------------------------------------+
|                            DATA INGESTION                                |
|   MNIST (70,000 samples) -> Stratified Train (10,000) / Test (2,000)     |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                          DATA PREPROCESSING                              |
|  Pixel Normalization [0, 1] + 784-Dimensional Flattening (float32)       |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                  BASELINE MODEL TRAINING & CV TUNING                     |
|  - 3-Fold Stratified Cross-Validation on Train Split ONLY               |
|  - k-NN: k ∈ {3, 5, 7}, weights ∈ {distance, uniform}, metric ∈ {L1, L2} |
|  - Linear SVM: C ∈ {0.01, 0.1, 1.0, 5.0} with CalibratedClassifierCV     |
|  - RBF SVM: C ∈ {1.0, 5.0, 10.0}, γ ∈ {'scale', 0.01}                    |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                    ENSEMBLE VOTING CLASSIFIER                            |
|  - Hard Majority Voting across k-NN, Linear SVM, and RBF SVM             |
|  - Majority Decision: ŷ = mode(y_knn, y_linear, y_rbf)                   |
|  - Deterministic Tie-Breaker: RBF SVM (highest individual baseline acc)  |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                     HELD-OUT TEST EVALUATION                             |
|  - Same untouched 2,000 test samples (zero data leakage)                 |
|  - Accuracy, Precision, Recall, Macro F1, Latency, Confusion Matrices    |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                 DIGITAI STUDIO STREAMLIT APPLICATION                     |
|  - Single & Multi-Digit Recognition (Canvas Drawing & Image Upload)      |
|  - Model Selector: Ensemble Voting, RBF SVM, k-NN, Linear SVM            |
|  - Live Voting Consensus Badge (Unanimous 3/3, Majority 2/3, 1/1/1 Tie)  |
|  - Cross-Classifier Consensus Comparison Tables                          |
+--------------------------------------------------------------------------+
```

### Ensemble Voting Decision Rule
Each classifier $h_m(\mathbf{x}) \in \{0, \dots, 9\}$ outputs its discrete class prediction. For $M=3$ classifiers:
$$\hat{y}_{\text{ensemble}} = \arg\max_{c \in \{0,\dots,9\}} \sum_{m=1}^{M} \mathbb{I}(h_m(\mathbf{x}) = c)$$
- **Case 1: Unanimous Agreement (3/3 votes)**: All 3 models agree on class $c$. Confidence = 100%.
- **Case 2: Majority Agreement (2/3 votes)**: Exactly 2 models agree on class $c$, while 1 model votes differently. Confidence = 66.7%.
- **Case 3: Split Decision (1/1/1 three-way tie)**: All 3 models vote for distinct classes. The tie is deterministically broken by RBF SVM, which exhibits the highest baseline accuracy on the test set.

---

## 📊 5. Measured Evaluation Results
All models and the ensemble were evaluated on the **exact same 2,000 held-out test samples** drawn from the official MNIST test partition. No test data was exposed during hyperparameter tuning.

| Classifier Model | Model Key | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Training Duration | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF Kernel SVM** | `rbf_svm` | **96.55%** | **96.57%** | **96.51%** | **96.53%** | 179.80 s | **4.03 ms** |
| **Ensemble Voting Classifier** | `ensemble` | **96.05%** | **96.07%** | **95.98%** | **96.01%** | 187.71 s | 46.21 ms |
| **k-Nearest Neighbors (k=3)** | `knn` | **94.60%** | 94.78% | 94.56% | 94.61% | **0.021 s** | 32.80 ms |
| **Linear SVM (Calibrated)** | `linear_svm` | **90.55%** | 90.58% | 90.36% | 90.39% | 7.888 s | 10.79 ms |

*(Benchmark configuration: 10,000 training samples with 3-Fold Stratified Cross-Validation; 2,000 untouched test samples; random_state=42. Measured on CPU without data leakage.)*

### Honest Scientific Findings on the Ensemble Performance:
1. **High Overall Accuracy (96.05%)**: The Ensemble Voting Classifier outperforms both individual baselines **k-NN (94.60%)** and **Linear SVM (90.55%)** by a wide margin.
2. **Why RBF SVM (96.55%) slightly exceeds the unweighted hard ensemble (96.05%)**:
   - Condorcet's Jury Theorem assumes base classifiers commit *independent* errors.
   - On challenging handwritten samples with complex nonlinear curves (e.g. ambiguous loops in 3 vs 5, or 4 vs 9), Linear SVM (90.55%) and k-NN (94.60%) occasionally share correlated linear-boundary misclassifications.
   - When both Linear SVM and k-NN agree on an incorrect label, their combined 2 votes outvote the correct prediction of RBF SVM.
3. **Primary Advantages of the Ensemble**:
   - **Variance Reduction & Reliability**: Reduces model-specific idiosyncrasies and eliminates single-algorithm failure modes.
   - **Diagnostic Transparency**: Surfaces multi-model consensus (3/3 vs 2/3 agreement), giving users actionable insight into prediction certainty.

---

## 💻 6. Requirements

### Software
- **Python**: 3.11 or 3.12 (64-bit recommended)
- **Dependencies** (`requirements.txt`):
  - `streamlit>=1.35.0`
  - `streamlit-drawable-canvas>=0.9.3`
  - `numpy>=1.26.0,<3.0.0`
  - `scipy>=1.11.0,<2.0.0`
  - `pandas>=2.1.0,<4.0.0`
  - `scikit-learn>=1.4.0,<2.0.0`
  - `matplotlib>=3.8.0,<4.0.0`
  - `seaborn>=0.13.0,<1.0.0`
  - `Pillow>=10.0.0,<13.0.0`
  - `joblib>=1.3.0,<2.0.0`
  - `opencv-python-headless>=4.8.0`
  - `pytest>=8.0.0`
- **Streamlit Cloud System Packages** (`packages.txt`): `libgl1`, `libglib2.0-0`

### Hardware
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher (runs entirely on CPU)
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: ~500 MB for dataset cache and model artifacts

---

## 🚀 7. Installation & Quick Start

### Step 1: Navigate to Project Directory
```bash
cd "d:/ML MINI-PROJECT"
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the Automated Test Suite (21 Unit Tests)
```bash
pytest
```
Expected output: `21 passed in ~6s`.

### Step 4: (Optional) Re-evaluate Models or Train
- To re-evaluate all models (including ensemble) on the held-out test split without re-training:
```bash
python src/evaluate_ensemble.py
```
- To run full cross-validation training from scratch:
```bash
python src/train.py --train-samples 10000 --test-samples 2000
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🔢 8. Multi-Digit Recognition Workflow
Single-digit models cannot parse multi-digit sequences directly. The application incorporates a specialized computer vision pipeline:
1. **Otsu Binarization & Polarity Normalization**: Automatically converts input into standard white-digit-on-black-background format.
2. **Contour Extraction**: Identifies candidate digit regions and discards specks (area < 40 px).
3. **Multi-Stroke Merging**: Heuristically merges disconnected strokes belonging to the same digit (e.g. crossbar of 4, top bar of 5, cross of 7).
4. **Touching Digit Splitting**: For boxes with aspect ratio $> 1.25$, vertical projection profiling identifies the deepest valley to bifurcate joined digits.
5. **Left-to-Right Sequencing & LeCun Centering**: Detected digits are sorted left-to-right, normalized into $28 \times 28$ center-of-mass grids, and classified independently via the selected model or ensemble.

---

## 🌐 10. Live Application Deployment
The interactive web application is deployed on Streamlit Community Cloud:
- **Live Streamlit App:** [https://ml-miniproject-digit-recognition-s7tzhwrftvm3tkuhnsf8tm.streamlit.app/](https://ml-miniproject-digit-recognition-s7tzhwrftvm3tkuhnsf8tm.streamlit.app/)

---

<<<<<<< HEAD
## 📄 11. License
This project is open-source under the [MIT License](LICENSE).
=======
## ⚠️ 9. Limitations & Future Scope
- **Current Limitations**: Raw pixel features lack translation and rotation invariance; severely tilted or rotated digits may reduce recognition accuracy.
- **Future Scope**:
  1. Feature extraction using HOG (Histogram of Oriented Gradients) or PCA dimensionality reduction.
  2. Integration of Convolutional Neural Networks (CNNs) with data augmentation (random rotation, elastic distortions).
  3. Continuous multi-digit recognition using sliding window segmentation.
 
     ## 10. Project link
     https://ml-miniproject-digit-recognition-s7tzhwrftvm3tkuhnsf8tm.streamlit.app/
>>>>>>> 2d83dff2a98730d8d8750ef5133562cc2cdeb332
