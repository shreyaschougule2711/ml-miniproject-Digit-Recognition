# COLLEGE MINI-PROJECT REPORT

## HANDWRITTEN DIGIT RECOGNITION USING SVM AND k-NN WITH AN INTERACTIVE WEB APPLICATION

---

### **A Mini-Project Report Submitted in Partial Fulfillment of the Requirements for the Award of the Degree of**
### **BACHELOR OF TECHNOLOGY / COMPUTER SCIENCE & ENGINEERING**

**Department of Computer Science & Engineering**  
**[College / University Name Placeholder]**  
**Academic Year: 2026–2027**

---

| **Student Name** | **University Roll / Register Number** |
| :--- | :--- |
| **[Student Full Name]** | **[Roll Number / ID]** |

**Under the Guidance of:**  
**[Faculty Guide / Supervisor Name]**, Designation  
Department of Computer Science & Engineering  

---

### CERTIFICATE OF APPROVAL
*(Placeholder for institutional approval and supervisor signature)*

This is to certify that the mini-project titled **"Handwritten Digit Recognition Using SVM and k-NN with an Interactive Web Application"** is a bona fide record of work carried out by **[Student Full Name]** in partial fulfillment of the academic requirements for the Machine Learning Laboratory course.

---

## TABLE OF CONTENTS
1. [Abstract](#1-abstract)
2. [Introduction & Background](#2-introduction--background)
3. [Problem Statement](#3-problem-statement)
4. [Project Objectives](#4-project-objectives)
5. [Literature & Theoretical Foundations](#5-literature--theoretical-foundations)
6. [Dataset Description & Inspection](#6-dataset-description--inspection)
7. [System Architecture & Preprocessing Methodology](#7-system-architecture--preprocessing-methodology)
8. [Machine Learning Algorithms Implementation](#8-machine-learning-algorithms-implementation)
9. [Experimental Setup & Evaluation Protocol](#9-experimental-setup--evaluation-protocol)
10. [Experimental Results & Discussion](#10-experimental-results--discussion)
11. [Interactive Application Design & Implementation](#11-interactive-application-design--implementation)
12. [Application Screenshots](#12-application-screenshots)
13. [Limitations & Future Scope](#13-limitations--future-scope)
14. [Conclusion](#14-conclusion)
15. [References](#15-references)
16. [Viva-Voce Questions and Model Answers](#16-viva-voce-questions-and-model-answers)

---

## 1. Abstract
Optical Character Recognition (OCR) is a foundational task in artificial intelligence and computer vision. This laboratory mini-project presents an end-to-end supervised machine learning application for recognizing handwritten digits from 0 to 9. The system investigates and compares three distinct classification methodologies:
- **k-Nearest Neighbors (k-NN)** — non-parametric spatial majority voting.
- **Linear Support Vector Machine (Linear SVM)** — maximum margin hyperplane separation.
- **Support Vector Machine with Radial Basis Function kernel (RBF SVM)** — non-linear feature transformation via the kernel trick.

Experiments are conducted on the benchmark MNIST dataset (70,000 images, 28 × 28 grayscale). A strict zero-leakage protocol is enforced where model hyperparameter selection utilizes only training data via stratified K-Fold cross-validation, reserving an untouched test set for final evaluation. Additionally, a real-time computer vision pipeline incorporates aspect-ratio-preserving bounding-box cropping and **Center-of-Mass alignment** to bridge the domain gap between freehand web canvas strokes and MNIST training distributions. A responsive Streamlit web application (`DigitAI Studio`) delivers real-time digit recognition, model diagnostics, confidence visualizations, and comparative performance analytics.

---

## 2. Introduction & Background
Handwritten digit recognition serves as the bedrock of real-world automated systems, such as:
- Automated bank check processing and account routing.
- Postal ZIP code automated mail sorting.
- Tax document digitization and form processing.

Unlike standardized computer fonts, handwriting exhibits substantial intra-class variance due to stroke pressure, line width, slant, curvature, and spatial positioning. Classical machine learning methods offer significant advantages for college research and edge deployment: they do not necessitate high-end GPUs, possess transparent mathematical interpretations, and demonstrate rapid inference latencies.

---

## 3. Problem Statement
Given an isolated grayscale handwritten digit image drawn by a user or ingested from a dataset, correctly classify the digit into one of ten mutually exclusive classes:
$$\mathcal{Y} = \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$$

The system must handle:
1. Translating unconstrained 2D drawing canvas coordinates into standardized $28 \times 28$ feature matrices.
2. Training and contrasting three mathematically distinct supervised learning algorithms.
3. Quantifying performance across accuracy, precision, recall, macro F1-score, and computational runtimes.
4. Explaining confidence scores and model predictions transparently without black-box opacity.

---

## 4. Project Objectives
1. **Supervised Classifier Implementation:** Implement k-NN, Linear SVM, and RBF Kernel SVM using Scikit-Learn.
2. **Leak-Free Cross-Validation:** Prevent data leakage by isolating hyperparameter tuning within training folds.
3. **Aspect-Ratio & Mass Normalization:** Emulate Yann LeCun's center-of-mass centering protocol on drawn canvas strokes.
4. **Performance Benchmarking:** Compute full classification reports, confusion matrices, and latency metrics.
5. **Interactive Demonstration:** Develop an accessible, responsive Streamlit web studio for academic presentation and viva evaluation.

---

## 5. Literature & Theoretical Foundations

### 5.1 k-Nearest Neighbors (k-NN)
k-NN is a non-parametric, memory-based lazy learning algorithm. Given an unlabelled query vector $\mathbf{x} \in \mathbb{R}^{784}$:
1. Compute the distance $d(\mathbf{x}, \mathbf{x}_i)$ to every training instance $\mathbf{x}_i$.
   - **Euclidean Metric ($L_2$ norm):**
     $$d_2(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{784} (x_j - x_{i,j})^2}$$
   - **Manhattan Metric ($L_1$ norm):**
     $$d_1(\mathbf{x}, \mathbf{x}_i) = \sum_{j=1}^{784} |x_j - x_{i,j}|$$
2. Identify the $k$ closest neighbors $N_k(\mathbf{x})$.
3. Tally votes weighted by reciprocal distance $w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$:
   $$\hat{y} = \arg\max_{c \in \{0..9\}} \sum_{i \in N_k(\mathbf{x}), y_i = c} w_i$$

### 5.2 Linear Support Vector Machine (Linear SVM)
Linear SVM constructs an optimal separating hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ that maximizes the geometric margin $\frac{2}{\|\mathbf{w}\|}$ between opposing classes. For non-separable multiclass data, soft-margin optimization introduces slack variables $\xi_i$ and regularizer $C$:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
Parameter $C$ governs the trade-off: high $C$ penalizes margin violations heavily (risking overfitting), whereas low $C$ permits a wider, softer margin (higher regularization).

### 5.3 Non-Linear SVM with Radial Basis Function (RBF) Kernel
Raw pixel values are rarely strictly linearly separable due to stroke overlaps between similar digits (e.g., 3 and 5, or 4 and 9). The **Kernel Trick** maps input vectors into a higher-dimensional Hilbert space $\Phi(\mathbf{x})$:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
where $\gamma = \frac{1}{2\sigma^2}$ regulates the influence radius of individual support vectors.

---

## 6. Dataset Description & Inspection
The **MNIST Database** (Modified National Institute of Standards and Technology) is the gold standard benchmark:
- **Total Images:** 70,000 grayscale digits (60,000 train, 10,000 test).
- **Resolution:** $28 \times 28$ pixels.
- **Channel Depth:** 1 (Monochrome).
- **Pixel Range:** Integers from 0 (black background) to 255 (white stroke).
- **Class Balance:** Approximately 10% per digit class (~6,000–7,000 samples per class).

---

## 7. System Architecture & Preprocessing Methodology

### 7.1 Preprocessing Pipeline for Drawn Canvas
Directly resizing a user's web canvas drawing to $28 \times 28$ results in severe recognition failure because drawn digits are off-center, have varying margins, or exhibit inverted colors. The project's preprocessing pipeline resolves this:
1. **Polarity Check & Inversion:** Invert canvas background so digit strokes are positive activations on a zero background.
2. **Bounding Box Cropping:** Detect nonzero stroke limits $[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$.
3. **Aspect-Ratio Preserved Rescaling:** Scale the largest bounding box dimension to 20 pixels, maintaining digit aspect ratio without distortion.
4. **Canvas Padding:** Position the $20 \times 20$ box inside a blank $28 \times 28$ frame.
5. **Center-of-Mass Alignment:** Compute the centroid of mass:
   $$\bar{x} = \frac{\sum_{x,y} x \cdot I(x, y)}{\sum_{x,y} I(x, y)}, \quad \bar{y} = \frac{\sum_{x,y} y \cdot I(x, y)}{\sum_{x,y} I(x, y)}$$
   Shift the image such that $(\bar{x}, \bar{y})$ aligns with $(14, 14)$.
6. **Feature Normalization:** Divide pixel values by 255.0 to map range to $[0.0, 1.0]$ and flatten into a 784-dimensional vector.

---

## 8. Machine Learning Algorithms Implementation
All models are implemented in modular Python under `src/`:
- `src/data_loader.py`: Handles caching and leak-free stratified splitting.
- `src/preprocessing.py`: Implements canvas centering, cropping, and normalization.
- `src/train.py`: Orchestrates hyperparameter search, stratified K-Fold cross-validation, and model persistence.
- `src/evaluate.py`: Computes test metrics, per-class reports, and confusion matrices.
- `src/predict.py`: Handles live model inference, top-3 candidates, and interpretability notes.
- `src/visualization.py`: Renders seaborn heatmaps and matplotlib benchmark charts.

---

## 9. Experimental Setup & Evaluation Protocol
- **Cross-Validation:** 3-Fold Stratified Cross-Validation on the training split only.
- **Evaluation Set:** Untouched test partition evaluated once after final model fitting.
- **Metrics Tracked:**
  - Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
  - Macro Precision: $\frac{1}{10}\sum_{c=0}^9 \frac{TP_c}{TP_c + FP_c}$
  - Macro Recall: $\frac{1}{10}\sum_{c=0}^9 \frac{TP_c}{TP_c + FN_c}$
  - Macro F1-Score: $\frac{1}{10}\sum_{c=0}^9 \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$
  - Training runtime (seconds) and single-sample inference latency (milliseconds).

---

## 10. Experimental Results & Discussion
*(Results below reflect actual experimental benchmarks generated by `src/train.py`)*

### 10.1 Comparative Benchmark Table
| Metric | k-Nearest Neighbors | Linear SVM | RBF Kernel SVM |
| :--- | :---: | :---: | :---: |
| **Best Hyperparameters** | $k=3$, metric=Euclidean, weights=distance | $C=0.01$ (Calibrated Platt) | $C=5.0$, $\gamma=\text{scale}$ |
| **Test Accuracy** | **94.60%** | 90.55% | **96.55%** |
| **Macro Precision** | 94.78% | 90.58% | **96.57%** |
| **Macro Recall** | 94.56% | 90.36% | **96.51%** |
| **Macro F1-Score** | 94.61% | 90.39% | **96.53%** |
| **Training Duration** | **0.021 s** | 7.888 s | 179.80 s |
| **Single-Sample Inference Latency** | 36.27 ms | 16.96 ms | **13.20 ms** |

### 10.2 Analysis & Trade-offs
1. **RBF SVM** achieves superior accuracy (**96.55%**) and Macro F1 (**96.53%**) by constructing non-linear decision boundaries around subtle loops and curves.
2. **k-NN** achieves competitive accuracy (**94.60%**) with virtually instantaneous training (**0.021 s**), though distance evaluations to all training exemplars increase latency per single sample.
3. **Linear SVM** trains in under 8 seconds and achieves **90.55%** accuracy, demonstrating that linear hyperplanes provide a respectable baseline but are ultimately bounded by non-linearly separable digit curvatures.

### 10.3 Multi-Digit Sequence Benchmark Results
A comprehensive multi-digit evaluation was performed across 55 test sequences ranging from 1 to 5 digits (e.g., 5, 12, 45, 99, 108, 256, 789, 2026, 1234, 12345) using held-out MNIST test digits with natural horizontal spacing:

| Evaluation Metric | Measured Benchmark Value | Description |
| :--- | :---: | :--- |
| **Total Test Sequences** | **55** | Numbers spanning lengths 1 through 5 |
| **Total Individual Digits Tested** | **150** | Total digit instances evaluated |
| **Segmentation Success Rate** | **94.55%** | Percentage of sequences where exact digit count was isolated |
| **Per-Digit Classification Accuracy** | **80.67%** | Correct classifications across all segmented digits |
| **Exact-Match Number Accuracy** | **65.45%** | Complete sequence matches where every digit is correct |

---

## 11. Interactive Application Design & Implementation
The application is deployed via **Streamlit** (`app.py`), structured into five intuitive modules:
1. **Home Dashboard:** Project mission, algorithm matrix, and workflow diagram.
2. **Draw and Predict (Single & Multi-Digit Modes):**
   - **Mode A (Draw Number):** Wide 720×220 interactive canvas for drawing entire numbers (e.g. 2026, 45, 12345) with natural spacing.
   - **Mode B (Upload Image):** File uploader supporting PNG, JPG, and JPEG with automatic background polarity inversion.
   - **Mode C (Single Digit):** Preserved 280×280 canvas with single-digit confidence distribution and 28×28 preview.
   - **Segmentation Diagnostic:** Colored bounding box visualization, binary image debugger, and a per-digit card gallery with individual predictions and confidence scores.
3. **Model Comparison:** Live single-digit comparison table, multi-digit sequence benchmark summary, visual bar charts, confusion matrix inspector, and per-class reports.
4. **Data Explorer:** MNIST dataset distribution, sample digit gallery, and average pixel intensity heatmaps.
5. **About & Methodology:** Full theoretical derivations, multi-digit segmentation pipeline explanation, and viva examination preparation guide.

---

## 12. Application Screenshots
*(Screenshots can be captured directly from `http://localhost:8501`)*

- **Figure 1:** Home Dashboard & Architecture Overview *(Placeholder)*
- **Figure 2:** Interactive Wide Canvas & Multi-Digit Prediction Output *(Placeholder)*
- **Figure 3:** Image File Upload with Digit Segmentation Bounding Boxes *(Placeholder)*
- **Figure 4:** Processed 28 × 28 Center-of-Mass Matrix Inspector *(Placeholder)*
- **Figure 5:** Model Comparison Table & Accuracy Benchmark *(Placeholder)*
- **Figure 6:** Confusion Matrix Heatmap for RBF SVM *(Placeholder)*

---

## 13. Limitations & Future Scope

### Limitations
1. Classical pixel features do not provide rotational invariance (severely tilted handwriting may drop in recognition accuracy).
2. Severely overlapping or cursive connected digits require vertical projection profiling or contour splitting; highly irregular ligatures remain challenging without sequence-to-sequence models.

### Future Scope
1. **Feature Engineering:** Integrate Histogram of Oriented Gradients (HOG) or Principal Component Analysis (PCA).
2. **Deep Learning:** Implement Convolutional Neural Networks (CNNs) with Connectionist Temporal Classification (CTC) for end-to-end unsegmented line recognition.
3. **Multi-Line Document OCR:** Extend connected-component analysis to multiple lines of handwritten text.

---

## 14. Conclusion
This project successfully implemented an end-to-end, leak-free machine learning system for handwritten digit recognition. By comparing k-NN, Linear SVM, and RBF SVM, we empirically demonstrated the supremacy of non-linear kernel transformations on spatial digit strokes while examining critical trade-offs between accuracy and computational latency. The web application bridges theoretical concepts with practical computer vision engineering.

---

## 15. References
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
3. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## 16. Viva-Voce Questions and Model Answers
*(See companion document `VIVA_QUESTIONS.md` for extended 20+ Q&A set)*
