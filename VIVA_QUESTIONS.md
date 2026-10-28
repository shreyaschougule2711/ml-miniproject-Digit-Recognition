# VIVA-VOCE EXAMINATION PREPARATION GUIDE
## Handwritten Digit Recognition Using SVM and k-NN with an Interactive Web Application

This document provides clear, concise, and academically rigorous answers to potential questions asked by external and internal examiners during your Machine Learning mini-project viva-voce.

---

### Category 1: Fundamental Concepts & Problem Formulation

#### Q1: What is the primary objective of your mini-project?
**Answer:**  
The objective is to design, implement, and evaluate an interactive computer vision application that recognizes handwritten digits from 0 to 9 using classical machine learning algorithms. The project systematically compares three distinct classifiers—k-Nearest Neighbors (k-NN), Linear Support Vector Machine (Linear SVM), and Radial Basis Function Kernel SVM (RBF SVM)—on the standard MNIST dataset, while ensuring zero data leakage and proper center-of-mass image preprocessing.

#### Q2: What is the MNIST dataset? Describe its characteristics.
**Answer:**  
MNIST (Modified National Institute of Standards and Technology) is a landmark benchmark in machine learning:
- **Total Images:** 70,000 grayscale images (60,000 standard training, 10,000 standard testing).
- **Dimensions:** $28 \times 28$ pixels, yielding 784 numerical features when flattened.
- **Color Depth:** 8-bit grayscale (0 = black background, 255 = white stroke).
- **Classes:** 10 balanced digit classes (0 through 9).

#### Q3: Why is handwritten digit recognition a challenging classification task?
**Answer:**  
Because of high intra-class variability and subtle inter-class differences. Different writers write with distinct slants, stroke thicknesses, curvatures, and speeds. Moreover, certain pairs of digits (such as 4 and 9, 3 and 5, or 1 and 7) have very similar stroke arrangements, requiring classifiers capable of learning subtle discriminative boundaries.

---

### Category 2: Image Preprocessing & Computer Vision

#### Q4: Why is Center-of-Mass (CoM) alignment so critical for user-drawn digits?
**Answer:**  
In the original MNIST dataset, Yann LeCun and colleagues normalized each digit by placing it inside a $20 \times 20$ box and centering its pixel intensity Center of Mass $(\bar{x}, \bar{y})$ exactly at $(14, 14)$ inside the $28 \times 28$ frame. If a user draws a digit off-center on a web canvas and we merely resize it without centering by mass, the pixel activations will align with the wrong feature indices in the 784-dimensional space, causing prediction failure. Centering by mass ensures spatial congruence with MNIST.

#### Q5: Explain the complete preprocessing pipeline applied to the drawn canvas.
**Answer:**  
1. **Luminance & Polarity Correction:** Invert the canvas if the user draws with black ink on white background so digits are positive values on a zero background.
2. **Bounding Box Isolation:** Detect coordinates of non-zero stroke pixels to crop away empty borders.
3. **Aspect-Ratio Preserved Rescaling:** Scale the largest bounding box dimension to 20 pixels, maintaining digit aspect ratio without stretching or distortion.
4. **Canvas Embedding:** Position the $20 \times 20$ digit into the center of a blank $28 \times 28$ canvas.
5. **Center-of-Mass Shift:** Calculate the centroid of intensity and apply small translational shifts using `scipy.ndimage.shift`.
6. **Feature Normalization:** Divide pixel intensities by 255.0 to map values into $[0.0, 1.0]$ float32.

#### Q6: Why do we normalize pixel intensities from [0, 255] to [0.0, 1.0]?
**Answer:**  
Both k-NN and SVM rely on mathematical distance or inner-product computations ($L_2$ norm or $\mathbf{w}^T\mathbf{x}$). Unnormalized features ranging from 0 to 255 can lead to numerical instability, excessive gradient updates in optimization, and slower convergence. Normalizing to $[0.0, 1.0]$ scales all features uniformly.

---

### Category 3: k-Nearest Neighbors (k-NN)

#### Q7: How does k-NN classify a new digit?
**Answer:**  
k-NN is an instance-based lazy learner:
1. It stores the training feature vectors in memory without an explicit parameter training phase.
2. Given a query digit $\mathbf{x}$, it computes the distance to every stored training instance.
3. It selects the $k$ nearest instances and predicts the class that has the majority vote (or distance-weighted vote).

#### Q8: What distance metrics did you compare in k-NN?
**Answer:**  
We compared **Euclidean distance** ($L_2$ norm: $\sqrt{\sum (x_j - x_{i,j})^2}$) and **Manhattan distance** ($L_1$ norm: $\sum |x_j - x_{i,j}|$). Euclidean distance yielded slightly higher cross-validation accuracy on continuous pixel intensity spaces.

#### Q9: What happens if $k$ is too small or too large?
**Answer:**  
- **If $k = 1$:** The decision boundary becomes overly complex, fitting noise and individual outlier drawings (high variance, overfitting).
- **If $k$ is very large (e.g., $k = 100$):** The boundary becomes excessively smoothed, diluting local neighborhood structure and favoring the most frequent class (high bias, underfitting).
- Our hyperparameter search identified $k = 3$ or $k = 5$ as optimal for MNIST.

#### Q10: What are the main limitations of k-NN?
**Answer:**  
High inference latency and memory requirements. During prediction, k-NN must calculate distances to all $N$ training points, making prediction $O(N \cdot D)$. For $N = 60,000$ and $D = 784$, real-time inference on low-power devices can be sluggish.

---

### Category 4: Support Vector Machines (SVM) & Kernels

#### Q11: What is a Support Vector Machine, and what are "support vectors"?
**Answer:**  
A Support Vector Machine is a supervised learning model that finds the optimal hyperplane that separates classes with the maximum geometric margin. **Support vectors** are the critical training instances that lie closest to the decision hyperplane; they alone define the position and orientation of the margin. Removing non-support-vector data points has zero effect on the decision boundary.

#### Q12: What is the role of the hyperparameter $C$ in SVM?
**Answer:**  
$C$ is the regularization parameter in the soft-margin formulation:
- **Large $C$:** Severely penalizes margin violations, aiming to classify every training point correctly. This leads to a narrower margin and risks overfitting.
- **Small $C$:** Tolerates more classification errors in favor of a wider, softer margin, enhancing generalization and reducing variance.

#### Q13: What is the "Kernel Trick", and why is it used?
**Answer:**  
The kernel trick allows an SVM to compute inner products in a high-dimensional feature space without explicitly calculating the high-dimensional coordinates of the data points. Instead of transforming $\mathbf{x} \mapsto \Phi(\mathbf{x})$, the kernel function computes:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$$
This enables non-linear classification with the computational complexity of the lower-dimensional input space.

#### Q14: What is the RBF kernel, and what does parameter $\gamma$ control?
**Answer:**  
The Radial Basis Function (RBF) or Gaussian kernel is defined as:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
- **$\gamma$ (gamma):** Dictates the radius of influence of each support vector.
- **High $\gamma$:** Each training point has a very localized influence, creating complex, tight boundaries around individual points (risks overfitting).
- **Low $\gamma$:** Each point has a broad reach, resulting in smoother, more linear-like decision boundaries (risks underfitting).

#### Q15: Why did RBF SVM outperform Linear SVM on MNIST?
**Answer:**  
Because raw pixel features of handwritten digits are not linearly separable. Digits like 8, 0, and 6 possess enclosed loops and curves that overlap spatially with open-stroke digits like 3, 5, and 1. A single flat hyperplane cannot separate these intertwined geometries, whereas the RBF kernel implicitly projects them into an infinite-dimensional space where clean linear separation is possible.

---

### Category 5: Experimental Methodology & Validation

#### Q16: How did you ensure zero test-set leakage in your experiments?
**Answer:**  
1. We strictly partitioned the dataset into training and test splits prior to any feature transformation or model tuning.
2. Hyperparameter optimization was conducted solely on the training partition using **Stratified 3-Fold Cross-Validation**.
3. The test set was held untouched until final model evaluation.
4. Preprocessing parameters (scaling parameters) were learned exclusively from the training set.

#### Q17: What is the difference between Macro F1 and Weighted F1?
**Answer:**  
- **Macro F1:** Computes the arithmetic mean of F1-scores across all 10 classes without weighting by class frequencies:
  $$\text{Macro F1} = \frac{1}{10} \sum_{c=0}^9 \text{F1}_c$$
  It treats all classes equally, making it sensitive to poor performance on underrepresented classes.
- **Weighted F1:** Weights each class F1-score by the number of ground-truth instances (support) in that class. Since MNIST is virtually balanced (~10% per class), both scores are very close.

#### Q18: Explain the components of a Confusion Matrix.
**Answer:**  
A confusion matrix is a $C \times C$ grid (where $C = 10$ for MNIST) where:
- Rows represent the true actual class labels.
- Columns represent the predicted class labels.
- Diagonal elements $(i, i)$ denote **True Positives** (correct predictions).
- Off-diagonal elements $(i, j)$ denote **Misclassifications** (true digit $i$ misclassified as digit $j$).
For example, off-diagonal spikes at row 4, column 9 indicate that written 4s are frequently confused with 9s.

#### Q19: How are confidence scores derived for SVM vs k-NN in your web app?
**Answer:**  
- **For k-NN:** Confidence represents the proportion of neighbor votes for the winning class among the $k$ nearest neighbors.
- **For Linear SVM & RBF SVM:** Standard SVM outputs geometric margin distances ($\mathbf{w}^T\mathbf{x} + b$). We obtain well-calibrated posterior probabilities $P(y=c|\mathbf{x})$ using Platt scaling (`CalibratedClassifierCV` / logistic regression fit on cross-validated decision values) or softmax transformation.

#### Q20: Summarize the runtime vs accuracy trade-offs observed among the three models.
**Answer:**  
1. **RBF SVM:** Highest test accuracy (96.55%) and Macro F1, but slowest training time ($O(N^2)$).
2. **k-NN:** High accuracy (94.60%) with instantaneous training ($O(1)$), but higher inference latency per query.
3. **Linear SVM:** Lowest inference latency (~16.96 ms / sample) and fast training (7.88 s), but lower ceiling accuracy (90.55%) due to linear boundary constraints.

---

### Category 6: Multi-Digit Number Recognition & Computer Vision

#### Q21: Why can't a single-digit MNIST model directly recognize multi-digit numbers like 2026?
**Answer:**  
A single-digit MNIST model expects a fixed 784-dimensional input vector corresponding to an isolated, centered $28 \times 28$ image. A multi-digit image has arbitrary width, variable character counts, and irregular spacing. To recognize complete numbers, we must first segment the image into individual digit components, normalize each component to $28 \times 28$, predict each digit sequentially, and concatenate the predictions from left to right.

#### Q22: How does your pipeline handle multi-stroke digits (e.g., cross on 7, top bar of 5, or disconnected strokes of 4)?
**Answer:**  
A naive connected-component approach would mistakenly treat the detached top bar of a '5' or cross on a '7' as separate digits. Our algorithm prevents this by computing horizontal overlap and proximity between detected bounding boxes. If two components have significant horizontal overlap ($> 50\%$) or are in close vertical proximity, they are merged into a single enclosing bounding box before classification.

#### Q23: How does the system split touching digits?
**Answer:**  
When two digits touch, their bounding box has an unusually wide aspect ratio ($w / h > 1.25$). Our system calculates the **Vertical Projection Profile**—the column-wise sum of pixel intensities across the bounding box:
$$\text{Profile}(x) = \sum_{y} I(x, y)$$
It searches for a deep local minimum (valley) in the central region (between 25% and 75% of width). If a valley is found whose intensity is significantly lower than neighboring peaks, the bounding box is bifurcated at that vertical line.

#### Q24: Why is it important to treat the predicted number as a string rather than an integer?
**Answer:**  
Numbers with leading zeros (e.g. "07", "007", "0123") lose their leading zeros if cast immediately to an integer (`int("07")` becomes `7`). By maintaining the predictions as a string sequence of character labels, leading zeros are preserved accurately.

