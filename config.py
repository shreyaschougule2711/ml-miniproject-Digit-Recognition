"""
Project Configuration File
Handwritten Digit Recognition Using SVM and k-NN
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TESTS_DIR = BASE_DIR / "tests"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT  # 784
NUM_CLASSES = 10
DIGIT_CLASSES = [str(i) for i in range(10)]

# Stratified subset defaults for fast, responsive CPU experimentation
# Default subset: 10,000 training images, 2,000 test images (ensures < 2 min CPU runtime)
# Full dataset: 60,000 training images, 10,000 test images
DEFAULT_TRAIN_SAMPLES = 10000
DEFAULT_TEST_SAMPLES = 2000
RANDOM_STATE = 42

# Hyperparameter search grids
KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9],
    "metric": ["euclidean", "manhattan"],
    "weights": ["distance", "uniform"]
}

LINEAR_SVM_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 5.0]
}

RBF_SVM_PARAM_GRID = {
    "C": [1.0, 5.0, 10.0],
    "gamma": ["scale", 0.01, 0.001]
}

# Cross-Validation Folds
CV_FOLDS = 3

# Model Save Paths
MODEL_PATHS = {
    "knn": MODELS_DIR / "knn_model.joblib",
    "linear_svm": MODELS_DIR / "linear_svm_model.joblib",
    "rbf_svm": MODELS_DIR / "rbf_svm_model.joblib",
}

# Artifact Paths
METRICS_JSON_PATH = RESULTS_DIR / "evaluation_metrics.json"
TRAINING_METADATA_PATH = RESULTS_DIR / "training_metadata.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "model_comparison.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
