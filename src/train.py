"""
Training and Hyperparameter Optimization Pipeline.

Trains:
1. k-Nearest Neighbors (KNeighborsClassifier)
2. Linear Support Vector Machine (LinearSVC)
3. Radial Basis Function Support Vector Machine (SVC with RBF kernel)

Features:
- Stratified K-Fold Cross-Validation on training data only (no data leakage).
- Hyperparameter tuning for k, distance metric, C, and gamma.
- Performance benchmarking (Training time, Inference latency).
- Evaluation on untouched test set.
- Persistence of trained models, metrics, metadata, and publication plots.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

import config
from src.data_loader import get_data_splits
from src.preprocessing import normalize_and_flatten
from src.evaluate import evaluate_model, create_comparison_dataframe
from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_bar_chart,
    plot_runtime_comparison,
    plot_sample_digit_grid,
)


def tune_and_train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = config.CV_FOLDS,
    tune: bool = True,
) -> Any:
    """
    Trains k-NN classifier with hyperparameter search over k and distance metrics.
    """
    print("\n" + "=" * 50)
    print(">>> [1/3] Training k-Nearest Neighbors Classifier...")
    print("=" * 50)

    if tune:
        print(f"[k-NN] Tuning hyperparameters via {cv_folds}-Fold Stratified CV on training data...")
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "metric": ["euclidean", "manhattan"],
            "weights": ["distance", "uniform"]
        }
        # Use stratified sub-sample for grid search if training set is large to keep runtime snappy
        cv_sample_size = min(4000, len(y_train))
        cv_idx = np.random.choice(len(y_train), cv_sample_size, replace=False)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        grid = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train[cv_idx], y_train[cv_idx])
        best_params = grid.best_params_
        print(f"[k-NN] Best CV Parameters found: {best_params} (CV Accuracy: {grid.best_score_*100:.2f}%)")
    else:
        best_params = {"n_neighbors": 5, "metric": "euclidean", "weights": "distance"}
        print(f"[k-NN] Using default parameters: {best_params}")

    # Fit final model on full training set
    print(f"[k-NN] Fitting final model on {len(y_train)} training instances...")
    final_knn = KNeighborsClassifier(**best_params, n_jobs=-1)
    
    t0 = time.perf_counter()
    final_knn.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"[k-NN] Model trained in {train_time:.2f} seconds.")
    return final_knn, best_params, train_time


def tune_and_train_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = config.CV_FOLDS,
    tune: bool = True,
) -> Any:
    """
    Trains Linear SVM with hyperparameter search over regularizer C.
    Wraps with CalibratedClassifierCV to support reliable probability estimates.
    """
    print("\n" + "=" * 50)
    print(">>> [2/3] Training Linear SVM Classifier...")
    print("=" * 50)

    if tune:
        print(f"[Linear SVM] Tuning penalty parameter C via {cv_folds}-Fold Stratified CV...")
        param_grid = {"C": [0.01, 0.1, 1.0, 5.0]}
        cv_sample_size = min(4000, len(y_train))
        cv_idx = np.random.choice(len(y_train), cv_sample_size, replace=False)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        grid = GridSearchCV(
            LinearSVC(dual=False, max_iter=2500, random_state=config.RANDOM_STATE),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train[cv_idx], y_train[cv_idx])
        best_params = grid.best_params_
        print(f"[Linear SVM] Best CV Parameter found: {best_params} (CV Accuracy: {grid.best_score_*100:.2f}%)")
    else:
        best_params = {"C": 0.1}
        print(f"[Linear SVM] Using default parameter: {best_params}")

    print(f"[Linear SVM] Fitting final calibrated model on {len(y_train)} training instances...")
    base_svc = LinearSVC(dual=False, C=best_params["C"], max_iter=3000, random_state=config.RANDOM_STATE)
    # CalibratedClassifierCV uses sigmoid/Platt calibration so predict_proba is calibrated
    final_linear_svm = CalibratedClassifierCV(base_svc, cv=3)

    t0 = time.perf_counter()
    final_linear_svm.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"[Linear SVM] Model trained and calibrated in {train_time:.2f} seconds.")
    return final_linear_svm, best_params, train_time


def tune_and_train_rbf_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = config.CV_FOLDS,
    tune: bool = True,
) -> Any:
    """
    Trains RBF Kernel SVM with hyperparameter search over C and gamma.
    """
    print("\n" + "=" * 50)
    print(">>> [3/3] Training RBF Kernel SVM Classifier...")
    print("=" * 50)

    if tune:
        print(f"[RBF SVM] Tuning C and gamma via {cv_folds}-Fold Stratified CV...")
        param_grid = {
            "C": [1.0, 5.0, 10.0],
            "gamma": ["scale", 0.01]
        }
        # RBF grid search on a representative subset
        cv_sample_size = min(3000, len(y_train))
        cv_idx = np.random.choice(len(y_train), cv_sample_size, replace=False)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        grid = GridSearchCV(
            SVC(kernel="rbf", random_state=config.RANDOM_STATE),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train[cv_idx], y_train[cv_idx])
        best_params = grid.best_params_
        print(f"[RBF SVM] Best CV Parameters found: {best_params} (CV Accuracy: {grid.best_score_*100:.2f}%)")
    else:
        best_params = {"C": 5.0, "gamma": "scale"}
        print(f"[RBF SVM] Using default parameters: {best_params}")

    print(f"[RBF SVM] Fitting final RBF model with probability estimates on {len(y_train)} instances...")
    final_rbf_svm = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        probability=True,
        random_state=config.RANDOM_STATE
    )

    t0 = time.perf_counter()
    final_rbf_svm.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"[RBF SVM] Model trained in {train_time:.2f} seconds.")
    return final_rbf_svm, best_params, train_time


def run_training_pipeline(
    train_samples: int = config.DEFAULT_TRAIN_SAMPLES,
    test_samples: int = config.DEFAULT_TEST_SAMPLES,
    tune: bool = True,
) -> Dict[str, Any]:
    """
    Complete end-to-end training, evaluation, artifact export, and visualization.
    """
    start_time_all = time.time()
    print("==================================================================")
    print("   HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING PIPELINE")
    print("==================================================================")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Requested Train Samples: {train_samples}")
    print(f"Requested Test Samples:  {test_samples}")
    print(f"Hyperparameter Tuning:   {tune}")

    # 1. Load Data Splits (No leakage guaranteed)
    X_train_raw, y_train, X_test_raw, y_test, split_metadata = get_data_splits(
        train_size=train_samples,
        test_size=test_samples,
        random_state=config.RANDOM_STATE,
    )

    # 2. Normalize and Flatten Features to [0, 1] float32
    print("[Pipeline] Normalizing pixel intensities to [0.0, 1.0]...")
    X_train = normalize_and_flatten(X_train_raw)
    X_test = normalize_and_flatten(X_test_raw)

    print(f"[Pipeline] X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"[Pipeline] X_test shape:  {X_test.shape},  dtype: {X_test.dtype}")

    # 3. Train all 3 models
    knn_model, knn_params, knn_train_time = tune_and_train_knn(X_train, y_train, tune=tune)
    linear_svm_model, linear_svm_params, linear_svm_train_time = tune_and_train_linear_svm(X_train, y_train, tune=tune)
    rbf_svm_model, rbf_svm_params, rbf_svm_train_time = tune_and_train_rbf_svm(X_train, y_train, tune=tune)

    models_dict = {
        "knn": (knn_model, "k-Nearest Neighbors", knn_params, knn_train_time),
        "linear_svm": (linear_svm_model, "Linear SVM (Calibrated)", linear_svm_params, linear_svm_train_time),
        "rbf_svm": (rbf_svm_model, "RBF Kernel SVM", rbf_svm_params, rbf_svm_train_time),
    }

    # 4. Save model artifacts
    print("\n[Pipeline] Serializing models to disk...")
    for key, (mod, display_name, _, _) in models_dict.items():
        save_path = config.MODEL_PATHS[key]
        joblib.dump(mod, save_path)
        print(f" -> Saved {display_name} to {save_path}")

    # 5. Evaluate on untouched test set
    print("\n" + "=" * 50)
    print(">>> Evaluating All Models on Untouched Test Set...")
    print("=" * 50)

    evaluation_results = {}
    confusion_matrices = {}

    for key, (mod, display_name, params, tr_time) in models_dict.items():
        metrics, cm = evaluate_model(mod, X_test, y_test, display_name, training_time_sec=tr_time)
        metrics["hyperparameters"] = params
        evaluation_results[key] = metrics
        confusion_matrices[key] = cm

        # Save confusion matrix plot
        plot_confusion_matrix(
            cm,
            model_name=display_name,
            save_path=config.PLOTS_DIR / f"confusion_matrix_{key}.png"
        )

    # 6. Create comparative results table
    comp_df = create_comparison_dataframe(evaluation_results)
    comp_df.to_csv(config.COMPARISON_CSV_PATH, index=False)
    print(f"\n[Pipeline] Model Comparison Results Table:\n{comp_df.to_string(index=False)}")

    # 7. Generate comparative charts
    plot_metrics_bar_chart(comp_df, save_path=config.PLOTS_DIR / "model_metrics_comparison.png")
    plot_runtime_comparison(comp_df, save_path=config.PLOTS_DIR / "runtime_comparison.png")
    plot_sample_digit_grid(X_train_raw, y_train, save_path=config.PLOTS_DIR / "sample_digits_grid.png")

    # 8. Record training metadata & results JSON
    metadata = {
        "execution_date": datetime.now().isoformat(),
        "total_pipeline_duration_sec": round(time.time() - start_time_all, 2),
        "dataset": {
            "source": "OpenML mnist_784 / Yann LeCun benchmark",
            "train_samples_used": len(y_train),
            "test_samples_used": len(y_test),
            "is_train_subset": split_metadata["is_train_subset"],
            "is_test_subset": split_metadata["is_test_subset"],
            "image_dimensions": [28, 28],
            "num_features": 784,
            "random_state": config.RANDOM_STATE,
        },
        "best_model": comp_df.loc[comp_df["Macro F1"].idxmax()]["Model"],
        "results_summary": evaluation_results,
    }

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"[Pipeline] Saved metrics to {config.METRICS_JSON_PATH}")

    with open(config.TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Pipeline] Saved training metadata to {config.TRAINING_METADATA_PATH}")

    print("\n" + "=" * 65)
    print(">>> TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY! <<<")
    print("==================================================================")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST classifiers (k-NN, Linear SVM, RBF SVM).")
    parser.add_argument("--train-samples", type=int, default=config.DEFAULT_TRAIN_SAMPLES,
                        help="Number of training samples (e.g. 10000 or 60000)")
    parser.add_argument("--test-samples", type=int, default=config.DEFAULT_TEST_SAMPLES,
                        help="Number of test samples (e.g. 2000 or 10000)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Disable CV hyperparameter tuning and use preset optimal parameters")
    args = parser.parse_args()

    run_training_pipeline(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        tune=not args.no_tune,
    )
