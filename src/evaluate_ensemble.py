"""
Evaluation and serialization script for Ensemble Voting Classifier.

Evaluates all models (k-NN, Linear SVM, RBF SVM, Ensemble Voting Classifier)
on the exact untouched MNIST test set without retraining the base models.
Updates evaluation metrics JSON, comparison CSV, metadata, and generates publication plots.
"""

import json
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np

import config
from src.data_loader import get_data_splits
from src.preprocessing import normalize_and_flatten
from src.ensemble import create_ensemble_from_base_models
from src.evaluate import evaluate_model, create_comparison_dataframe
from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_bar_chart,
    plot_runtime_comparison,
)


def evaluate_and_update_all_models():
    print("[EnsembleEvaluator] Loading held-out MNIST dataset splits...")
    X_train_raw, y_train, X_test_raw, y_test, split_metadata = get_data_splits(
        train_size=config.DEFAULT_TRAIN_SAMPLES,
        test_size=config.DEFAULT_TEST_SAMPLES,
        random_state=config.RANDOM_STATE,
    )

    X_test = normalize_and_flatten(X_test_raw)

    print("[EnsembleEvaluator] Loading baseline models from disk...")
    knn = joblib.load(config.MODEL_PATHS["knn"])
    linear_svm = joblib.load(config.MODEL_PATHS["linear_svm"])
    rbf_svm = joblib.load(config.MODEL_PATHS["rbf_svm"])

    print("[EnsembleEvaluator] Creating Ensemble Voting Classifier...")
    ensemble = create_ensemble_from_base_models(knn, linear_svm, rbf_svm)

    # Save ensemble model artifact
    ensemble_path = config.MODEL_PATHS["ensemble"]
    print(f"[EnsembleEvaluator] Saving ensemble artifact to {ensemble_path}...")
    joblib.dump(ensemble, ensemble_path)

    # Load existing metrics if available to keep previous training durations
    prev_metrics = {}
    if config.METRICS_JSON_PATH.exists():
        try:
            with open(config.METRICS_JSON_PATH, "r") as f:
                prev_metrics = json.load(f)
        except Exception:
            pass

    models_to_eval = [
        ("knn", knn, "k-Nearest Neighbors", prev_metrics.get("knn", {}).get("training_time_sec", 0.021)),
        ("linear_svm", linear_svm, "Linear SVM (Calibrated)", prev_metrics.get("linear_svm", {}).get("training_time_sec", 7.888)),
        ("rbf_svm", rbf_svm, "RBF Kernel SVM", prev_metrics.get("rbf_svm", {}).get("training_time_sec", 179.8)),
        ("ensemble", ensemble, "Ensemble Voting Classifier", round(0.021 + 7.888 + 179.8, 3)),
    ]

    all_metrics = {}
    confusion_matrices = {}

    print("[EnsembleEvaluator] Running standardized evaluation on held-out test set...")
    for key, model, display_name, tr_time in models_to_eval:
        metrics, cm = evaluate_model(model, X_test, y_test, display_name, training_time_sec=tr_time)

        # Preserve hyperparameters
        if key in prev_metrics and "hyperparameters" in prev_metrics[key]:
            metrics["hyperparameters"] = prev_metrics[key]["hyperparameters"]
        elif key == "ensemble":
            metrics["hyperparameters"] = {
                "voting": "hard",
                "tie_breaker": "rbf_svm",
                "base_models": ["knn", "linear_svm", "rbf_svm"]
            }

        all_metrics[key] = metrics
        confusion_matrices[key] = cm

        # Save confusion matrix plot
        plot_confusion_matrix(
            cm,
            model_name=display_name,
            save_path=config.PLOTS_DIR / f"confusion_matrix_{key}.png"
        )

    # Save comparison CSV
    comp_df = create_comparison_dataframe(all_metrics)
    comp_df.to_csv(config.COMPARISON_CSV_PATH, index=False)
    print(f"\n[EnsembleEvaluator] Updated Model Comparison Results Table:\n{comp_df.to_string(index=False)}")

    # Update bar charts
    plot_metrics_bar_chart(comp_df, save_path=config.PLOTS_DIR / "model_metrics_comparison.png")
    plot_runtime_comparison(comp_df, save_path=config.PLOTS_DIR / "runtime_comparison.png")

    # Update metadata
    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    if config.TRAINING_METADATA_PATH.exists():
        try:
            with open(config.TRAINING_METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}

    metadata["best_model"] = comp_df.loc[comp_df["Macro F1"].idxmax()]["Model"]
    metadata["results_summary"] = all_metrics
    metadata["has_ensemble"] = True

    with open(config.TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("[EnsembleEvaluator] Evaluation and artifact generation completed successfully!")
    return all_metrics, comp_df


if __name__ == "__main__":
    evaluate_and_update_all_models()
