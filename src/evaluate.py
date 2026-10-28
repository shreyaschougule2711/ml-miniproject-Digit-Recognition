"""
Model Evaluation and Metrics Computation Module.

Computes comprehensive metrics:
- Accuracy
- Macro & Weighted Precision, Recall, F1-Score
- Per-class classification reports
- Confusion Matrices
- Single-sample and batch prediction latency
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    training_time_sec: float = 0.0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates a trained classifier on the test dataset.

    Returns:
        metrics_dict: Dictionary of all calculated metrics and latency benchmarks
        cm: Confusion matrix array of shape (10, 10)
    """
    print(f"[Evaluate] Evaluating {model_name} on {X_test.shape[0]} test samples...")

    # Benchmark prediction latency on batch
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    batch_inference_time = time.perf_counter() - t0

    # Benchmark single sample latency (average of 50 single predictions)
    single_latencies = []
    num_benchmarks = min(50, len(X_test))
    for i in range(num_benchmarks):
        sample = X_test[i : i + 1]
        t_single = time.perf_counter()
        _ = model.predict(sample)
        single_latencies.append((time.perf_counter() - t_single) * 1000.0)  # ms
    avg_single_latency_ms = float(np.mean(single_latencies))

    # Standard metrics
    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(10)), zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    clf_report = classification_report(
        y_test, y_pred, labels=list(range(10)), output_dict=True, zero_division=0
    )

    metrics_dict = {
        "model_name": model_name,
        "test_samples": int(len(y_test)),
        "accuracy": round(acc, 5),
        "macro_precision": round(float(p_macro), 5),
        "macro_recall": round(float(r_macro), 5),
        "macro_f1": round(float(f1_macro), 5),
        "weighted_f1": round(float(f1_weighted), 5),
        "training_time_sec": round(training_time_sec, 3),
        "batch_prediction_time_sec": round(batch_inference_time, 3),
        "latency_ms_per_sample": round(avg_single_latency_ms, 2),
        "per_class": {
            str(i): {
                "precision": round(float(p_per_class[i]), 4),
                "recall": round(float(r_per_class[i]), 4),
                "f1_score": round(float(f1_per_class[i]), 4),
                "support": int(support_per_class[i]),
            }
            for i in range(10)
        },
        "classification_report_raw": clf_report,
        "confusion_matrix": cm.tolist(),
    }

    print(f"[Evaluate] {model_name} -> Accuracy: {acc*100:.2f}%, Macro F1: {f1_macro*100:.2f}%")
    return metrics_dict, cm


def create_comparison_dataframe(all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms dictionary of model metrics into a structured Pandas DataFrame.
    """
    rows = []
    for model_key, m in all_metrics.items():
        rows.append({
            "Model Key": model_key,
            "Model": m.get("model_name", model_key),
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Training Time (s)": m["training_time_sec"],
            "Prediction Time (s)": m["batch_prediction_time_sec"],
            "Inference Latency (ms/sample)": m["latency_ms_per_sample"],
        })
    df = pd.DataFrame(rows)
    return df
