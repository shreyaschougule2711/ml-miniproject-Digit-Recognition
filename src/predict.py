"""
Prediction and Inference Engine for Handwritten Digits.

Loads serialized models and provides unified prediction methods with
transparent confidence scores, top alternative predictions, and score interpretations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import joblib

import config
from src.preprocessing import preprocess_canvas_image, normalize_and_flatten


# In-memory model cache for fast real-time inference
_LOADED_MODELS: Dict[str, Any] = {}


def load_model(model_key: str) -> Optional[Any]:
    """
    Loads a model by key ('knn', 'linear_svm', 'rbf_svm') from disk with caching.
    """
    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key]

    path = config.MODEL_PATHS.get(model_key)
    if not path or not path.exists():
        return None

    try:
        model = joblib.load(path)
        _LOADED_MODELS[model_key] = model
        return model
    except Exception as e:
        print(f"[Predict] Error loading model {model_key} from {path}: {e}")
        return None


def are_models_available() -> Dict[str, bool]:
    """
    Checks if trained model artifacts exist on disk.
    """
    return {
        key: path.exists() for key, path in config.MODEL_PATHS.items()
    }


def softmax(scores: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax over scores."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def predict_digit(
    model_key: str,
    flattened_784: np.ndarray,
) -> Dict[str, Any]:
    """
    Executes digit prediction using the specified model.

    Args:
        model_key: 'knn', 'linear_svm', or 'rbf_svm'
        flattened_784: np.ndarray of shape (1, 784), values in [0.0, 1.0]

    Returns:
        result dictionary containing:
        - predicted_digit (int)
        - confidence (float)
        - top_3_predictions (list of dicts)
        - all_class_scores (dict)
        - score_type (str)
        - explanation (str)
    """
    model = load_model(model_key)
    if model is None:
        return {
            "error": f"Model '{model_key}' is not trained or found at {config.MODEL_PATHS.get(model_key)}. Please run model training first.",
            "success": False
        }

    # Ensure shape (1, 784)
    if flattened_784.ndim == 1:
        flattened_784 = flattened_784.reshape(1, -1)

    predicted_digit = int(model.predict(flattened_784)[0])

    score_type = "Confidence Score"
    explanation = ""
    class_probabilities = np.zeros(10)

    # 1. k-NN: Uses predict_proba if supported or voting weights
    if model_key == "knn":
        score_type = "Neighbor Vote Proportion (%)"
        explanation = (
            "Based on the proportion of the k-nearest training examples in 784-dimensional pixel space "
            "that voted for this digit class."
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flattened_784)[0]
            # Ensure 10 classes
            classes = model.classes_
            for idx, c in enumerate(classes):
                class_probabilities[int(c)] = probs[idx]
        else:
            class_probabilities[predicted_digit] = 1.0

    # 2. Linear SVM: Uses decision_function (signed distance to decision hyperplane)
    elif model_key == "linear_svm":
        score_type = "Decision Margin (Normalized via Softmax)"
        explanation = (
            "Linear SVM separates classes using hyperplanes. The decision score reflects the geometric "
            "distance of this sample to the separating hyperplane margin."
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flattened_784)[0]
            for idx, c in enumerate(model.classes_):
                class_probabilities[int(c)] = probs[idx]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(flattened_784)[0]
            # Convert raw margins to relative probabilities via softmax
            class_probabilities = softmax(raw_scores)
        else:
            class_probabilities[predicted_digit] = 1.0

    # 3. RBF SVM: Radial Basis Function kernel
    elif model_key == "rbf_svm":
        score_type = "Nonlinear Kernel Probability"
        explanation = (
            "RBF SVM maps pixel patterns into an infinite-dimensional feature space using Gaussian kernels, "
            "distinguishing curved strokes and nonlinear digit contours."
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flattened_784)[0]
            for idx, c in enumerate(model.classes_):
                class_probabilities[int(c)] = probs[idx]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(flattened_784)[0]
            class_probabilities = softmax(raw_scores)
        else:
            class_probabilities[predicted_digit] = 1.0

    confidence = float(class_probabilities[predicted_digit])

    # Top 3 predictions
    sorted_indices = np.argsort(class_probabilities)[::-1]
    top_3 = []
    for rank, idx in enumerate(sorted_indices[:3], 1):
        top_3.append({
            "rank": rank,
            "digit": int(idx),
            "score": round(float(class_probabilities[idx]), 4),
            "percentage": round(float(class_probabilities[idx]) * 100, 2)
        })

    all_scores = {
        str(i): round(float(class_probabilities[i]), 4) for i in range(10)
    }

    return {
        "success": True,
        "model_key": model_key,
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "top_3": top_3,
        "all_scores": all_scores,
        "score_type": score_type,
        "explanation": explanation,
    }
