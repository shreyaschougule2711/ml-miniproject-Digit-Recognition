"""
Prediction and Inference Engine for Handwritten Digits.

Loads serialized models and provides unified prediction methods with
transparent confidence scores, top alternative predictions, and score interpretations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import joblib

import config
from src.preprocessing import preprocess_canvas_image, normalize_and_flatten
from src.ensemble import EnsembleVotingClassifier, create_ensemble_from_base_models


# In-memory model cache for fast real-time inference
_LOADED_MODELS: Dict[str, Any] = {}


def load_model(model_key: str) -> Optional[Any]:
    """
    Loads a model by key ('knn', 'linear_svm', 'rbf_svm', 'ensemble') from disk with caching.
    If 'ensemble' artifact is not serialized on disk, dynamically constructs it from the 3 base models.
    """
    canonical_key = "ensemble" if model_key in ("ensemble", "ensemble_voting") else model_key
    if canonical_key in _LOADED_MODELS:
        return _LOADED_MODELS[canonical_key]

    path = config.MODEL_PATHS.get(canonical_key)
    if path and path.exists():
        try:
            model = joblib.load(path)
            _LOADED_MODELS[canonical_key] = model
            return model
        except Exception as e:
            print(f"[Predict] Error loading model {canonical_key} from {path}: {e}")

    # Fallback for ensemble: build dynamically from base models if available
    if canonical_key == "ensemble":
        knn = load_model("knn")
        linear_svm = load_model("linear_svm")
        rbf_svm = load_model("rbf_svm")
        if knn is not None and linear_svm is not None and rbf_svm is not None:
            ensemble = create_ensemble_from_base_models(knn, linear_svm, rbf_svm)
            _LOADED_MODELS["ensemble"] = ensemble
            return ensemble

    return None


def are_models_available() -> Dict[str, bool]:
    """
    Checks if trained model artifacts exist on disk and have non-trivial size (>1KB).
    Ensemble is considered ready if its serialized file exists OR if all 3 base models exist.
    """
    res = {}
    for key in ["knn", "linear_svm", "rbf_svm"]:
        path = config.MODEL_PATHS.get(key)
        try:
            res[key] = bool(path and path.exists() and path.stat().st_size > 1000)
        except Exception:
            res[key] = False

    ens_path = config.MODEL_PATHS.get("ensemble")
    if ens_path and ens_path.exists() and ens_path.stat().st_size > 1000:
        res["ensemble"] = True
    else:
        res["ensemble"] = bool(res.get("knn") and res.get("linear_svm") and res.get("rbf_svm"))

    return res


def softmax(scores: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax over scores."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def predict_digit(
    model_or_data: Union[str, np.ndarray] = "ensemble",
    flattened_784: Optional[np.ndarray] = None,
    model_key: str = "ensemble",
) -> Dict[str, Any]:
    """
    Executes digit prediction using the unified Ensemble Voting Classifier (default)
    or a specified baseline model.

    Supports:
        - predict_digit(flattened_784)
        - predict_digit("ensemble", flattened_784)
        - predict_digit("knn", flattened_784)
    """
    if isinstance(model_or_data, str):
        target_model_key = model_or_data
        target_data = flattened_784
    else:
        target_model_key = model_key
        target_data = model_or_data

    if target_data is None:
        return {
            "error": "No feature vector provided for prediction.",
            "success": False,
        }

    # Ensure shape (1, 784)
    if target_data.ndim == 1:
        target_data = target_data.reshape(1, -1)

    model = load_model(target_model_key)
    if model is None:
        return {
            "error": f"Model '{target_model_key}' is not trained or found at {config.MODEL_PATHS.get(target_model_key)}. Please run model training first.",
            "success": False,
        }

    predicted_digit = int(model.predict(target_data)[0])

    score_type = "Confidence Score"
    explanation = ""
    class_probabilities = np.zeros(10)

    # 1. k-NN: Uses predict_proba if supported or voting weights
    if target_model_key == "knn":
        score_type = "Neighbor Vote Proportion (%)"
        explanation = (
            "Based on the proportion of the k-nearest training examples in 784-dimensional pixel space "
            "that voted for this digit class."
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(target_data)[0]
            classes = model.classes_
            for idx, c in enumerate(classes):
                class_probabilities[int(c)] = probs[idx]
        else:
            class_probabilities[predicted_digit] = 1.0

    # 2. Linear SVM: Uses decision_function (signed distance to decision hyperplane)
    elif target_model_key == "linear_svm":
        score_type = "Decision Margin (Normalized via Softmax)"
        explanation = (
            "Linear SVM separates classes using hyperplanes. The decision score reflects the geometric "
            "distance of this sample to the separating hyperplane margin."
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(target_data)[0]
            for idx, c in enumerate(model.classes_):
                class_probabilities[int(c)] = probs[idx]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(target_data)[0]
            class_probabilities = softmax(raw_scores)
        else:
            class_probabilities[predicted_digit] = 1.0

    # 3. RBF SVM: Radial Basis Function kernel
    elif target_model_key == "rbf_svm":
        score_type = "Nonlinear Kernel Probability"
        explanation = (
            "RBF SVM maps pixel patterns into an infinite-dimensional feature space using Gaussian kernels, "
            "distinguishing curved strokes and nonlinear digit contours."
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(target_data)[0]
            for idx, c in enumerate(model.classes_):
                class_probabilities[int(c)] = probs[idx]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(target_data)[0]
            class_probabilities = softmax(raw_scores)
        else:
            class_probabilities[predicted_digit] = 1.0

    # 4. Ensemble: Hard Majority Voting over k-NN, Linear SVM, and RBF SVM
    elif target_model_key in ("ensemble", "ensemble_voting"):
        score_type = "Hard Majority Vote Consensus (3 Classifiers)"
        pred_knn = predict_digit("knn", target_data)
        pred_lin = predict_digit("linear_svm", target_data)
        pred_rbf = predict_digit("rbf_svm", target_data)

        base_models_results = {
            "knn": pred_knn,
            "linear_svm": pred_lin,
            "rbf_svm": pred_rbf,
        }

        # Fallback if any base model is unavailable
        if not (pred_knn.get("success") and pred_lin.get("success") and pred_rbf.get("success")):
            if hasattr(model, "predict"):
                predicted_digit = int(model.predict(target_data)[0])
                conf_val = 1.0
                class_probabilities = np.zeros(10)
                class_probabilities[predicted_digit] = 1.0
                consensus_type = "Ensemble Classifier Inference"
                explanation = "Inference performed using loaded Ensemble Voting Classifier artifact."
                votes = {"ensemble": predicted_digit}
            else:
                return {
                    "error": "Ensemble prediction failed because one or more baseline models could not be executed.",
                    "success": False,
                }
        else:
            v_knn = pred_knn["predicted_digit"]
            v_lin = pred_lin["predicted_digit"]
            v_rbf = pred_rbf["predicted_digit"]
            votes = {"knn": v_knn, "linear_svm": v_lin, "rbf_svm": v_rbf}

            vote_counts = {}
            for v in votes.values():
                vote_counts[v] = vote_counts.get(v, 0) + 1

            max_count = max(vote_counts.values())
            if max_count == 3:
                predicted_digit = v_knn
                consensus_type = "Unanimous Consensus (3/3 Votes)"
                conf_val = 1.0
            elif max_count == 2:
                predicted_digit = [d for d, c in vote_counts.items() if c == 2][0]
                consensus_type = "Majority Consensus (2/3 Votes)"
                conf_val = 2.0 / 3.0
            else:
                predicted_digit = v_rbf
                consensus_type = "Split Decision (1/1/1 Tie — Resolved by RBF SVM)"
                conf_val = 1.0 / 3.0

            explanation = (
                f"Hard majority voting across 3 baseline models: k-NN predicted {v_knn}, "
                f"Linear SVM predicted {v_lin}, and RBF SVM predicted {v_rbf}. "
                f"Final decision: {consensus_type}."
            )

            prob_knn = np.array([pred_knn["all_scores"].get(str(i), 0.0) for i in range(10)])
            prob_lin = np.array([pred_lin["all_scores"].get(str(i), 0.0) for i in range(10)])
            prob_rbf = np.array([pred_rbf["all_scores"].get(str(i), 0.0) for i in range(10)])
            class_probabilities = (prob_knn + prob_lin + prob_rbf) / 3.0

    if target_model_key in ("ensemble", "ensemble_voting"):
        confidence = conf_val
    else:
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

    result = {
        "success": True,
        "model_key": target_model_key,
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "top_3": top_3,
        "all_scores": all_scores,
        "score_type": score_type,
        "explanation": explanation,
    }

    if target_model_key in ("ensemble", "ensemble_voting"):
        result["base_predictions"] = base_models_results
        result["votes"] = votes
        result["consensus_type"] = consensus_type

    return result
