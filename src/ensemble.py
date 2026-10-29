"""
Ensemble Learning Module for Handwritten Digit Recognition.

Implements an Ensemble Voting Classifier combining:
1. k-Nearest Neighbors (k-NN) - Instance-based spatial voting
2. Linear Support Vector Machine (Linear SVM) - Calibrated maximum-margin hyperplane
3. Radial Basis Function Support Vector Machine (RBF SVM) - Non-linear kernel mapping

Supports Hard Majority Voting with deterministic tie-breaking, comprehensive vote breakdown,
and full scikit-learn API compatibility (BaseEstimator, ClassifierMixin).
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import config


class EnsembleVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Hard Majority-Voting Ensemble Classifier combining k-NN, Linear SVM, and RBF SVM.

    Attributes:
        estimators: List of (name, estimator) tuples.
        voting: Voting strategy (defaults to 'hard').
        tie_breaker: Model key used to resolve 1-1-1 three-way ties (default: 'rbf_svm').
        classes_: Array of class labels (0 to 9).
    """

    def __init__(
        self,
        estimators: Optional[List[Tuple[str, Any]]] = None,
        voting: str = "hard",
        tie_breaker: str = "rbf_svm",
    ):
        self.estimators = estimators or []
        self.voting = voting
        self.tie_breaker = tie_breaker
        self.classes_ = np.arange(config.NUM_CLASSES)
        self.named_estimators_ = dict(self.estimators) if self.estimators else {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleVotingClassifier":
        """
        Fits all underlying base estimators on training data if not already fitted.
        """
        self.classes_ = np.unique(y)
        self.named_estimators_ = {}
        fitted_estimators = []

        for name, est in self.estimators:
            # Fit estimator if not already fitted
            try:
                check_is_fitted(est)
            except Exception:
                est.fit(X, y)
            fitted_estimators.append((name, est))
            self.named_estimators_[name] = est

        self.estimators = fitted_estimators
        self.estimators_ = [est for _, est in self.estimators]
        return self

    def _get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Collects raw predictions from each base estimator."""
        preds = {}
        for name, est in self.named_estimators_.items():
            preds[name] = np.asarray(est.predict(X))
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs hard majority voting over the base estimators.

        For each sample:
        - If 2 or 3 models agree, predict the majority class.
        - If all 3 models predict different classes (1-1-1 tie), break the tie
          using the tie_breaker estimator (RBF SVM by default, having the highest individual test accuracy).
        """
        if not self.named_estimators_:
            raise RuntimeError("Ensemble has no estimators configured.")

        base_preds = self._get_base_predictions(X)
        names = list(base_preds.keys())
        n_samples = len(X)
        final_predictions = np.zeros(n_samples, dtype=int)

        pred_matrix = np.column_stack([base_preds[name] for name in names])

        for i in range(n_samples):
            votes = pred_matrix[i]
            unique_classes, counts = np.unique(votes, return_counts=True)
            max_count = np.max(counts)

            if max_count >= 2:
                # Clear majority (either unanimous 3-0 or majority 2-1)
                majority_class = unique_classes[np.argmax(counts)]
                final_predictions[i] = majority_class
            else:
                # 3-way tie (1-1-1): Use tie_breaker classifier
                if self.tie_breaker in self.named_estimators_:
                    final_predictions[i] = base_preds[self.tie_breaker][i]
                else:
                    # Fallback to the first estimator
                    final_predictions[i] = votes[0]

        return final_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes consensus vote distribution (fraction of base models voting for each class)
        or normalized vote proportions.
        """
        n_samples = len(X)
        n_classes = len(self.classes_)
        proportions = np.zeros((n_samples, n_classes), dtype=np.float32)

        base_preds = self._get_base_predictions(X)
        names = list(base_preds.keys())
        n_estimators = len(names)

        pred_matrix = np.column_stack([base_preds[name] for name in names])

        for i in range(n_samples):
            for c in pred_matrix[i]:
                if 0 <= c < n_classes:
                    proportions[i, c] += 1.0 / n_estimators

        return proportions

    def predict_detailed_single(
        self,
        flattened_784: np.ndarray,
        base_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Provides full diagnostic consensus details for a single digit sample.

        Returns:
            Dictionary containing:
            - predicted_digit (int)
            - agreement_count (int: 3, 2, or 1)
            - consensus_type ("Unanimous (3/3)", "Majority (2/3)", or "Split Decision (1/1/1)")
            - votes (Dict[model_key, predicted_digit])
            - vote_counts (Dict[str, int])
            - tie_broken (bool)
            - tie_breaker_model (str or None)
        """
        if flattened_784.ndim == 1:
            flattened_784 = flattened_784.reshape(1, -1)

        votes: Dict[str, int] = {}
        for name, est in self.named_estimators_.items():
            pred = int(est.predict(flattened_784)[0])
            votes[name] = pred

        # Count votes per digit
        vote_counts: Dict[int, int] = {}
        for d in votes.values():
            vote_counts[d] = vote_counts.get(d, 0) + 1

        max_count = max(vote_counts.values()) if vote_counts else 0
        tie_broken = False
        tie_breaker_model = None

        if max_count == 3:
            predicted_digit = [d for d, c in vote_counts.items() if c == 3][0]
            consensus_type = "Unanimous Consensus (3/3 Votes)"
            confidence = 1.0
        elif max_count == 2:
            predicted_digit = [d for d, c in vote_counts.items() if c == 2][0]
            consensus_type = "Majority Consensus (2/3 Votes)"
            confidence = 2.0 / 3.0
        else:
            # 1-1-1 tie
            tie_broken = True
            tie_breaker_model = self.tie_breaker if self.tie_breaker in votes else list(votes.keys())[0]
            predicted_digit = votes[tie_breaker_model]
            consensus_type = f"Split Decision (1/1/1 Tie — Resolved by {tie_breaker_model})"
            confidence = 1.0 / 3.0

        return {
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "agreement_count": max_count,
            "consensus_type": consensus_type,
            "votes": votes,
            "vote_counts": {str(k): v for k, v in sorted(vote_counts.items())},
            "tie_broken": tie_broken,
            "tie_breaker_model": tie_breaker_model,
        }


def create_ensemble_from_base_models(
    knn_model: Any,
    linear_svm_model: Any,
    rbf_svm_model: Any,
    tie_breaker: str = "rbf_svm",
) -> EnsembleVotingClassifier:
    """
    Constructs an EnsembleVotingClassifier from three pre-fitted base models.
    """
    estimators = [
        ("knn", knn_model),
        ("linear_svm", linear_svm_model),
        ("rbf_svm", rbf_svm_model),
    ]
    ensemble = EnsembleVotingClassifier(estimators=estimators, voting="hard", tie_breaker=tie_breaker)
    ensemble.classes_ = np.arange(config.NUM_CLASSES)
    ensemble.named_estimators_ = dict(estimators)
    ensemble.estimators_ = [est for _, est in estimators]
    return ensemble

