"""
Unit tests for the Ensemble Voting Classifier and multi-model consensus engine.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

import config
from src.ensemble import EnsembleVotingClassifier, create_ensemble_from_base_models
from src.predict import predict_digit, load_model, are_models_available
from src.multi_digit import predict_multi_digit_number
from tests.test_multi_digit import _create_synthetic_number_image


class MockClassifier:
    """Mock classifier that always predicts a constant digit."""
    def __init__(self, pred_digit: int):
        self.pred_digit = pred_digit
        self.classes_ = np.arange(10)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.pred_digit, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), 10), dtype=np.float32)
        probs[:, self.pred_digit] = 1.0
        return probs


def test_ensemble_voting_unanimous():
    # 3 models all predict 7
    knn = MockClassifier(7)
    lin = MockClassifier(7)
    rbf = MockClassifier(7)

    ensemble = create_ensemble_from_base_models(knn, lin, rbf)
    X = np.zeros((5, 784), dtype=np.float32)
    preds = ensemble.predict(X)

    assert len(preds) == 5
    assert np.all(preds == 7)

    details = ensemble.predict_detailed_single(X[0])
    assert details["predicted_digit"] == 7
    assert details["agreement_count"] == 3
    assert "Unanimous" in details["consensus_type"]
    assert details["confidence"] == 1.0
    assert details["tie_broken"] is False


def test_ensemble_voting_majority():
    # 2 models predict 3, 1 model predicts 5
    knn = MockClassifier(3)
    lin = MockClassifier(5)
    rbf = MockClassifier(3)

    ensemble = create_ensemble_from_base_models(knn, lin, rbf)
    X = np.zeros((4, 784), dtype=np.float32)
    preds = ensemble.predict(X)

    assert np.all(preds == 3)

    details = ensemble.predict_detailed_single(X[0])
    assert details["predicted_digit"] == 3
    assert details["agreement_count"] == 2
    assert "Majority" in details["consensus_type"]
    assert np.isclose(details["confidence"], 2.0 / 3.0, atol=1e-3)
    assert details["tie_broken"] is False


def test_ensemble_voting_tie_breaker():
    # 3-way tie: knn=1, lin=4, rbf=9. Tie breaker defaults to rbf_svm -> 9
    knn = MockClassifier(1)
    lin = MockClassifier(4)
    rbf = MockClassifier(9)

    ensemble = create_ensemble_from_base_models(knn, lin, rbf, tie_breaker="rbf_svm")
    X = np.zeros((2, 784), dtype=np.float32)
    preds = ensemble.predict(X)

    assert np.all(preds == 9)

    details = ensemble.predict_detailed_single(X[0])
    assert details["predicted_digit"] == 9
    assert details["agreement_count"] == 1
    assert "Split Decision" in details["consensus_type"]
    assert details["tie_broken"] is True
    assert details["tie_breaker_model"] == "rbf_svm"


def test_ensemble_predict_proba():
    knn = MockClassifier(0)
    lin = MockClassifier(1)
    rbf = MockClassifier(0)

    ensemble = create_ensemble_from_base_models(knn, lin, rbf)
    X = np.zeros((3, 784), dtype=np.float32)
    probs = ensemble.predict_proba(X)

    assert probs.shape == (3, 10)
    # Digits 0 got 2 votes (2/3), digit 1 got 1 vote (1/3)
    assert np.allclose(probs[:, 0], 2.0 / 3.0)
    assert np.allclose(probs[:, 1], 1.0 / 3.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_ensemble_predict_digit_mocked(monkeypatch):
    # Mock load_model to return predictable models
    def mock_load(key):
        if key == "knn":
            return MockClassifier(4)
        elif key == "linear_svm":
            return MockClassifier(4)
        elif key == "rbf_svm":
            return MockClassifier(9)
        elif key in ("ensemble", "ensemble_voting"):
            return create_ensemble_from_base_models(MockClassifier(4), MockClassifier(4), MockClassifier(9))
        return None

    monkeypatch.setattr("src.predict.load_model", mock_load)

    dummy_input = np.zeros((1, 784), dtype=np.float32)
    res = predict_digit("ensemble", dummy_input)

    assert res["success"] is True
    assert res["predicted_digit"] == 4
    assert res["model_key"] == "ensemble"
    assert "votes" in res
    assert res["votes"]["knn"] == 4
    assert res["votes"]["linear_svm"] == 4
    assert res["votes"]["rbf_svm"] == 9
    assert len(res["top_3"]) <= 3
    assert len(res["all_scores"]) == 10


def test_ensemble_multi_digit_mocked(monkeypatch):
    def mock_load(key):
        return MockClassifier(2)

    monkeypatch.setattr("src.predict.load_model", mock_load)

    img = _create_synthetic_number_image("2026", width=400, height=120)
    res = predict_multi_digit_number(img, model_key="ensemble")

    assert res["success"] is True
    assert res["digit_count"] == 4
    assert res["predicted_number_str"] == "2222"
    assert res["model_key"] == "ensemble"
    assert res["annotated_image"] is not None


def test_ensemble_available_status():
    status = are_models_available()
    assert "ensemble" in status
    assert isinstance(status["ensemble"], bool)

