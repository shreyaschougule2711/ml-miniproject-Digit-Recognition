"""
Unit tests for model prediction and inference engine.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

import config
from src.predict import are_models_available, predict_digit, softmax


def test_softmax_properties():
    logits = np.array([1.0, 2.0, 3.0, 0.0])
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0.0)
    # Highest logit should have highest probability
    assert np.argmax(probs) == 2


def test_are_models_available_keys():
    status = are_models_available()
    assert "knn" in status
    assert "linear_svm" in status
    assert "rbf_svm" in status
    assert "ensemble" in status
    assert all(isinstance(v, bool) for v in status.values())


def test_predict_digit_missing_model():
    dummy_input = np.zeros((1, 784), dtype=np.float32)
    result = predict_digit("non_existent_model_key", dummy_input)
    assert result["success"] is False
    assert "error" in result


def test_predict_digit_with_mock(monkeypatch):
    # Mock load_model to return a fitted DummyClassifier
    dummy = DummyClassifier(strategy="constant", constant=7)
    dummy.fit(np.zeros((10, 784)), np.full(10, 7))

    monkeypatch.setattr("src.predict.load_model", lambda key: dummy)

    dummy_input = np.zeros((1, 784), dtype=np.float32)
    res = predict_digit("knn", dummy_input)

    assert res["success"] is True
    assert res["predicted_digit"] == 7
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_3"]) <= 3
    assert len(res["all_scores"]) == 10
