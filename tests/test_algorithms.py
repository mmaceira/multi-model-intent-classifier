"""Tests for all algorithm classes in `intent_classifier.algorithms`."""

from __future__ import annotations

import numpy as np
import pytest

from intent_classifier.algorithms import (
    LinearSVMBigrams,
    LinearSVMClassifier,
    NaiveBayesClassifier,
)


@pytest.fixture
def toy_corpus():
    X = [
        "buy groceries at the supermarket",
        "purchase food from the store",
        "play some jazz music",
        "listen to rock songs",
        "book a flight to London",
        "reserve an airplane ticket",
    ]
    y = ["shopping", "shopping", "music", "music", "travel", "travel"]
    return X, y


def _basic_fit_predict_checks(clf, X, y):
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(y))
    # If predict_proba exists, ensure shape and probabilities look sane
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X)
        assert probas.shape[0] == len(X)
        # Allow both binary and multi-class shapes; just ensure finite numbers
        assert np.isfinite(probas).all()


def test_naive_bayes_classifier(toy_corpus):
    X, y = toy_corpus
    clf = NaiveBayesClassifier(max_features=1000, alpha=0.5)
    _basic_fit_predict_checks(clf, X, y)


def test_linear_svm_classifier(toy_corpus):
    X, y = toy_corpus
    clf = LinearSVMClassifier(max_features=1000, C=0.1, calibrate=False)
    _basic_fit_predict_checks(clf, X, y)


def test_linear_svm_bigrams(toy_corpus):
    X, y = toy_corpus
    clf = LinearSVMBigrams(max_features=1000, C=0.1, calibrate=False)
    _basic_fit_predict_checks(clf, X, y)


@pytest.mark.skip(
    "TransformerLogReg depends on heavy SentenceTransformer models; "
    "covered indirectly via pipeline tests."
)
def test_transformer_logreg_tiny_corpus(toy_corpus):
    """Placeholder test to document why this path is not exercised directly."""
    assert True


@pytest.mark.skip(
    "EmbeddingLogReg (SBERT backend) is heavy; exercised via higher-level RAG + pipeline tests."
)
def test_embedding_logreg_sbert_backend(toy_corpus):
    """Placeholder test to document why this path is not exercised directly."""
    assert True
