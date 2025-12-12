"""Tests for all algorithm classes in `intent_classifier.algorithms`."""

from __future__ import annotations

import numpy as np
import pytest

from intent_classifier.algorithms import (
    EmbeddingLogReg,
    LinearSVMBigrams,
    LinearSVMClassifier,
    NaiveBayesClassifier,
    TransformerLogReg,
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


def test_transformer_logreg_tiny_corpus(toy_corpus):
    """Test TransformerLogReg with a tiny corpus."""
    X, y = toy_corpus
    clf = TransformerLogReg(n_jobs=1, cv=2)
    _basic_fit_predict_checks(clf, X, y)


def test_embedding_logreg_sbert_backend(toy_corpus):
    """Test EmbeddingLogReg with SBERT backend."""
    X, y = toy_corpus
    # Use cv=2 since we only have 2 samples per class
    clf = EmbeddingLogReg(use_openai=False, model="sentence-transformers/all-MiniLM-L6-v2", cv=2)
    _basic_fit_predict_checks(clf, X, y)


@pytest.mark.skip("RAG models require embeddings/index files to be built first.")
def test_rag_kmajority(toy_corpus):
    """Test RAG-kMajority classifier."""
    from intent_classifier.rag import load_kmajority
    from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

    X, y = toy_corpus
    try:
        rag_model = load_kmajority(top_k=3, use_openai=False)
        clf = RagSklearnAdapter(rag_model)
        # RAG models don't need fit, but we can test predict
        preds = clf.predict(X)
        assert len(preds) == len(X)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"RAG-kMajority test skipped: embeddings not available ({e})")


@pytest.mark.skip("RAG models require embeddings/index files to be built first.")
def test_rag_centroid(toy_corpus):
    """Test RAG-CentroidNN classifier."""
    from intent_classifier.rag import load_centroid
    from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

    X, y = toy_corpus
    try:
        rag_model = load_centroid(top_k=3, use_openai=False)
        clf = RagSklearnAdapter(rag_model)
        # RAG models don't need fit, but we can test predict
        preds = clf.predict(X)
        assert len(preds) == len(X)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"RAG-CentroidNN test skipped: embeddings not available ({e})")


@pytest.mark.skip("RAG-LLM models require embeddings/index files and LLM availability.")
def test_rag_llm_local(toy_corpus):
    """Test RAG-LLM classifier with local embeddings."""
    from intent_classifier.rag import load_llm
    from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

    X, y = toy_corpus
    try:
        rag_model = load_llm(top_k=3, model="ollama/llama3.1:8b", use_openai=False)
        clf = RagSklearnAdapter(rag_model)
        # RAG models don't need fit, but we can test predict
        preds = clf.predict(X)
        assert len(preds) == len(X)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        pytest.skip(f"RAG-LLM test skipped: embeddings/LLM not available ({e})")
