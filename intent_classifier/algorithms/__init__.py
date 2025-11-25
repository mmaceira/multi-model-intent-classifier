"""Algorithms module for text classification.

This module provides various text classification algorithms including:
- Linear SVM classifiers
- Naive Bayes
- Embedding-based Logistic Regression (OpenAI and SBERT)
- Transformer-based Logistic Regression
"""

from .embedding_logreg import EmbeddingLogReg  # noqa: F401
from .linear_svm import LinearSVMBigrams, LinearSVMClassifier  # noqa: F401
from .naive_bayes import NaiveBayesClassifier  # noqa: F401
from .transformer_logreg import TransformerLogReg  # noqa: F401

__all__ = [
    "EmbeddingLogReg",
    "LinearSVMBigrams",
    "LinearSVMClassifier",
    "NaiveBayesClassifier",
    "TransformerLogReg",
]
