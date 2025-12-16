"""Algorithms module for text classification.

This module provides various text classification algorithms including:
- Linear SVM classifiers
- Naive Bayes
- Embedding-based Logistic Regression (OpenAI and SBERT)
- Transformer-based Logistic Regression
"""

from .embedding_logreg import EmbeddingLogReg
from .linear_svm import LinearSVMBigrams, LinearSVMClassifier
from .naive_bayes import NaiveBayesClassifier
from .transformer_logreg import TransformerLogReg

__all__ = [
    "EmbeddingLogReg",
    "LinearSVMBigrams",
    "LinearSVMClassifier",
    "NaiveBayesClassifier",
    "TransformerLogReg",
]
