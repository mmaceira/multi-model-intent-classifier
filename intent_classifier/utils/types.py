"""
Type definitions and Protocol interfaces for the intent classifier.

This module provides Protocol definitions for interfaces and type aliases
to improve type safety across the codebase.

Protocols:
    VectorizerProtocol: Interface for text vectorizers (must implement transform() or be callable)
    ClassifierProtocol: Interface for text classifiers (must implement fit() and predict())

These protocols enable structural typing, allowing any object that implements
the required methods to be used, regardless of its class hierarchy.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorizerProtocol(Protocol):
    """Protocol for text vectorizers.

    A vectorizer must implement either a transform() method or be callable.
    This protocol allows for both scikit-learn-style vectorizers and
    function-based vectorizers.
    """

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform text documents to feature vectors.

        Args:
            texts: Sequence of text documents to vectorize

        Returns:
            Array of feature vectors with shape (n_samples, n_features)
        """
        ...

    def fit(self, texts: Sequence[str]) -> None:
        """Fit the vectorizer on training data (optional).

        Some vectorizers (e.g., TF-IDF) need to learn vocabulary statistics
        from training data. This method is optional - if not present,
        the vectorizer is assumed to be stateless.

        Args:
            texts: Sequence of text documents for training
        """
        ...


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Protocol for text classifiers.

    This protocol defines the interface that all text classifiers must implement
    to be compatible with the intent classifier system.
    """

    def fit(self, X: Sequence[str], y: np.ndarray) -> "ClassifierProtocol":
        """Train the classifier on text data.

        Args:
            X: Sequence of raw text documents for training
            y: Array of target labels (shape depends on single-label vs multi-label)

        Returns:
            Self, to enable method chaining
        """
        ...

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Make predictions on text data.

        Args:
            X: Sequence of raw text documents to classify

        Returns:
            Array of predicted labels
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (scikit-learn compatibility).

        Args:
            deep: If True, return parameters of nested objects

        Returns:
            Dictionary mapping parameter names to values
        """
        ...

    def set_params(self, **params: Any) -> "ClassifierProtocol":
        """Set parameters for this estimator (scikit-learn compatibility).

        Args:
            **params: Estimator parameters to set

        Returns:
            Self, to enable method chaining
        """
        ...
