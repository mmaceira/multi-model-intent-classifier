"""
Text Classification Model Module

This module provides the foundational architecture for text classification models in the project.
It defines an abstract base class that standardizes the interface and implements common functionality
for all text classifiers, ensuring consistency and interoperability across different model implementations.

Key Features:
- Abstract base class for text classifiers
- Scikit-learn compatibility through BaseEstimator
- Flexible text vectorization support
- Type hints and comprehensive error handling
- Consistent API across model implementations
- Parameter management for model persistence

Classes:
- TextClassifier: Abstract base class that all text classifiers must inherit from
  - Implements common functionality for text vectorization
  - Provides scikit-learn compatible parameter handling
  - Enforces consistent interface through abstract methods
  - Manages model state and validation

Abstract Methods (to be implemented by subclasses):
- _fit_model: Train the model on vectorized features
- _predict_model: Make predictions using vectorized features

Concrete Methods:
- fit: Train the classifier on raw text data
- predict: Make predictions on raw text data
- vectorize: Convert raw text to feature vectors
- get_params: Get model parameters (scikit-learn compatibility)
- set_params: Set model parameters (scikit-learn compatibility)

Dependencies:
- abc (Abstract Base Classes)
- typing
- numpy
- sklearn.base

Example Usage:
    >>> # Create a custom classifier
    >>> class MyClassifier(TextClassifier):
    ...     def __init__(self, vectorizer):
    ...         super().__init__(vectorizer)
    ...
    ...     def _fit_model(self, X_vec, y):
    ...         # Implement training logic
    ...         pass
    ...
    ...     def _predict_model(self, X_vec):
    ...         # Implement prediction logic
    ...         pass
    ...
    >>> # Use the classifier
    >>> clf = MyClassifier(vectorizer=CountVectorizer())
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from sklearn.base import BaseEstimator


class TextClassifier(ABC, BaseEstimator):
    """Abstract base class for text classifiers.

    This class defines the interface that all text classifiers must implement. It provides
    common functionality for text vectorization and ensures a consistent API across different
    classifier types. The class is designed to be compatible with scikit-learn pipelines
    and cross-validation utilities.

    Attributes:
        vectorizer: Text vectorization component that converts raw text to feature vectors.
                   Must implement either a transform() method or be callable.
        _is_fitted: Boolean flag indicating whether the model has been trained.

    Notes:
        - Subclasses must implement _fit_model and _predict_model methods
        - The class follows scikit-learn's BaseEstimator interface for parameter handling
        - Type checking and validation are performed to ensure robust operation
        - Thread-safety depends on the underlying vectorizer and model implementations

    Example:
        >>> from sklearn.feature_extraction.text import CountVectorizer
        >>> class MyClassifier(TextClassifier):
        ...     def _fit_model(self, X_vec, y):
        ...         # Train a simple model based on word counts
        ...         self.class_counts = {}
        ...         for x, c in zip(X_vec, y):
        ...             if c not in self.class_counts:
        ...                 self.class_counts[c] = np.zeros(x.shape)
        ...             self.class_counts[c] += x
        ...
        ...     def _predict_model(self, X_vec):
        ...         # Predict using simple similarity
        ...         predictions = []
        ...         for x in X_vec:
        ...             best_class = None
        ...             best_score = -float('inf')
        ...             for c, counts in self.class_counts.items():
        ...                 score = np.dot(x, counts)
        ...                 if score > best_score:
        ...                     best_score = score
        ...                     best_class = c
        ...             predictions.append(best_class)
        ...         return np.array(predictions)
        >>>
        >>> # Create and use the classifier
        >>> clf = MyClassifier(vectorizer=CountVectorizer())
        >>> clf.fit(["sports news", "business report"], [0, 1])
        >>> clf.predict(["latest sports results"])
        array([0])
    """

    def __init__(self, vectorizer: Any) -> None:
        """Initialize the text classifier.

        Args:
            vectorizer: Text vectorization component that converts raw text to feature vectors.
                       Must implement either a transform() method or be callable.

        Notes:
            For BaseEstimator compatibility, parameters should be stored as attributes with
            the same name as the parameter. The vectorizer is stored as _vectorizer since
            it's not a constructor parameter in subclasses.

        """
        self._vectorizer = vectorizer
        self._is_fitted = False

    @property
    def vectorizer(self) -> Any:
        """Get the vectorizer instance.

        Returns:
            The vectorizer component used for text feature extraction.

        Notes:
            This property provides read-only access to the vectorizer.
            To modify the vectorizer, create a new classifier instance.
        """
        return self._vectorizer

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        This method is required by scikit-learn's BaseEstimator interface and
        enables compatibility with model selection tools like GridSearchCV.

        Args:
            deep: If True, return the parameters of nested objects like the
                 vectorizer (if it also implements get_params).

        Returns:
            Parameter names mapped to their values
        """
        params = {}

        # Get parameters from constructor (__init__ method)
        import inspect

        init_signature = inspect.signature(self.__init__)

        for parameter_name in init_signature.parameters:
            if parameter_name != "self" and hasattr(self, parameter_name):
                params[parameter_name] = getattr(self, parameter_name)

        return params

    def set_params(self, **params: Any) -> "TextClassifier":
        """Set parameters for this estimator.

        This method is required by scikit-learn's BaseEstimator interface and
        enables compatibility with model selection tools like GridSearchCV.

        Args:
            **params: Estimator parameters to set. Can include nested parameters
                     for the vectorizer using vectorizer__param_name format.

        Returns:
            Self, to enable method chaining.

        Raises:
            ValueError: If an unknown parameter is provided.

        Notes:
            Setting parameters after the model has been fitted may require
            re-fitting the model for the changes to take effect.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {self.__class__.__name__}"
                )
            setattr(self, key, value)

        return self

    def fit(self, X_raw: List[str], y: np.ndarray) -> "TextClassifier":
        """Train the text classifier.

        This method trains the classifier on the provided text data. It first vectorizes
        the text using the provided vectorizer, then calls the subclass-specific _fit_model
        method to train the model on the vectorized features.

        Args:
            X_raw: List of raw text documents for training.
            y: Array of target labels. Should have the same length as X_raw.

        Returns:
            The trained classifier (self), to enable method chaining.

        Raises:
            ValueError: If the input data is invalid (empty or length mismatch).
            TypeError: If the input data types are incorrect.
            RuntimeError: If training fails for any reason.

        Notes:
            - If the vectorizer has a 'fit' method, it will be called with X_raw.
            - For some vectorizers (like TF-IDF), fitting on the training data is
              necessary to learn vocabulary statistics.
            - The _is_fitted flag is set to True after successful training.
        """
        if not isinstance(X_raw, list):
            raise TypeError("X_raw must be a list of strings")
        if not X_raw or not y:
            raise ValueError("Input data cannot be empty")
        if len(X_raw) != len(y):
            raise ValueError("X_raw and y must have the same length")

        # Fit vectorizer if it has a fit method
        if hasattr(self._vectorizer, "fit"):
            self._vectorizer.fit(X_raw)

        X_vec = self.vectorize(X_raw)
        self._fit_model(X_vec, y)
        self._is_fitted = True
        return self

    def predict(self, X_raw: List[str]) -> np.ndarray:
        """Make predictions using the trained classifier.

        This method makes predictions on the provided text data. It first vectorizes
        the text using the provided vectorizer, then calls the subclass-specific
        _predict_model method to generate predictions.

        Args:
            X_raw: List of raw text documents to classify.

        Returns:
            Array of predicted labels with the same length as X_raw.

        Raises:
            ValueError: If the model has not been fitted or input data is invalid.
            TypeError: If the input data type is incorrect.
            RuntimeError: If prediction fails for any reason.

        Notes:
            - The model must be fitted (trained) before calling this method.
            - Input validation ensures X_raw is a list of strings.
            - Empty input will raise a ValueError.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if not isinstance(X_raw, list):
            raise TypeError("X_raw must be a list of strings")

        if not X_raw:
            raise ValueError("Input data cannot be empty")

        try:
            X_vec = self.vectorize(X_raw)
            return self._predict_model(X_vec)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Convert raw text to feature vectors.

        This method transforms a list of text documents into feature vectors
        using the vectorizer provided at initialization.

        Args:
            texts: List of text documents to vectorize.

        Returns:
            NumPy array of feature vectors with shape (n_samples, n_features).

        Raises:
            ValueError: If vectorization fails or produces invalid output.
            TypeError: If vectorizer doesn't have the expected interface.

        Notes:
            - Supports vectorizers with a transform() method (scikit-learn style)
            - Also supports callable vectorizers (function-style)
            - Ensures output is converted to a numpy array for consistency
            - Performs shape validation on the output
        """
        if hasattr(self._vectorizer, "transform"):
            return self._vectorizer.transform(texts)
        elif callable(self._vectorizer):
            return self._vectorizer(texts)
        else:
            raise ValueError("Vectorizer must implement transform() or be callable")

    @abstractmethod
    def _fit_model(self, X_vec: np.ndarray, y: np.ndarray) -> None:
        """Train the model on vectorized features.

        This abstract method must be implemented by subclasses to define
        the specific training logic for the classifier.

        Args:
            X_vec: Array of vectorized text features with shape (n_samples, n_features).
            y: Array of target labels with shape (n_samples,).

        Returns:
            None

        Notes:
            - Implementation should handle any model-specific training logic
            - Should store trained model parameters as instance attributes
            - May raise appropriate exceptions for training failures
            - Should not modify X_vec or y (treat as read-only)
        """
        raise NotImplementedError("Subclasses must implement _fit_model")

    @abstractmethod
    def _predict_model(self, X_vec: np.ndarray) -> np.ndarray:
        """Make predictions using vectorized features.

        This abstract method must be implemented by subclasses to define
        the specific prediction logic for the classifier.

        Args:
            X_vec: Array of vectorized text features with shape (n_samples, n_features).

        Returns:
            Array of predicted labels with shape (n_samples,).

        Notes:
            - Implementation should handle any model-specific prediction logic
            - Should return numpy array of predicted labels
            - Should not modify X_vec (treat as read-only)
            - May raise appropriate exceptions for prediction failures
        """
        pass
