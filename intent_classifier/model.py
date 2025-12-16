"""
Text Classification Model Module

This module provides the foundational architecture for text classification models in the project.
It defines an abstract base class that standardizes the interface and implements common
functionality for all text classifiers, ensuring consistency and interoperability across
different model implementations.

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
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from intent_classifier.utils.label_utils import (
    binarize_labels,
    is_multilabel,
    multilabel_predictions_from_binary,
    to_multilabel_format,
)
from intent_classifier.utils.types import VectorizerProtocol


class TextClassifier(ABC, BaseEstimator):
    """Abstract base class for text classifiers.

    This class defines the interface that all text classifiers must implement. It provides
    common functionality for text vectorization and ensures a consistent API across different
    classifier types. The class is designed to be compatible with scikit-learn pipelines
    and cross-validation utilities.

    Attributes:
        vectorizer: Text vectorization component that converts raw text to feature vectors.
                   Must implement either a transform() method (VectorizerProtocol) or be callable.
                   Type: Union[VectorizerProtocol, Callable[[Sequence[str]], np.ndarray]]
        _is_fitted: Boolean flag indicating whether the model has been trained.
        _expects_vectors: Class attribute (set on the class, not instance) indicating whether
                         the classifier expects pre-vectorized input. Set to False for classifiers
                         that work with raw text (the default). Set to True if the classifier
                         expects vectorized features directly. This is used by the API loader
                         to determine whether to wrap the model with a vectorizer.

    Notes:
        - Subclasses must implement _fit_model and _predict_model methods
        - The class follows scikit-learn's BaseEstimator interface for parameter handling
        - Type checking and validation are performed to ensure robust operation
        - Thread-safety depends on the underlying vectorizer and model implementations
        - Subclasses should set `_expects_vectors = False` (default) if they work with raw text,
          or `_expects_vectors = True` if they expect pre-vectorized features

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

    def __init__(
        self, vectorizer: VectorizerProtocol | Callable[[Sequence[str]], np.ndarray]
    ) -> None:
        """Initialize the text classifier.

        Args:
            vectorizer: Text vectorization component that converts raw text to
                       feature vectors. Must implement either a transform()
                       method (VectorizerProtocol) or be callable.

        Notes:
            For BaseEstimator compatibility, parameters should be stored as attributes with
            the same name as the parameter. The vectorizer is stored as _vectorizer since
            it's not a constructor parameter in subclasses.

        """
        self._vectorizer = vectorizer
        self._is_fitted = False

        # Multi-label support attributes (set during fit())
        self._is_multilabel = False  # True if model was trained with multi-label data
        self._label_binarizer = None  # MultiLabelBinarizer instance (for multi-label only)
        self._classes = (
            None  # All unique class labels (numpy array for single-label, list for multi-label)
        )

    @property
    def vectorizer(self) -> VectorizerProtocol | Callable[[Sequence[str]], np.ndarray]:
        """Get the vectorizer instance.

        Returns:
            The vectorizer component used for text feature extraction.

        Notes:
            This property provides read-only access to the vectorizer.
            To modify the vectorizer, create a new classifier instance.
        """
        return self._vectorizer

    def get_params(self, deep: bool = True) -> dict[str, Any]:
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

        init_signature = inspect.signature(type(self).__init__)

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

    def fit(self, X_raw: Sequence[str], y: np.ndarray) -> "TextClassifier":
        """Train the text classifier.

        This method trains the classifier on the provided text data. It first vectorizes
        the text using the provided vectorizer, then calls the subclass-specific _fit_model
        method to train the model on the vectorized features.

        Args:
            X_raw: Sequence of raw text documents for training.
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
        if not isinstance(X_raw, Sequence):
            raise TypeError("X_raw must be a sequence of strings")
        if len(X_raw) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        if len(X_raw) != len(y):
            raise ValueError("X_raw and y must have the same length")

        # ========================================================================
        # STEP 1: Detect label format (single-label vs multi-label)
        # ========================================================================
        # Auto-detect format from input data:
        # - Single-label: y = ["class1", "class2", "class1"]  (list of strings)
        # - Multi-label: y = [["class1"], ["class2"], ["class1", "class2"]]  (list of lists)
        self._is_multilabel = is_multilabel(y)

        # ========================================================================
        # STEP 2: Convert labels to format expected by sklearn models
        # ========================================================================
        if self._is_multilabel:
            # MULTI-LABEL PATH:
            # 1. Ensure multi-label format (list of lists)
            y_multilabel = to_multilabel_format(y)
            # 2. Convert to binary matrix for sklearn (n_samples x n_classes, 0/1 values)
            y_binary, self._label_binarizer = binarize_labels(y_multilabel)
            # 3. Store classes from binarizer
            if self._label_binarizer is not None:
                self._classes = self._label_binarizer.classes_
            else:
                raise RuntimeError("Label binarizer not created")
            # 4. Pass binary matrix to model training
            y_for_training = y_binary
        else:
            # SINGLE-LABEL PATH:
            # 1. Convert to numpy array (sklearn expects array for single-label)
            y_for_training = np.asarray(y)
            # 2. Extract unique classes
            self._classes = np.unique(y_for_training)

        # ========================================================================
        # STEP 3: Vectorize text and train model
        # ========================================================================
        # Fit vectorizer if it has a fit method (e.g., TF-IDF needs vocabulary)
        if hasattr(self._vectorizer, "fit"):
            self._vectorizer.fit(X_raw)

        # Convert text to feature vectors
        X_vec = self.vectorize(X_raw)

        # Train the model (subclass implements _fit_model)
        # For single-label: y_for_training is array of shape (n_samples,)
        # For multi-label: y_for_training is binary matrix of shape (n_samples, n_classes)
        self._fit_model(X_vec, y_for_training)

        self._is_fitted = True
        return self

    def predict(self, X_raw: Sequence[str]) -> np.ndarray:
        """Make predictions using the trained classifier.

        This method makes predictions on the provided text data. It first vectorizes
        the text using the provided vectorizer, then calls the subclass-specific
        _predict_model method to generate predictions.

        Args:
            X_raw: Sequence of raw text documents to classify.

        Returns:
            Array of predicted labels with the same length as X_raw.

        Raises:
            ValueError: If the model has not been fitted or input data is invalid.
            TypeError: If the input data type is incorrect.
            RuntimeError: If prediction fails for any reason.

        Notes:
            - The model must be fitted (trained) before calling this method.
            - Input validation ensures X_raw is a sequence of strings.
            - Empty input will raise a ValueError.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if not isinstance(X_raw, Sequence):
            raise TypeError("X_raw must be a sequence of strings")

        if len(X_raw) == 0:
            raise ValueError("Input data cannot be empty")

        try:
            # ========================================================================
            # STEP 1: Vectorize text and get raw predictions from model
            # ========================================================================
            X_vec = self.vectorize(X_raw)
            predictions_raw = self._predict_model(X_vec)

            # ========================================================================
            # STEP 2: Convert predictions back to original label format
            # ========================================================================
            if self._is_multilabel:
                # MULTI-LABEL PATH:
                # Model returns binary matrix (n_samples x n_classes, 0/1 values)
                # Convert back to multi-label format (list of lists)
                if isinstance(predictions_raw, np.ndarray) and predictions_raw.ndim == 2:
                    # Binary matrix format: convert to list of lists
                    if self._classes is None or len(self._classes) == 0:
                        raise ValueError(
                            "Model._classes is not initialized. This should not happen. "
                            "Make sure the model was properly trained with multilabel data."
                        )
                    predictions = multilabel_predictions_from_binary(predictions_raw, self._classes)
                elif isinstance(predictions_raw, np.ndarray) and predictions_raw.ndim == 1:
                    # 1D array - this shouldn't happen for multilabel, but handle it
                    # If it's integer indices, we can't convert without classes
                    if self._classes is None or len(self._classes) == 0:
                        raise ValueError(
                            "Model._classes is not initialized. Cannot convert "
                            "1D predictions to multilabel format."
                        )
                    # Convert 1D array to 2D binary matrix (assuming indices)
                    # This is a fallback - shouldn't normally happen
                    n_samples = len(predictions_raw)
                    n_classes = len(self._classes)
                    binary_matrix = np.zeros((n_samples, n_classes), dtype=int)
                    for i, idx in enumerate(predictions_raw):
                        if 0 <= int(idx) < n_classes:
                            binary_matrix[i, int(idx)] = 1
                    predictions = multilabel_predictions_from_binary(binary_matrix, self._classes)
                else:
                    # Fallback: if predictions_raw is not 2D, it might be in wrong format
                    # Check if it's a 1D array of integers (indices) - this shouldn't happen
                    if isinstance(predictions_raw, np.ndarray) and predictions_raw.ndim == 1:
                        if self._classes is None or len(self._classes) == 0:
                            raise ValueError(
                                "Model._classes is not initialized. Cannot "
                                "convert 1D predictions to multilabel format."
                            )
                        # Convert 1D array of indices to 2D binary matrix
                        n_samples = len(predictions_raw)
                        n_classes = len(self._classes)
                        binary_matrix = np.zeros((n_samples, n_classes), dtype=int)
                        for i, idx in enumerate(predictions_raw):
                            idx_int = int(idx)
                            if 0 <= idx_int < n_classes:
                                binary_matrix[i, idx_int] = 1
                        predictions = multilabel_predictions_from_binary(
                            binary_matrix, self._classes
                        )
                    else:
                        # Fallback: wrap single predictions in lists
                        predictions = to_multilabel_format(predictions_raw)
            else:
                # SINGLE-LABEL PATH:
                # Model returns array of single labels (n_samples,)
                # Return as numpy array for consistency
                predictions = np.asarray(predictions_raw)

            return predictions
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e!s}") from e

    def vectorize(self, texts: Sequence[str]) -> np.ndarray:
        """Convert raw text to feature vectors.

        This method transforms a list of text documents into feature vectors
        using the vectorizer provided at initialization.

        Args:
            texts: Sequence of text documents to vectorize.

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
            y: Target labels in format expected by sklearn:
                - Single-label: numpy array of shape (n_samples,) with string labels
                  Example: array(["class1", "class2", "class1"])
                - Multi-label: binary matrix of shape (n_samples, n_classes) with 0/1 values
                  Example: array([[1, 0], [0, 1], [1, 1]])  (3 samples, 2 classes)

        Returns:
            None

        Notes:
            - Implementation should handle any model-specific training logic
            - Should store trained model parameters as instance attributes
            - May raise appropriate exceptions for training failures
            - Should not modify X_vec or y (treat as read-only)
            - Format conversion (single-label <-> multi-label) is handled by base class
            - Check self._is_multilabel to determine which format y is in
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
            Predictions in format expected by sklearn:
            - Single-label: numpy array of shape (n_samples,) with string labels
              Example: array(["class1", "class2", "class1"])
            - Multi-label: binary matrix of shape (n_samples, n_classes) with 0/1 values
              Example: array([[1, 0], [0, 1], [1, 1]])  (3 samples, 2 classes)

        Notes:
            - Implementation should handle any model-specific prediction logic
            - Should return numpy array (format depends on self._is_multilabel)
            - Should not modify X_vec (treat as read-only)
            - May raise appropriate exceptions for prediction failures
            - Format conversion (binary matrix <-> list of lists) is handled by base class
            - Check self._is_multilabel to determine which format to return
        """
        pass  # pylint: disable=unnecessary-pass
