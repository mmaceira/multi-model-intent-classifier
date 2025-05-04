"""\
Model module for text classification.

This module provides the base text classifier interface and common
functionality for all text classification models in the project.
It defines the abstract base class that all classifiers must implement.

Classes:
- TextClassifier: Abstract base class for text classifiers

Functions:
- None

Created: 2025-05-03
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator

class TextClassifier(ABC, BaseEstimator):
    """Abstract base class for text classifiers.
    
    This class defines the interface that all text classifiers must
    implement. It provides common functionality for text vectorization
    and ensures a consistent API across different classifier types.
    
    Attributes:
        vectorizer: Text vectorization component
        _is_fitted: Whether the model has been trained
        
    Example:
        >>> class MyClassifier(TextClassifier):
        ...     def _fit_model(self, X_vec, y):
        ...         # Implement model training
        ...         pass
        ...     def _predict_model(self, X_vec):
        ...         # Implement prediction
        ...         pass
    """
    def __init__(self, vectorizer: Any):
        """Initialize the text classifier.
        
        Args:
            vectorizer: Text vectorization component
        """
        # NOTE: For BaseEstimator compatibility, parameters should be stored
        # as attributes with the same name as the parameter.
        # Don't store vectorizer directly, as it's not a constructor parameter in subclasses.
        self._vectorizer = vectorizer
        self._is_fitted = False
    
    @property
    def vectorizer(self):
        """Get the vectorizer."""
        return self._vectorizer
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        This method is required by scikit-learn's BaseEstimator interface.
        
        Args:
            deep: If True, return the parameters of nested objects
            
        Returns:
            Parameter names mapped to their values
        """
        # Only include params that are actually constructor parameters
        # The BaseEstimator looks at constructor signature, not at instance attributes
        params = {}
        
        # Get parameters from constructor (__init__ method)
        import inspect
        init_signature = inspect.signature(self.__init__)
        
        for parameter_name in init_signature.parameters:
            # Skip 'self' parameter
            if parameter_name != 'self':
                if hasattr(self, parameter_name):
                    params[parameter_name] = getattr(self, parameter_name)
                    
        return params
    
    def set_params(self, **params: Any) -> 'TextClassifier':
        """Set parameters for this estimator.
        
        This method is required by scikit-learn's BaseEstimator interface.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            Self
        """
        for key, value in params.items():
            setattr(self, key, value)
                    
        return self

    def fit(self, X_raw: List[str], y: np.ndarray) -> 'TextClassifier':
        """Train the text classifier.
        
        This method trains the classifier on the provided text data.
        It first vectorizes the text using the provided vectorizer,
        then calls the subclass-specific _fit_model method.
        
        Args:
            X_raw: List of raw text documents
            y: Array of labels
            
        Returns:
            The trained classifier (self)
            
        Raises:
            ValueError: If the input data is invalid
        """
        if not X_raw or not y:
            raise ValueError("Input data cannot be empty")
            
        # Fit vectorizer if it has a fit method
        if hasattr(self._vectorizer, "fit"):
            self._vectorizer.fit(X_raw)
            
        X_vec = self.vectorize(X_raw)
        self._fit_model(X_vec, y)
        self._is_fitted = True
        return self

    def predict(self, X_raw: List[str]) -> np.ndarray:
        """Make predictions using the trained classifier.
        
        This method makes predictions on the provided text data.
        It first vectorizes the text using the provided vectorizer,
        then calls the subclass-specific _predict_model method.
        
        Args:
            X_raw: List of raw text documents
            
        Returns:
            Array of predicted labels
            
        Raises:
            RuntimeError: If the model hasn't been trained
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_vec = self.vectorize(X_raw)
        return self._predict_model(X_vec)

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Convert raw text documents to feature vectors.
        
        This method uses the provided vectorizer to convert raw text
        documents into numerical feature vectors that can be used by
        the classifier.
        
        Args:
            texts: List of raw text documents
            
        Returns:
            Array of feature vectors
        """
        if hasattr(self._vectorizer, "transform"):
            return self._vectorizer.transform(texts)
        else:
            return self._vectorizer(texts)

    @abstractmethod
    def _fit_model(self, X_vec: np.ndarray, y: np.ndarray) -> None:
        """Train the model on vectorized features.
        
        This method must be implemented by subclasses to provide
        model-specific training logic.
        
        Args:
            X_vec: Array of feature vectors
            y: Array of labels
        """
        pass

    @abstractmethod
    def _predict_model(self, X_vec: np.ndarray) -> np.ndarray:
        """Make predictions using vectorized features.
        
        This method must be implemented by subclasses to provide
        model-specific prediction logic.
        
        Args:
            X_vec: Array of feature vectors
            
        Returns:
            Array of predicted labels
        """
        pass
