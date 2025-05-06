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
from typing import List, Any, Dict, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator


class TextClassifier(ABC, BaseEstimator):
    """Abstract base class for text classifiers.
    
    This class defines the interface that all text classifiers must implement. It provides
    common functionality for text vectorization and ensures a consistent API across different
    classifier types.
    
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
    
    def __init__(self, vectorizer: Any) -> None:
        """Initialize the text classifier.
        
        Args:
            vectorizer: Text vectorization component that converts raw text to feature vectors.
                       Must implement either a transform() method or be callable.
        
        Note:
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
        """
        return self._vectorizer
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        This method is required by scikit-learn's BaseEstimator interface.
        
        Args:
            deep: If True, return the parameters of nested objects
            
        Returns:
            Parameter names mapped to their values
        """
        params = {}
        
        # Get parameters from constructor (__init__ method)
        import inspect
        init_signature = inspect.signature(self.__init__)
        
        for parameter_name in init_signature.parameters:
            if parameter_name != 'self' and hasattr(self, parameter_name):
                params[parameter_name] = getattr(self, parameter_name)
                    
        return params
    
    def set_params(self, **params: Any) -> 'TextClassifier':
        """Set parameters for this estimator.
        
        This method is required by scikit-learn's BaseEstimator interface.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            Self
            
        Raises:
            ValueError: If any parameter is invalid
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
            setattr(self, key, value)
                    
        return self

    def fit(self, X_raw: List[str], y: np.ndarray) -> 'TextClassifier':
        """Train the text classifier.
        
        This method trains the classifier on the provided text data. It first vectorizes
        the text using the provided vectorizer, then calls the subclass-specific _fit_model
        method.
        
        Args:
            X_raw: List of raw text documents
            y: Array of labels
            
        Returns:
            The trained classifier (self)
            
        Raises:
            ValueError: If the input data is invalid
            TypeError: If the input data types are incorrect
        """
        if not isinstance(X_raw, list):
            raise TypeError("X_raw must be a list of strings")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
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
        _predict_model method.
        
        Args:
            X_raw: List of raw text documents
            
        Returns:
            Array of predicted labels
            
        Raises:
            RuntimeError: If the model hasn't been trained
            ValueError: If the input data is invalid
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        if not isinstance(X_raw, list):
            raise TypeError("X_raw must be a list of strings")
        if not X_raw:
            raise ValueError("Input data cannot be empty")
            
        X_vec = self.vectorize(X_raw)
        return self._predict_model(X_vec)

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Convert raw text documents to feature vectors.
        
        This method uses the provided vectorizer to convert raw text documents into
        numerical feature vectors that can be used by the classifier.
        
        Args:
            texts: List of raw text documents
            
        Returns:
            Array of feature vectors
            
        Raises:
            ValueError: If the vectorizer doesn't support transformation
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
        
        This method must be implemented by subclasses to provide model-specific
        training logic.
        
        Args:
            X_vec: Array of feature vectors
            y: Array of labels
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _fit_model")

    @abstractmethod
    def _predict_model(self, X_vec: np.ndarray) -> np.ndarray:
        """Make predictions using vectorized features.
        
        This method must be implemented by subclasses to provide model-specific
        prediction logic.
        
        Args:
            X_vec: Array of feature vectors
            
        Returns:
            Array of predicted labels
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _predict_model")
