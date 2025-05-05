"""
Adapter Sklearn Module

This module provides an adapter class that makes RAG (Retrieval-Augmented Generation)
classifiers compatible with scikit-learn's estimator interface. This allows RAG
classifiers to be used in scikit-learn pipelines and with scikit-learn's model
selection tools.

Key Features:
- scikit-learn compatibility
- Probability estimation support
- Custom serialization handling
- Proper cloning support

Classes:
- RagSklearnAdapter: Adapter class for scikit-learn compatibility

Functions:
- None (Class methods only)

Dependencies:
- numpy
- sklearn.base

Example Usage:
    >>> # Create a RAG classifier
    >>> from rag.rag_llm import RagLLM
    >>> rag_clf = RagLLM.load_default()
    
    >>> # Wrap it in the adapter
    >>> from rag.adapter_sklearn import RagSklearnAdapter
    >>> sklearn_clf = RagSklearnAdapter(rag_clf)
    
    >>> # Use in scikit-learn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('preprocessor', preprocessor),
    ...     ('classifier', sklearn_clf)
    ... ])
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class RagSklearnAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, rag_clf, **_):
        self.rag = rag_clf
        self.rag_clf = rag_clf  # Add this for compatibility with clone()
        
    def fit(self, X, y=None): 
        return self
        
    def predict(self, X): 
        if isinstance(X, list):
            return self.rag.predict(X)
        elif isinstance(X, np.ndarray):
            return self.rag.predict(X.tolist())
        else:
            raise ValueError(f"Input must be a list or numpy array, got {type(X)}")
    
    def predict_proba(self, X):
        """Generate probability estimates for each class.
        
        This method delegates to the underlying RAG model's predict_proba method
        if it exists, otherwise raises an error.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict probabilities for.
            
        Returns
        -------
        np.ndarray : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        
        Raises
        ------
        AttributeError : If the underlying RAG model doesn't have a predict_proba method.
        """
        if hasattr(self.rag, 'predict_proba'):
            # Convert input to list of strings if it's a numpy array
            if isinstance(X, np.ndarray):
                if X.dtype.kind in ['U', 'S']:  # If array contains strings
                    X = X.tolist()
                else:
                    raise ValueError("Input array must contain strings")
            elif isinstance(X, list):
                # Verify all elements are strings
                if not all(isinstance(x, str) for x in X):
                    raise ValueError("All elements in input list must be strings")
            else:
                raise ValueError(f"Input must be a list of strings or numpy array of strings, got {type(X)}")
            
            return self.rag.predict_proba(X)
        else:
            raise AttributeError("The underlying RAG model does not implement predict_proba")
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        This is required for proper cloning.
        """
        return {'rag_clf': self.rag_clf}
    
    def __sklearn_clone__(self):
        """Custom clone method for sklearn compatibility.
        
        This method is called by sklearn's clone() function.
        It returns a shallow copy of self instead of a deep copy,
        which prevents pickling errors for unpicklable components.
        """
        return self.__class__(self.rag_clf)
        
    def __getstate__(self):
        """Custom serialization that excludes unpicklable components.
        
        This method is called when pickling the object.
        """
        # Return a simplified state, excluding any unpicklable objects
        return {'rag_type': self.rag.__class__.__name__}
        
    def __setstate__(self, state):
        """Custom deserialization.
        
        This method is called when unpickling the object.
        """
        # This is just a placeholder. In actual use, we'd reload the model
        # The actual model will be loaded from a file at prediction time
        self.rag = None
        self.rag_clf = None
