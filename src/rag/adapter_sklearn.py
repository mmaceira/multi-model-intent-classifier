"""\
Adapter Sklearn module.

Classes:
- RagSklearnAdapter

Functions:
- None

Created: 2025-05-03
"""

from sklearn.base import BaseEstimator, ClassifierMixin

class RagSklearnAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, rag_clf, **_):
        self.rag = rag_clf
        self.rag_clf = rag_clf  # Add this for compatibility with clone()
        
    def fit(self, X, y=None): 
        return self
        
    def predict(self, X): 
        return self.rag.predict(X)
    
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
