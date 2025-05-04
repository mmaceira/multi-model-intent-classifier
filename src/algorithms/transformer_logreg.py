"""\
Transformer Logistic Regression module for text classification.

This module implements a text classifier using transformer-based embeddings
and logistic regression. It leverages pre-trained language models for
high-quality text representations.

Classes:
- TransformerLogReg: Transformer embeddings + Logistic Regression classifier

Functions:
- None

Created: 2025-05-03
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import logging
from src.model import TextClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class TransformerLogReg(TextClassifier):
    _expects_vectors = False
    """Transformer embeddings + Logistic Regression classifier.
    
    This class implements a text classifier using transformer-based
    embeddings and logistic regression. It uses pre-trained language
    models to generate high-quality text representations, which are
    then classified using logistic regression.
    
    Attributes:
        model_name: Name of the pre-trained transformer model
        C: Logistic regression regularization parameter
        clf: LogisticRegression classifier instance
        
    Example:
        >>> clf = TransformerLogReg(model_name='all-MiniLM-L6-v2', C=1.0)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', C: float = 1.0):
        """Initialize the classifier.
        
        Args:
            model_name: Name of the pre-trained transformer model (default: 'all-MiniLM-L6-v2')
            C: Logistic regression regularization parameter (default: 1.0)
        """
        # Store constructor parameters as attributes for BaseEstimator
        self.model_name = model_name
        self.C = C
        
        # Initialize base class with the embedder as the vectorizer
        self.embedder = SentenceTransformer(model_name)
        super().__init__(self.embedder)
        
        # Initialize classifier
        self.clf = LogisticRegression(C=C, max_iter=1000)
        self._is_fitted = False
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        This method is required by scikit-learn's BaseEstimator interface.
        
        Args:
            deep: If True, return the parameters of nested objects
            
        Returns:
            Parameter names mapped to their values
        """
        return {'model_name': self.model_name, 'C': self.C}

    def vectorize(self, texts):
        """Convert raw text to transformer embeddings.
        
        Args:
            texts: List of raw text documents
            
        Returns:
            Numpy array of document embeddings
        """
        return self.embedder.encode(texts)

    def _fit_model(self, X_vec, y):
        """Train the logistic regression classifier.
        
        Args:
            X_vec: Vectorized text features (transformer embeddings)
            y: Labels
        """
        self.clf.fit(X_vec, y)
        self._is_fitted = True

    def _predict_model(self, X_vec):
        """Make predictions using the trained classifier.
        
        Args:
            X_vec: Vectorized text features (transformer embeddings)
            
        Returns:
            List of predicted labels
        """
        return self.clf.predict(X_vec)

    # ------------------------------------------------------------------ #
    # Compatibility helpers expected by evaluation.py                     #
    # ------------------------------------------------------------------ #
    def fit(self, X: List[str], y: List[str]):
        """Encode X and fit multinomial logistic regression."""
        logger.info("Encoding %d documents for training", len(X))
        X_vec = self.vectorize(X)
        self._fit_model(X_vec, y)
        return self  # important for evaluate()

    def predict(self, X: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        X_vec = self.vectorize(X)
        return self._predict_model(X_vec)

    # evaluate() calls .transform() for confusion-matrix convenience
    def transform(self, X: List[str]) -> np.ndarray:
        """Return embeddings so other utilities can re‑use feature vectors."""
        return self.vectorize(X)

    # Optional: probability outputs
    def predict_proba(self, X: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting probabilities")
        X_vec = self.transform(X)
        return self.clf.predict_proba(X_vec)