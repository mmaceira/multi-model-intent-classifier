
"""MiniLM Sentence-Transformer embeddings + multinomial logistic regression.

Keeps the same interface as NaiveBayesClassifier / LinearSVM so that the
evaluation pipeline can treat all models uniformly.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TransformerLogReg:
    """Text classifier using frozen sentence embeddings + logistic regression."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 batch_size: int = 64,
                 max_iter: int = 1000,
                 random_state: int = 42):
        self.model_name = model_name
        self.batch_size = batch_size
        self._embedder = SentenceTransformer(model_name)
        self._clf = LogisticRegression(max_iter=max_iter,
                                       multi_class='multinomial',
                                       n_jobs=-1,
                                       random_state=random_state)
        
        # **Alias for evaluation.retrieval_recall_at_1**
        # so that classifier.vectorizer.transform(...) calls our transform()
        self.vectorizer = self

    # ------------------------------------------------------------------ #
    # Compatibility helpers expected by evaluation.py                     #
    # ------------------------------------------------------------------ #
    def fit(self, X: List[str], y: List[str]):
        """Encode X and fit multinomial logistic regression."""
        logger.info("Encoding %d documents for training", len(X))
        X_vec = self._embedder.encode(X, batch_size=self.batch_size,
                                      show_progress_bar=True)
        self._clf.fit(X_vec, y)
        return self  # important for evaluate()

    def predict(self, X: List[str]) -> np.ndarray:
        X_vec = self.transform(X)
        return self._clf.predict(X_vec)

    # evaluate() calls .transform() for confusion-matrix convenience
    def transform(self, X: List[str]) -> np.ndarray:
        """Return embeddings so other utilities can re‑use feature vectors."""
        return self._embedder.encode(X, batch_size=self.batch_size,
                                     show_progress_bar=False)

    # Optional: probability outputs
    def predict_proba(self, X: List[str]) -> np.ndarray:
        X_vec = self.transform(X)
        return self._clf.predict_proba(X_vec)
