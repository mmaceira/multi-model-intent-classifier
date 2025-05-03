
"""model.py – Generic TextClassifier base class (no evaluation)."""
from abc import ABC, abstractmethod
from typing import List

class TextClassifier(ABC):
    """Abstract base class for text classifiers with external evaluation."""
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self._is_fitted = False

    def fit(self, X_raw: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X_raw)
        self._fit_model(X_vec, y)
        self._is_fitted = True
        # cache training vecs + labels for retrieval recall
        self._X_train_vec = X_vec
        self._y_train = y
        return self

    def predict(self, X_raw: List[str]):
        assert self._is_fitted, "You must fit() first."
        X_vec = self.vectorizer.transform(X_raw)
        return self._predict_model(X_vec)

    def vectorize(self, X_raw: List[str]):
        return self.vectorizer.transform(X_raw)

    # ---------------- Abstract methods ----------------
    @abstractmethod
    def _fit_model(self, X_vec, y):
        ...

    @abstractmethod
    def _predict_model(self, X_vec):
        ...
