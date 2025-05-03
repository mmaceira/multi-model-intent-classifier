
from sklearn.base import BaseEstimator, ClassifierMixin
class RagSklearnAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, rag_clf, **_):
        self.rag = rag_clf
    def fit(self, X, y=None): return self
    def predict(self, X): return self.rag.predict(X)
