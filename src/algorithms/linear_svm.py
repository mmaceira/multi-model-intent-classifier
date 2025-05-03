"""Linear SVM implementation of TextClassifier."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src.model import TextClassifier

class LinearSVMClassifier(TextClassifier):
    def __init__(self, max_features: int = 10000, C: float = 1.0):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        super().__init__(vectorizer)
        self.C = C
        self.clf = LinearSVC(C=C)

    def _fit_model(self, X_vec, y):
        self.clf.fit(X_vec, y)

    def _predict_model(self, X_vec):
        return self.clf.predict(X_vec)

class LinearSVMBigrams(TextClassifier):
    """TF‑IDF (1‑2 grams) + LinearSVC (C=5.0)."""
    def __init__(self, max_features: int = 40000, C: float = 5.0):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        super().__init__(vectorizer)
        self.C = C
        self.clf = LinearSVC(C=C)

    def _fit_model(self, X_vec, y):
        self.clf.fit(X_vec, y)

    def _predict_model(self, X_vec):
        return self.clf.predict(X_vec)
