
"""Multinomial Naive Bayes implementation of TextClassifier."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.model import TextClassifier

class NaiveBayesClassifier(TextClassifier):
    def __init__(self, max_features: int = 10000, alpha: float = 0.1):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        super().__init__(vectorizer)
        self.alpha = alpha
        self.clf = MultinomialNB(alpha=alpha)

    def _fit_model(self, X_vec, y):
        self.clf.fit(X_vec, y)

    def _predict_model(self, X_vec):
        return self.clf.predict(X_vec)
