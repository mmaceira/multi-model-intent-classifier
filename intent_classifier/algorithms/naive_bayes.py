"""\
Naive Bayes module for text classification.

This module implements a text classifier using TF-IDF features and
Multinomial Naive Bayes. It's particularly suitable for text classification
tasks with large vocabularies and provides fast training and prediction.

Classes:
- NaiveBayesClassifier: TF-IDF + Multinomial Naive Bayes classifier

Functions:
- None
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from intent_classifier.model import TextClassifier


class NaiveBayesClassifier(TextClassifier):
    _expects_vectors = False
    """TF-IDF + Multinomial Naive Bayes classifier.

    This class implements a text classifier using TF-IDF features and
    Multinomial Naive Bayes. It's particularly efficient for text
    classification tasks and works well with high-dimensional sparse data.

    Attributes:
        alpha: Smoothing parameter
        clf: MultinomialNB classifier instance

    Example:
        >>> clf = NaiveBayesClassifier(max_features=15000, alpha=0.5)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(self, max_features: int = 10000, alpha: float = 0.1):
        """Initialize the classifier.

        Args:
            max_features: Maximum vocabulary size (default: 10000)
            alpha: Smoothing parameter (default: 0.1)
        """
        # Store constructor parameters as attributes with the same name for BaseEstimator
        self.max_features = max_features
        self.alpha = alpha

        # Initialize vectorizer and base class
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        super().__init__(vectorizer)

        # Initialize classifier
        self.clf = MultinomialNB(alpha=alpha)

    def _fit_model(self, X_vec, y):
        # Keep scikit‑learn compatibility
        self.classes_ = getattr(self.clf, "classes_", None)
        """Train the Naive Bayes classifier.

        Args:
            X_vec: Vectorized text features
            y: Labels
        """
        self.clf.fit(X_vec, y)

    def _predict_model(self, X_vec):
        """Make predictions using the trained classifier.

        Args:
            X_vec: Vectorized text features

        Returns:
            List of predicted labels
        """
        return self.clf.predict(X_vec)

    def predict_proba(self, X_raw):
        """Generate probability estimates for each class.

        This method returns probability estimates for each class
        by vectorizing the input and using the underlying MultinomialNB's
        predict_proba method.

        Parameters
        ----------
        X_raw : list of str
            Raw text documents to classify

        Returns
        -------
        np.ndarray : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting probabilities")

        X_vec = self.vectorize(X_raw)
        return self.clf.predict_proba(X_vec)
