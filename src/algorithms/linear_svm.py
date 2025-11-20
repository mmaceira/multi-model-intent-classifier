"""\
Linear SVM module for text classification.

This module implements SVM-based text classifiers using TF-IDF features.
It provides both unigram and bigram variants of the classifier.

Classes:
- LinearSVMClassifier: TF-IDF + Linear SVM classifier
- LinearSVMBigrams: TF-IDF with bigrams + Linear SVM

Functions:
- None

Created: 2025-05-03
"""

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.model import TextClassifier


class LinearSVMClassifier(TextClassifier):
    _expects_vectors = False
    """TF-IDF + Linear SVM classifier.

    This class implements a text classifier using TF-IDF features and
    linear SVM. It's suitable for high-dimensional sparse text data
    and provides good baseline performance.

    Attributes:
        C: SVM regularization parameter
        clf: LinearSVC classifier instance

    Example:
        >>> clf = LinearSVMClassifier(max_features=20000, C=0.1)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(self, max_features: int = 10000, C: float = 1.0):
        """Initialize the classifier.

        Args:
            max_features: Maximum vocabulary size (default: 10000)
            C: SVM regularization parameter (default: 1.0)
        """
        # Store constructor parameters as attributes with the same name for BaseEstimator
        self.max_features = max_features
        self.C = C

        # Initialize vectorizer and base class
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        super().__init__(vectorizer)

        # Initialize classifier
        self.clf = LinearSVC(C=C)
        self.classes_ = None

    def _fit_model(self, X_vec, y):
        # Keep scikit‑learn compatibility
        self.classes_ = getattr(self.clf, "classes_", None)
        """Train the SVM classifier.

        Args:
            X_vec: Vectorized text features
            y: Labels
        """
        self.clf.fit(X_vec, y)
        self.classes_ = self.clf.classes_

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

        This method approximates probability estimates by applying
        softmax to the decision function scores from LinearSVC.

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

        # Get decision scores
        decision_scores = self.clf.decision_function(X_vec)

        # Check if classes_ is set, if not, try to get it from the classifier
        if not hasattr(self, "classes_") or self.classes_ is None:
            if hasattr(self.clf, "classes_"):
                self.classes_ = self.clf.classes_
            else:
                # Infer number of classes from decision function output
                if decision_scores.ndim == 1:
                    self.classes_ = np.array([0, 1])  # Binary classification
                else:
                    # Multi-class, assume classes are 0...n-1
                    self.classes_ = np.arange(decision_scores.shape[1])

        # For binary classification, reshape the decision scores
        if len(self.classes_) == 2:
            decision_scores = np.column_stack([-decision_scores, decision_scores])

        # Convert to probabilities using softmax with temperature scaling
        temperature = 1.0  # Adjust this for calibration if needed
        probabilities = softmax(decision_scores / temperature, axis=1)

        return probabilities


class LinearSVMBigrams(TextClassifier):
    _expects_vectors = False
    """TF-IDF with bigrams + Linear SVM classifier.

    This class extends the basic LinearSVMClassifier by using both
    unigrams and bigrams as features. This can capture more complex
    patterns in the text but requires more memory and computation.

    Attributes:
        C: SVM regularization parameter
        clf: LinearSVC classifier instance

    Example:
        >>> clf = LinearSVMBigrams(max_features=50000, C=1.0)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(self, max_features: int = 40000, C: float = 5.0):
        """Initialize the classifier.

        Args:
            max_features: Maximum vocabulary size (default: 40000)
            C: SVM regularization parameter (default: 5.0)
        """
        # Store constructor parameters as attributes with the same name for BaseEstimator
        self.max_features = max_features
        self.C = C

        # Initialize vectorizer and base class
        vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        super().__init__(vectorizer)

        # Initialize classifier
        self.clf = LinearSVC(C=C)
        self.classes_ = None

    def _fit_model(self, X_vec, y):
        # Keep scikit‑learn compatibility
        self.classes_ = getattr(self.clf, "classes_", None)
        """Train the SVM classifier.

        Args:
            X_vec: Vectorized text features
            y: Labels
        """
        self.clf.fit(X_vec, y)
        self.classes_ = self.clf.classes_

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

        This method approximates probability estimates by applying
        softmax to the decision function scores from LinearSVC.

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

        # Get decision scores
        decision_scores = self.clf.decision_function(X_vec)

        # Check if classes_ is set, if not, try to get it from the classifier
        if not hasattr(self, "classes_") or self.classes_ is None:
            if hasattr(self.clf, "classes_"):
                self.classes_ = self.clf.classes_
            else:
                # Infer number of classes from decision function output
                if decision_scores.ndim == 1:
                    self.classes_ = np.array([0, 1])  # Binary classification
                else:
                    # Multi-class, assume classes are 0...n-1
                    self.classes_ = np.arange(decision_scores.shape[1])

        # For binary classification, reshape the decision scores
        if len(self.classes_) == 2:
            decision_scores = np.column_stack([-decision_scores, decision_scores])

        # Convert to probabilities using softmax with temperature scaling
        temperature = 1.0  # Adjust this for calibration if needed
        probabilities = softmax(decision_scores / temperature, axis=1)

        return probabilities
