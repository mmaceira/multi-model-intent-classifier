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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from intent_classifier.model import TextClassifier
from intent_classifier.utils.model_registry import register_model


@register_model("NaiveBayesClassifier")
class NaiveBayesClassifier(TextClassifier):
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

    _expects_vectors = False

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

        # Initialize base classifier
        # Note: For multi-label, this will be wrapped with OneVsRestClassifier in _fit_model()
        self._base_clf = MultinomialNB(alpha=alpha)
        self.clf = None  # Will be set in _fit_model() based on label format (single vs multi-label)

    def _fit_model(self, X_vec, y):
        """Train the Naive Bayes classifier.

        Args:
            X_vec: Vectorized text features with shape (n_samples, n_features)
            y: Target labels in sklearn format:
                - Single-label: array of shape (n_samples,) with string labels
                - Multi-label: binary matrix of shape (n_samples, n_classes) with 0/1 values
        """
        # ========================================================================
        # STEP 1: Wrap base classifier for multi-label support if needed
        # ========================================================================
        # MultinomialNB doesn't natively support multi-label, so we wrap it with
        # OneVsRestClassifier which trains one binary classifier per class
        if self._is_multilabel:
            # MULTI-LABEL PATH: Wrap with OneVsRestClassifier
            self.clf = OneVsRestClassifier(self._base_clf)
        else:
            # SINGLE-LABEL PATH: Use base classifier directly
            self.clf = self._base_clf

        # ========================================================================
        # STEP 2: Train the classifier
        # ========================================================================
        self.clf.fit(X_vec, y)

        # Keep scikit‑learn compatibility (for API compatibility)
        self.classes_ = getattr(self.clf, "classes_", None)

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

        This method returns probability estimates for each class by vectorizing
        the input and using the underlying classifier's predict_proba method.

        Parameters
        ----------
        X_raw : list of str
            Raw text documents to classify

        Returns
        -------
        np.ndarray : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
            - Single-label: probabilities sum to 1.0 per sample
            - Multi-label: probabilities for each label independently (may not sum to 1.0)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting probabilities")

        X_vec = self.vectorize(X_raw)
        # Both single-label and multi-label use the same predict_proba interface
        # (OneVsRestClassifier handles multi-label automatically)
        return self.clf.predict_proba(X_vec)
