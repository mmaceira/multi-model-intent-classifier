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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src.model import TextClassifier

class LinearSVMClassifier(TextClassifier):
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

    def _fit_model(self, X_vec, y):
        """Train the SVM classifier.
        
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

class LinearSVMBigrams(TextClassifier):
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
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        super().__init__(vectorizer)
        
        # Initialize classifier
        self.clf = LinearSVC(C=C)

    def _fit_model(self, X_vec, y):
        """Train the SVM classifier.
        
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
