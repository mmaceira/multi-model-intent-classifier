"""Sentence-transformer embeddings + tuned logistic regression for text classification."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_classifier.model import TextClassifier
from intent_classifier.utils.model_registry import register_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@register_model("TransformerLogReg")
class TransformerLogReg(TextClassifier):
    """Sentence‑Transformer embeddings + scaled, C‑tuned Logistic Regression."""

    _expects_vectors = False  # pipeline embeds texts internally

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        Cs: Sequence[float] | None = None,
        max_iter: int = 2000,
        cv: int = 5,
        n_jobs: int = 2,
        scoring: str = "f1_macro",
    ) -> None:
        # ------------------------------------------------------------------
        # Store params WITHOUT mutating them (needed for sklearn.clone)
        # ------------------------------------------------------------------
        self.model_name = model_name
        self.Cs = tuple(Cs) if Cs is not None else (0.1, 0.5, 1, 2, 5, 10)
        self.max_iter = max_iter
        self.cv = cv  # Public attribute for sklearn compatibility
        self._cv_param = cv  # Internal use for dynamic CV adjustment
        self.n_jobs = n_jobs
        self.scoring = scoring

        # ------------------------------------------------------------------
        # Sentence‑Transformer encoder
        # ------------------------------------------------------------------
        self.embedder = SentenceTransformer(model_name)
        super().__init__(self.embedder)

        # ------------------------------------------------------------------
        # Pipeline: StandardScaler → LogisticRegression, wrapped in GridSearch
        # Will be wrapped with MultiOutputClassifier if multi-label detected
        # ------------------------------------------------------------------
        # Base classifier will be created in _fit_model with proper CV strategy

        # Store cv parameter for dynamic adjustment during fit
        self._cv_param = cv
        self.clf = None  # Will be initialized in _fit_model with proper CV strategy
        self._is_fitted = False
        self._base_clf = None  # Will store GridSearchCV if multi-label (before wrapping)

    # ------------------------------------------------------------------
    # scikit‑learn plumbing
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "model_name": self.model_name,
            "Cs": self.Cs,
            "max_iter": self.max_iter,
            "cv": self.cv,
            "n_jobs": self.n_jobs,
            "scoring": self.scoring,
        }
        if deep:
            params.update({"embedder": self.embedder})
        return params

    # ------------------------------------------------------------------
    # Vectorisation helpers
    # ------------------------------------------------------------------
    def vectorize(self, texts):
        return self.embedder.encode(texts, convert_to_numpy=True)

    @staticmethod
    def _adjust_cv_strategy(cv: int, y: np.ndarray, is_multilabel: bool) -> int:
        """Adjust CV strategy based on class distribution to avoid warnings.

        Args:
            cv: Original CV parameter (number of folds)
            y: Target labels (binary matrix for multilabel, array for single-label)
            is_multilabel: Whether this is a multilabel problem

        Returns:
            Adjusted CV parameter (reduced if needed to avoid warnings)
        """
        if is_multilabel:
            # For multilabel, check minimum class frequency across all labels
            min_class_freq = int(y.sum(axis=0).min()) if y.ndim == 2 else 1
        else:
            # For single-label, check minimum class frequency
            unique, counts = np.unique(y, return_counts=True)
            min_class_freq = int(counts.min()) if len(counts) > 0 else 1

        # Ensure we have at least 2 samples per class for CV to work
        # Reduce CV folds if needed (minimum 2 folds for meaningful CV)
        adjusted_cv = min(cv, max(2, min_class_freq))
        if adjusted_cv < cv:
            logger.info(
                f"Reducing CV folds from {cv} to {adjusted_cv} due to small class sizes "
                f"(minimum class frequency: {min_class_freq})"
            )
        return adjusted_cv

    # ------------------------------------------------------------------
    # Fit / predict API required by TextClassifier
    # ------------------------------------------------------------------
    def _fit_model(self, X_vec, y):
        """Train the TransformerLogReg classifier.

        Args:
            X_vec: Vectorized text features with shape (n_samples, n_features)
            y: Target labels in sklearn format:
                - Single-label: array of shape (n_samples,) with string labels
                - Multi-label: binary matrix of shape (n_samples, n_classes) with 0/1 values
        """
        logger.info(
            "[TransformerLogReg] Fitting GridSearchCV on %d vectors (dims=%d)",
            X_vec.shape[0],
            X_vec.shape[1],
        )

        # ========================================================================
        # STEP 1: Initialize GridSearchCV with CV strategy (adjusted for small datasets)
        # ========================================================================
        # Adjust CV strategy based on class distribution to avoid warnings
        cv_strategy = self._adjust_cv_strategy(self._cv_param, y, self._is_multilabel)
        pipe = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(
                max_iter=self.max_iter,
                solver="lbfgs",
                n_jobs=self.n_jobs,
            ),
        )

        self.clf = GridSearchCV(
            estimator=pipe,
            param_grid={"logisticregression__C": self.Cs},
            cv=cv_strategy,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            error_score=0.0,  # Use 0.0 for failed fits instead of raising error
        )

        # ========================================================================
        # STEP 2: Wrap GridSearchCV with MultiOutputClassifier if multi-label
        # ========================================================================
        # LogisticRegression supports multi-label via MultiOutputClassifier wrapper
        # which trains one binary classifier per label
        if self._is_multilabel:
            # MULTI-LABEL PATH: Wrap GridSearchCV with MultiOutputClassifier
            self._base_clf = self.clf  # Store original GridSearchCV for logging
            self.clf = MultiOutputClassifier(self.clf, n_jobs=self.n_jobs)
        # else: SINGLE-LABEL PATH: Use GridSearchCV directly (no wrapping needed)

        # ========================================================================
        # STEP 3: Train the classifier
        # ========================================================================
        self.clf.fit(X_vec, y)
        self._is_fitted = True

        # ========================================================================
        # STEP 4: Log training results
        # ========================================================================
        if self._is_multilabel:
            logger.info(
                "[TransformerLogReg] Multi-label mode | CV‑score = %.4f",
                self._base_clf.best_score_ if hasattr(self._base_clf, "best_score_") else 0.0,
            )
        else:
            logger.info(
                "[TransformerLogReg] Best C = %.3f | CV‑score = %.4f",
                self.clf.best_params_["logisticregression__C"],
                self.clf.best_score_,
            )

    def _predict_model(self, X_vec):
        """Hook expected by the TextClassifier base class."""
        return self.clf.predict(X_vec)

    def fit(self, X, y):
        # Handle label format detection and conversion (same as base class)
        import numpy as np

        from intent_classifier.utils.label_utils import (
            binarize_labels,
            is_multilabel,
            to_multilabel_format,
        )

        # Detect label format
        self._is_multilabel = is_multilabel(y)

        # Convert labels to format expected by sklearn models
        if self._is_multilabel:
            y_multilabel = to_multilabel_format(y)
            y_binary, self._label_binarizer = binarize_labels(y_multilabel)
            self._classes = self._label_binarizer.classes_
            y_for_training = y_binary
        else:
            y_for_training = np.asarray(y)
            self._classes = np.unique(y_for_training)

        # Vectorize text (SentenceTransformer doesn't need fit, just transform)
        X_vec = self.vectorize(X)

        # Train model with converted labels
        self._fit_model(X_vec, y_for_training)
        self._is_fitted = True
        return self

    def predict(self, X):
        # Use parent class predict() to handle label format conversion
        # This ensures proper conversion from binary matrix to list of lists for multi-label
        return super().predict(X)

    # Optional helpers
    def transform(self, X):
        return self.vectorize(X)

    def predict_proba(self, X):
        """Generate probability estimates for each class.

        Args:
            X: Raw text documents to classify

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes)
                - Single-label: probabilities sum to 1.0 per sample
                - Multi-label: probabilities for each label independently
        """
        if not self._is_fitted:
            raise RuntimeError("Fit the model before predict_proba()")
        X_vec = self.vectorize(X)

        # Both single-label and multi-label use predict_proba, but access differs:
        # - Single-label: access best_estimator_ from GridSearchCV
        # - Multi-label: MultiOutputClassifier wraps GridSearchCV and handles it automatically
        if self._is_multilabel:
            return self.clf.predict_proba(X_vec)
        else:
            return self.clf.best_estimator_.predict_proba(X_vec)
