"""Sentence-transformer embeddings + tuned logistic regression for text classification."""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_classifier.model import TextClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring

        # ------------------------------------------------------------------
        # Sentence‑Transformer encoder
        # ------------------------------------------------------------------
        self.embedder = SentenceTransformer(model_name)
        super().__init__(self.embedder)

        # ------------------------------------------------------------------
        # Pipeline: StandardScaler → LogisticRegression, wrapped in GridSearch
        # ------------------------------------------------------------------
        base_clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=n_jobs,
        )
        pipe = make_pipeline(StandardScaler(with_mean=True), base_clf)

        self.clf = GridSearchCV(
            estimator=pipe,
            param_grid={"logisticregression__C": self.Cs},
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # scikit‑learn plumbing
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Fit / predict API required by TextClassifier
    # ------------------------------------------------------------------
    def _fit_model(self, X_vec, y):
        logger.info(
            "[TransformerLogReg] Fitting GridSearchCV on %d vectors (dims=%d)",
            X_vec.shape[0],
            X_vec.shape[1],
        )
        self.clf.fit(X_vec, y)
        self._is_fitted = True
        logger.info(
            "[TransformerLogReg] Best C = %.3f | CV‑score = %.4f",
            self.clf.best_params_["logisticregression__C"],
            self.clf.best_score_,
        )

    def _predict_model(self, X_vec):
        """Hook expected by the TextClassifier base class."""
        return self.clf.predict(X_vec)

    def fit(self, X, y):
        X_vec = self.vectorize(X)
        self._fit_model(X_vec, y)
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")
        X_vec = self.vectorize(X)
        return self._predict_model(X_vec)

    # Optional helpers
    def transform(self, X):
        return self.vectorize(X)

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("Fit the model before predict_proba()")
        X_vec = self.vectorize(X)
        return self.clf.best_estimator_.predict_proba(X_vec)
