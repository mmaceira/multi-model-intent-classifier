"""
Embedding-based Logistic Regression module for text classification (scaled + C‑tuned).

This module provides a flexible text classification pipeline that can use either:
- OpenAI embeddings (via API, requires OPENAI_API_KEY)
- SBERT embeddings (local, no API key needed)

The implementation mirrors `TransformerLogReg` but supports multiple embedding backends.
Everything else – scaling, C‑grid search, TextClassifier interface – is identical.

Why a *learnt* classifier?
-------------------------
Retrieval‑based RAG uses *nearest neighbours* and majority voting.  That works
when each class has a tight, linearly separable cluster in embedding space and
`k` is large enough to sample it.  With 25 classes × ~70 train examples/class
and `k=5`, the retrieved set often mixes labels – effectively acting like a
*1‑NN* rule with high variance.  A linear classifier learns *global* decision
boundaries across *all* training points, weighting every dimension according to
its class‑discriminative power and smoothing out idiosyncrasies of individual
examples.  On the same embeddings this usually buys you +5–15 macro‑F1.

Implementation details
---------------------
* StandardScaler(with_mean=True) → centers each feature and scales variance,
  which is appropriate for dense embedding vectors.
* GridSearchCV sweeps C across (0.1 … 10) – edit the tuple if you need more.
* Supports both OpenAI and SBERT embeddings via `use_openai` parameter.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_classifier.model import TextClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingLogReg(TextClassifier):
    """Flexible embedding-based Logistic Regression (OpenAI or SBERT embeddings).

    Supports both OpenAI API embeddings and local SBERT embeddings. The embedding
    backend is selected via the `use_openai` parameter.

    Parameters
    ----------
    use_openai : bool, default=False
        If True, uses OpenAI embeddings (requires OPENAI_API_KEY).
        If False, uses local SBERT embeddings (no API key needed).
    model : str, default="text-embedding-3-small"
        Model name. For OpenAI: embedding model name.
        For SBERT: sentence-transformers model name
        (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    api_key : str | None, default=None
        OpenAI API key. If None and use_openai=True, falls back to OPENAI_API_KEY env var.
    batch_size : int, default=96
        Batch size for embedding generation (OpenAI only).
    Cs : Sequence[float] | None, default=None
        Regularization parameter grid. Default: (0.1, 0.5, 1, 2, 5, 10).
    max_iter : int, default=2000
        Maximum iterations for LogisticRegression.
    cv : int, default=5
        Cross-validation folds for GridSearchCV.
    n_jobs : int, default=2
        Number of parallel jobs for GridSearchCV. Default is 2 to avoid memory issues.
        Use -1 to use all cores, but beware of memory pressure.
    scoring : str, default="f1_macro"
        Scoring metric for GridSearchCV.
    """

    _expects_vectors = False  # Texts in → vectors internally

    def __init__(
        self,
        use_openai: bool = False,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 96,
        Cs: Sequence[float] | None = None,
        max_iter: int = 2000,
        cv: int = 5,
        n_jobs: int = 2,
        scoring: str = "f1_macro",
    ) -> None:
        # ------------------------------------------------------------------
        # Store parameters without mutating (clone‑safe)
        # ------------------------------------------------------------------
        self.use_openai = use_openai
        # Set default model based on embedding backend
        if model is None:
            if use_openai:
                model = "text-embedding-3-small"
            else:
                model = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.Cs = tuple(Cs) if Cs is not None else (0.1, 0.5, 1, 2, 5, 10)
        self.max_iter = max_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring

        # ------------------------------------------------------------------
        # Initialize embedder based on backend
        # ------------------------------------------------------------------
        if use_openai:
            from intent_classifier.embeddings.openai_embedder import OpenAIEmbedder

            # Check for API key
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "use_openai=True requires OPENAI_API_KEY environment variable "
                        "or api_key parameter"
                    )
            self.embedder = OpenAIEmbedder(model=model, api_key=api_key, batch_size=batch_size)
        else:
            # Use SBERT embeddings (local, no API key needed)
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(model)
        super().__init__(self.embedder)

        # ------------------------------------------------------------------
        # Pipeline: scaler → logistic regression, then GridSearch
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
            "use_openai": self.use_openai,
            "model": self.model,
            "api_key": self.api_key,
            "batch_size": self.batch_size,
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
    # Vectorisation helper
    # ------------------------------------------------------------------
    def vectorize(self, texts: List[str]):
        vectors = self.embedder.encode(texts)
        return np.array(vectors)  # Ensure we return a numpy array

    # ------------------------------------------------------------------
    # Fit / predict API
    # ------------------------------------------------------------------
    def _fit_model(self, X_vec: np.ndarray, y):
        backend_name = "OpenAI" if self.use_openai else "SBERT"
        logger.info(
            "[EmbeddingLogReg-%s] Fitting GridSearchCV on %d vectors (dims=%d)",
            backend_name,
            X_vec.shape[0],
            X_vec.shape[1],
        )
        self.clf.fit(X_vec, y)
        self._is_fitted = True
        logger.info(
            "[EmbeddingLogReg-%s] Best C = %.3f | CV‑score = %.4f",
            backend_name,
            self.clf.best_params_["logisticregression__C"],
            self.clf.best_score_,
        )

    def _predict_model(self, X_vec):
        """Hook expected by the TextClassifier base class."""
        return self.clf.predict(X_vec)

    def fit(self, X: List[str], y):
        X_vec = self.vectorize(X)
        self._fit_model(X_vec, y)
        return self

    def predict(self, X: List[str]):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")
        X_vec = self.vectorize(X)
        return self._predict_model(X_vec)

    # Optional convenience
    def transform(self, X: List[str]):
        return self.vectorize(X)

    def predict_proba(self, X: List[str]):
        if not self._is_fitted:
            raise RuntimeError("Fit the model before predict_proba()")
        X_vec = self.vectorize(X)
        return self.clf.best_estimator_.predict_proba(X_vec)

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------
    def __getstate__(self):
        """
        Build a picklable representation by *excluding* any object that
        contains thread locks or live network handles.  We only store:

        * the hyper‑parameters that define the model,
        * the fitted coef_/intercept_ of the LogisticRegression,
        * the fitted scale_/mean_ of the StandardScaler,
        * a few GridSearchCV statistics for convenience.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Serialising an unfitted model makes little sense – " "call .fit(...) first"
            )

        # ----------------------------------------------------------------
        # 1. Grab fitted pieces
        # ----------------------------------------------------------------
        pipe: Pipeline = self.clf.best_estimator_
        scaler: StandardScaler = pipe.named_steps["standardscaler"]
        logreg: LogisticRegression = pipe.named_steps["logisticregression"]

        state: dict[str, Any] = {
            # --- bare hyper‑parameters ----------------------------------
            "use_openai": self.use_openai,
            "model": self.model,
            "api_key": None,  # never persist secrets
            "batch_size": self.batch_size,
            "Cs": self.Cs,
            "max_iter": self.max_iter,
            "cv": self.cv,
            "n_jobs": 1,  # always 1 on reload
            "scoring": self.scoring,
            # --- fitted weights -----------------------------------------
            "scaler_params": {
                "mean_": deepcopy(scaler.mean_),
                "scale_": deepcopy(scaler.scale_),
                "n_features_in_": scaler.n_features_in_,
            },
            "logreg_params": {
                "coef_": deepcopy(logreg.coef_),
                "intercept_": deepcopy(logreg.intercept_),
                "classes_": deepcopy(logreg.classes_),
                "n_features_in_": logreg.n_features_in_,
            },
            # --- a bit of CV bookkeeping (optional) ---------------------
            "best_C": self.clf.best_params_["logisticregression__C"],
            "best_score": self.clf.best_score_,
        }

        # We *intentionally* leave out:
        #   * self.embedder  (holds httpx.Client → RLock or SentenceTransformer)
        #   * self.clf       (holds joblib.Parallel → RLock)
        # and anything else that might drag a lock in.

        return state

    def __setstate__(self, state):
        """
        Rebuild everything from the lightweight state persisted by
        __getstate__.  No network handles – you can unpickle safely even
        on a machine without internet (for SBERT embeddings).
        """
        # ------------- restore bare attributes --------------------------
        self.__dict__.update(
            {
                k: state[k]
                for k in (
                    "use_openai",
                    "model",
                    "batch_size",
                    "Cs",
                    "max_iter",
                    "cv",
                    "n_jobs",
                    "scoring",
                )
            }
        )

        # ------------- resurrect the embedder (fresh instance) -------
        self.api_key = None  # supply at runtime if needed
        if state["use_openai"]:
            from intent_classifier.embeddings.openai_embedder import OpenAIEmbedder

            self.embedder = OpenAIEmbedder(
                model=self.model,
                api_key=self.api_key,
                batch_size=self.batch_size,
            )
        else:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.model)

        # ------------- rebuild the scaler + classifier ------------------
        scaler = StandardScaler(with_mean=True)
        scaler.mean_ = state["scaler_params"]["mean_"]
        scaler.scale_ = state["scaler_params"]["scale_"]
        scaler.n_features_in_ = state["scaler_params"]["n_features_in_"]

        logreg = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            multi_class="ovr",
            n_jobs=1,  # no parallel backend on load
        )
        logreg.classes_ = state["logreg_params"]["classes_"]
        logreg.coef_ = state["logreg_params"]["coef_"]
        logreg.intercept_ = state["logreg_params"]["intercept_"]
        logreg.n_features_in_ = state["logreg_params"]["n_features_in_"]

        from sklearn.pipeline import Pipeline

        self.clf = Pipeline(
            steps=[
                ("standardscaler", scaler),
                ("logisticregression", logreg),
            ]
        )

        # ------------- book‑keeping -------------------------------------
        self._is_fitted = True
        self.best_score_ = state["best_score"]
        self.best_C_ = state["best_C"]
