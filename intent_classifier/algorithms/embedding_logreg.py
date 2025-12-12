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
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_classifier.model import TextClassifier
from intent_classifier.utils.model_registry import register_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@register_model("EmbeddingLogReg")
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
        self.cv = cv  # Public attribute for sklearn compatibility
        self._cv_param = cv  # Internal use for dynamic CV adjustment
        self.n_jobs = n_jobs
        self.scoring = scoring

        # ------------------------------------------------------------------
        # Initialize embedder based on backend
        # ------------------------------------------------------------------
        if use_openai:
            from intent_classifier.utils.embeddings import EmbeddingGenerator

            # Check for API key
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "use_openai=True requires OPENAI_API_KEY environment variable "
                        "or api_key parameter"
                    )
            embedder: Any = EmbeddingGenerator(
                api_key=api_key, model=model or "text-embedding-3-small", batch_size=batch_size
            )
        else:
            # Use SBERT embeddings (local, no API key needed)
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(model)

        self.embedder = embedder

        # Wrap embedder to match TextClassifier's expected type
        # Both EmbeddingGenerator and SentenceTransformer have encode() method
        # that matches Callable[[Sequence[str]], np.ndarray]
        # We create a closure that captures the embedder
        def embedder_callable(texts: Sequence[str]) -> np.ndarray:
            # normalize to list for some embedders
            return embedder.encode(list(texts))

        super().__init__(embedder_callable)

        # ------------------------------------------------------------------
        # Pipeline: scaler → logistic regression, then GridSearch
        # ------------------------------------------------------------------
        # Base classifier will be created in _fit_model with proper CV strategy

        # Store cv parameter for dynamic adjustment during fit
        self._cv_param = cv
        self.clf = None  # Will be initialized in _fit_model with proper CV strategy
        self._is_fitted = False
        self._base_clf = None  # Will store GridSearchCV if multi-label (before wrapping)
        self._is_multilabel = False  # Will be set during fit()
        self._label_binarizer = None  # Will be set during fit() for multi-label
        self._classes = None  # Will be set during fit()

    # ------------------------------------------------------------------
    # scikit‑learn plumbing
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
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
    def vectorize(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.embedder.encode(list(texts))
        return np.array(vectors)  # Ensure we return a numpy array

    # ------------------------------------------------------------------
    # Fit / predict API
    # ------------------------------------------------------------------
    def _fit_model(self, X_vec: np.ndarray, y: np.ndarray) -> None:
        """Train the EmbeddingLogReg classifier.

        Args:
            X_vec: Vectorized text features with shape (n_samples, n_features)
            y: Target labels in sklearn format:
                - Single-label: array of shape (n_samples,) with string labels
                - Multi-label: binary matrix of shape (n_samples, n_classes) with 0/1 values
        """
        backend_name = "OpenAI" if self.use_openai else "SBERT"
        logger.info(
            "[EmbeddingLogReg-%s] Fitting GridSearchCV on %d vectors (dims=%d)",
            backend_name,
            X_vec.shape[0],
            X_vec.shape[1],
        )

        # ========================================================================
        # STEP 1: Initialize GridSearchCV with CV strategy
        # ========================================================================
        cv_strategy = self._cv_param
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
        if self.clf is None:
            raise RuntimeError("Classifier not initialized")
        self.clf.fit(X_vec, y)
        self._is_fitted = True

        # ========================================================================
        # STEP 4: Log training results
        # ========================================================================
        if self._is_multilabel:
            best_score = (
                self._base_clf.best_score_
                if self._base_clf is not None and hasattr(self._base_clf, "best_score_")
                else 0.0
            )
            logger.info(
                "[EmbeddingLogReg-%s] Multi-label mode | CV‑score = %.4f",
                backend_name,
                best_score,
            )
        else:
            if (
                self.clf is not None
                and hasattr(self.clf, "best_params_")
                and hasattr(self.clf, "best_score_")
            ):
                logger.info(
                    "[EmbeddingLogReg-%s] Best C = %.3f | CV‑score = %.4f",
                    backend_name,
                    self.clf.best_params_["logisticregression__C"],
                    self.clf.best_score_,
                )

    def _predict_model(self, X_vec: np.ndarray) -> np.ndarray:
        """Hook expected by the TextClassifier base class."""
        if self.clf is None:
            raise RuntimeError("Classifier not fitted")
        return self.clf.predict(X_vec)

    def fit(self, X: Sequence[str], y: np.ndarray) -> EmbeddingLogReg:
        # Handle label format detection and conversion (same as TransformerLogReg)
        import numpy as np

        from intent_classifier.utils.label_utils import (
            binarize_labels,
            is_multilabel,
            to_multilabel_format,
        )

        # Detect label format - convert to list if numpy array for is_multilabel
        y_for_check = y.tolist() if isinstance(y, np.ndarray) else y
        self._is_multilabel = is_multilabel(y_for_check)

        # Convert labels to format expected by sklearn models
        if self._is_multilabel:
            y_multilabel = to_multilabel_format(y_for_check)
            y_binary, self._label_binarizer = binarize_labels(y_multilabel)
            if self._label_binarizer is not None:
                self._classes = self._label_binarizer.classes_
            y_for_training = y_binary
        else:
            y_for_training = np.asarray(y)
            self._classes = np.unique(y_for_training)

        # Vectorize text
        X_vec = self.vectorize(X)

        # Train model with converted labels
        self._fit_model(X_vec, y_for_training)
        self._is_fitted = True
        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        # Use parent class predict() to handle label format conversion
        # This ensures proper conversion from binary matrix to list of lists for multi-label
        return super().predict(X)

    # Optional convenience
    def transform(self, X: Sequence[str]) -> np.ndarray:
        return self.vectorize(X)

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
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
        if self.clf is None:
            raise RuntimeError("Classifier not fitted")
        if self._is_multilabel:
            return self.clf.predict_proba(X_vec)
        else:
            if hasattr(self.clf, "best_estimator_"):
                return self.clf.best_estimator_.predict_proba(X_vec)
            else:
                raise RuntimeError("Classifier not properly fitted")

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
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
                "Serialising an unfitted model makes little sense – call .fit(...) first"
            )

        # ----------------------------------------------------------------
        # 1. Grab fitted pieces - handle single-label vs multi-label
        # ----------------------------------------------------------------
        if self._is_multilabel:
            # Multi-label: self.clf is MultiOutputClassifier wrapping GridSearchCV objects
            # Extract from ALL estimators (one per label)
            # Note: MultiOutputClassifier.estimators_ contains the fitted GridSearchCV objects
            if self.clf is None:
                raise RuntimeError("Classifier not fitted")
            if not hasattr(self.clf, "estimators_") or len(self.clf.estimators_) == 0:
                raise RuntimeError("Multi-label model not properly fitted")

            # Store parameters from all estimators
            estimators_params = []
            for grid_search in self.clf.estimators_:
                pipe: Pipeline = grid_search.best_estimator_
                scaler: StandardScaler = pipe.named_steps["standardscaler"]
                logreg: LogisticRegression = pipe.named_steps["logisticregression"]

                estimators_params.append(
                    {
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
                        "best_C": grid_search.best_params_["logisticregression__C"],
                        "best_score": grid_search.best_score_,
                    }
                )

            # Use first estimator's params for backward compatibility fields
            first_params = estimators_params[0]
            scaler_params = first_params["scaler_params"]
            logreg_params = first_params["logreg_params"]
            best_C = first_params["best_C"]
            best_score = first_params["best_score"]
        else:
            # Single-label: self.clf is GridSearchCV directly
            if self.clf is None:
                raise RuntimeError("Classifier not fitted")
            single_pipe: Pipeline = self.clf.best_estimator_
            single_scaler: StandardScaler = single_pipe.named_steps["standardscaler"]
            single_logreg: LogisticRegression = single_pipe.named_steps["logisticregression"]

            scaler_params = {
                "mean_": deepcopy(single_scaler.mean_),
                "scale_": deepcopy(single_scaler.scale_),
                "n_features_in_": single_scaler.n_features_in_,
            }
            logreg_params = {
                "coef_": deepcopy(single_logreg.coef_),
                "intercept_": deepcopy(single_logreg.intercept_),
                "classes_": deepcopy(single_logreg.classes_),
                "n_features_in_": single_logreg.n_features_in_,
            }
            best_C = self.clf.best_params_["logisticregression__C"]
            best_score = self.clf.best_score_
            estimators_params = None  # Not used for single-label

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
            "scaler_params": scaler_params,
            "logreg_params": logreg_params,
            # --- multi-label: store all estimators' params --------------
            "estimators_params": estimators_params,  # None for single-label, list for multi-label
            # --- a bit of CV bookkeeping (optional) ---------------------
            "best_C": best_C,
            "best_score": best_score,
            # --- multi-label flag for __setstate__ ----------------------
            "_is_multilabel": self._is_multilabel,
            # --- classes for multi-label reconstruction -------------------
            "_classes": self._classes,
        }

        # We *intentionally* leave out:
        #   * self.embedder  (holds httpx.Client → RLock or SentenceTransformer)
        #   * self.clf       (holds joblib.Parallel → RLock)
        # and anything else that might drag a lock in.

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
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
            from intent_classifier.utils.embeddings import EmbeddingGenerator

            # Get API key from environment if not set
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "use_openai=True requires OPENAI_API_KEY environment variable "
                    "or api_key parameter"
                )
            embedder: Any = EmbeddingGenerator(
                api_key=api_key,
                model=self.model or "text-embedding-3-small",
                batch_size=self.batch_size,
            )
        else:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(self.model)

        self.embedder = embedder

        # ------------- rebuild the scaler + classifier ------------------
        from sklearn.pipeline import Pipeline

        is_multilabel = state.get("_is_multilabel", False)
        estimators_params = state.get("estimators_params")

        # Backward compatibility: if multi-label but no estimators_params, use single estimator
        # (old pickle format that only stored first estimator)
        if is_multilabel and estimators_params is None:
            # Convert single estimator format to list format for consistency
            estimators_params = [
                {
                    "scaler_params": state["scaler_params"],
                    "logreg_params": state["logreg_params"],
                    "best_C": state.get("best_C", 1.0),
                    "best_score": state.get("best_score", 0.0),
                }
            ]

        if is_multilabel and estimators_params is not None and len(estimators_params) > 0:
            # Multi-label: reconstruct all estimators and wrap in MultiOutputClassifier
            estimators = []
            for est_params in estimators_params:
                scaler = StandardScaler(with_mean=True)
                scaler.mean_ = est_params["scaler_params"]["mean_"]
                scaler.scale_ = est_params["scaler_params"]["scale_"]
                scaler.n_features_in_ = est_params["scaler_params"]["n_features_in_"]

                logreg = LogisticRegression(
                    max_iter=self.max_iter,
                    solver="lbfgs",
                    multi_class="ovr",
                    n_jobs=1,  # no parallel backend on load
                )
                logreg.classes_ = est_params["logreg_params"]["classes_"]
                logreg.coef_ = est_params["logreg_params"]["coef_"]
                logreg.intercept_ = est_params["logreg_params"]["intercept_"]
                logreg.n_features_in_ = est_params["logreg_params"]["n_features_in_"]

                pipe = Pipeline(
                    steps=[
                        ("standardscaler", scaler),
                        ("logisticregression", logreg),
                    ]
                )
                estimators.append(pipe)

            # Create MultiOutputClassifier with first estimator as template, then replace with all
            # This ensures proper initialization of MultiOutputClassifier internals
            self.clf = MultiOutputClassifier(
                estimators[0] if estimators else Pipeline([]), n_jobs=1
            )
            # Set attributes that MultiOutputClassifier expects
            self.clf.estimators_ = estimators  # type: ignore[attr-defined]
            self.clf.n_outputs_ = len(estimators)  # type: ignore[attr-defined]
            # Mark as fitted (since estimators are already fitted)
            self.clf._fitted = True  # type: ignore[attr-defined]
        else:
            # Single-label: reconstruct single pipeline
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

            self.clf = Pipeline(
                steps=[
                    ("standardscaler", scaler),
                    ("logisticregression", logreg),
                ]
            )

        # Restore multi-label flag and related attributes
        self._is_multilabel = is_multilabel
        self._label_binarizer = None
        self._classes = state.get("_classes", state["logreg_params"]["classes_"])
        self._base_clf = None  # Not restored (was only for logging during fit)

        # ------------- book‑keeping -------------------------------------
        self._is_fitted = True
        self.best_score_ = state["best_score"]
        self.best_C_ = state["best_C"]
