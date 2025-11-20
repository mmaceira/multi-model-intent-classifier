"""
OpenAI‑Embedding Logistic Regression module for text classification (scaled + C‑tuned).

This mirrors `TransformerLogReg` but swaps in the OpenAI embedding endpoint
(or the project‑local `OpenAIEmbedder`) as the vectoriser.  Everything else –
scaling, C‑grid search, TextClassifier interface – is identical, so you can
plug it into `models = { 'OpenAI + LogReg': OpenAIEmbedLogReg(...) }` without
changing your training script.

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
* StandardScaler(with_mean=False) → centres *each* feature while preserving the
  L2 length of the original embedding.
* GridSearchCV sweeps C across (0.1 … 10) – edit the tuple if you need more.
* Works with any callable that implements `encode(texts) -> np.ndarray`; by
  default we import `OpenAIEmbedder` from your code‑base.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# 👉 Adapt this import if your embedder lives elsewhere
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.model import TextClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenAIEmbedLogReg(TextClassifier):
    """OpenAI embeddings + scaled, C‑tuned Logistic Regression."""

    _expects_vectors = False  # Texts in → vectors internally

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 96,
        Cs: Sequence[float] | None = None,
        max_iter: int = 2000,
        cv: int = 5,
        n_jobs: int = -1,
        scoring: str = "f1_macro",
    ) -> None:
        # ------------------------------------------------------------------
        # Store parameters without mutating (clone‑safe)
        # ------------------------------------------------------------------
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.Cs = tuple(Cs) if Cs is not None else (0.1, 0.5, 1, 2, 5, 10)
        self.max_iter = max_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring

        # ------------------------------------------------------------------
        # OpenAI embedder instance (wraps API calls + caching)
        # ------------------------------------------------------------------
        self.embedder = OpenAIEmbedder(model=model, api_key=api_key, batch_size=batch_size)
        super().__init__(self.embedder)

        # ------------------------------------------------------------------
        # Pipeline: scaler → logistic regression, then GridSearch
        # ------------------------------------------------------------------
        base_clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=n_jobs,
        )
        pipe = make_pipeline(StandardScaler(with_mean=False), base_clf)

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
        logger.info(
            "[OpenAIEmbedLogReg] Fitting GridSearchCV on %d vectors (dims=%d)",
            X_vec.shape[0],
            X_vec.shape[1],
        )
        self.clf.fit(X_vec, y)
        self._is_fitted = True
        logger.info(
            "[OpenAIEmbedLogReg] Best C = %.3f | CV‑score = %.4f",
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
        #   * self.embedder  (holds httpx.Client → RLock)
        #   * self.clf       (holds joblib.Parallel → RLock)
        # and anything else that might drag a lock in.

        return state

    def __setstate__(self, state):
        """
        Rebuild everything from the lightweight state persisted by
        __getstate__.  No network handles – you can unpickle safely even
        on a machine without internet.
        """
        # ------------- restore bare attributes --------------------------
        self.__dict__.update(
            {
                k: state[k]
                for k in ("model", "batch_size", "Cs", "max_iter", "cv", "n_jobs", "scoring")
            }
        )

        # ------------- resurrect the embedder (fresh HTTP client) -------
        self.api_key = None  # supply at runtime if needed
        from src.embeddings.openai_embedder import OpenAIEmbedder

        self.embedder = OpenAIEmbedder(
            model=self.model,
            api_key=self.api_key,
            batch_size=self.batch_size,
        )

        # ------------- rebuild the scaler + classifier ------------------
        scaler = StandardScaler(with_mean=False)
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
