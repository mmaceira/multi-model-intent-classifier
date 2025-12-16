"""
Adapter Sklearn Module

This module provides an adapter class that makes RAG (Retrieval-Augmented Generation)
classifiers compatible with scikit-learn's estimator interface. This allows RAG
classifiers to be used in scikit-learn pipelines and with scikit-learn's model
selection tools.

Key Features:
- scikit-learn compatibility
- Probability estimation support
- Custom serialization handling
- Proper cloning support

Classes:
- RagSklearnAdapter: Adapter class for scikit-learn compatibility

Functions:
- None (Class methods only)

Dependencies:
- numpy
- sklearn.base

Example Usage:
    >>> # Create a RAG classifier
    >>> from intent_classifier.rag.rag_llm import RagLLM
    >>> rag_clf = RagLLM.load_default()

    >>> # Wrap it in the adapter
    >>> from rag.adapter_sklearn import RagSklearnAdapter
    >>> sklearn_clf = RagSklearnAdapter(rag_clf)

    >>> # Use in scikit-learn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('preprocessor', preprocessor),
    ...     ('classifier', sklearn_clf)
    ... ])
"""

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from intent_classifier.utils.method_logger import log_method
from intent_classifier.utils.model_registry import register_model

logger = logging.getLogger(__name__)


@register_model("RagSklearnAdapter")
class RagSklearnAdapter(BaseEstimator, ClassifierMixin):
    """Adapter to make RAG classifiers compatible with sklearn interface."""

    def __init__(self, rag_clf: Any, **_: Any) -> None:
        self.rag = rag_clf
        self.rag_clf = rag_clf  # Add this for compatibility with clone()

    @log_method()
    def fit(self, X: Any, y: Any = None) -> "RagSklearnAdapter":
        return self

    @log_method()
    def predict(self, X: Any) -> Any:
        if isinstance(X, list):
            return self.rag.predict(X)
        elif isinstance(X, np.ndarray):
            return self.rag.predict(X.tolist())
        else:
            raise ValueError(f"Input must be a list or numpy array, got {type(X)}")

    @log_method()
    def predict_proba(self, X: Any) -> np.ndarray:
        """Generate probability estimates for each class.

        This method delegates to the underlying RAG model's predict_proba method
        if it exists, otherwise raises an error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict probabilities for.

        Returns
        -------
        np.ndarray : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.

        Raises
        ------
        AttributeError : If the underlying RAG model doesn't have a predict_proba method.
        """
        if hasattr(self.rag, "predict_proba"):
            # Convert input to list of strings if it's a numpy array
            if isinstance(X, np.ndarray):
                if X.dtype.kind in ["U", "S"]:  # If array contains strings
                    X = X.tolist()
                else:
                    raise ValueError("Input array must contain strings")
            elif isinstance(X, list):
                # Verify all elements are strings
                if not all(isinstance(x, str) for x in X):
                    raise ValueError("All elements in input list must be strings")
            else:
                raise ValueError(
                    f"Input must be a list of strings or numpy array of strings, got {type(X)}"
                )

            return self.rag.predict_proba(X)
        else:
            raise AttributeError("The underlying RAG model does not implement predict_proba")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        This is required for proper cloning.
        """
        return {"rag_clf": self.rag_clf}

    def __sklearn_clone__(self):
        """Custom clone method for sklearn compatibility.

        This method is called by sklearn's clone() function.
        It returns a shallow copy of self instead of a deep copy,
        which prevents pickling errors for unpicklable components.
        """
        return self.__class__(self.rag_clf)

    @staticmethod
    def _get_log_dir_from_config(config: dict):
        """Get log directory from config for saving prompts and responses."""
        try:
            from pathlib import Path

            paths_cfg = config.get("paths", {}) or {}

            # Prefer the dedicated llm_logs_dir from the new path schema.
            llm_logs_dir = paths_cfg.get("llm_logs_dir")
            if llm_logs_dir:
                return Path(llm_logs_dir)

            # Fallback: derive from eval_dir if present.
            eval_dir = paths_cfg.get("eval_dir")
            if eval_dir:
                return Path(eval_dir) / "rag_llm_logs"
        except Exception:
            pass
        return None

    @staticmethod
    def _get_config():
        """Get configuration from standard locations.

        This helper method delegates to the centralized layered config loader and
        returns the merged configuration dictionary for the current run.

        Returns:
            dict: Configuration dictionary
        """
        try:
            from intent_classifier.utils.config_loader import load_config_with_metadata

            metadata = load_config_with_metadata()
            return metadata["config"]
        except Exception:
            # In case of any errors, return empty dict
            return {}

    def __getstate__(self):
        """Custom serialization that excludes unpicklable components.

        This method is called when pickling the object.
        """
        # Save the rag type and other relevant information to recreate it later
        state = {"rag_type": None}

        if self.rag is not None:
            state["rag_type"] = self.rag.__class__.__name__
            # Store information about whether it uses OpenAI
            if hasattr(self.rag, "retriever") and hasattr(self.rag.retriever, "store"):
                state["use_openai"] = "openai" in str(self.rag.retriever.store.__class__).lower()

            # Store top_k if available
            if hasattr(self.rag, "top_k"):
                state["top_k"] = self.rag.top_k

            # Store LLM model name if it's a RagLLM
            if state["rag_type"] == "RagLLM" and hasattr(self.rag, "model"):
                state["llm_model"] = self.rag.model
                # Store min_labels if available (new parameter)
                if hasattr(self.rag, "min_labels"):
                    state["min_labels"] = self.rag.min_labels

        return state

    def __setstate__(self, state):
        """Custom deserialization that reinitializes the RAG component.

        This method is called when unpickling the object.
        """
        # Initialize empty attributes
        self.rag = None
        self.rag_clf = None

        # Only attempt reinitialization if we have a rag_type
        if state.get("rag_type"):
            # Import here to avoid circular imports
            from intent_classifier.rag import load_centroid, load_kmajority, load_llm

            # Extract parameters from state
            use_openai = state.get("use_openai", False)
            top_k = state.get("top_k", 5)

            # Get config - this will try to load it from standard locations
            config = self._get_config()

            # Get model names from config if available
            sbert_model = config.get("model", {}).get(
                "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            openai_model = config.get("model", {}).get(
                "openai_model_name", "text-embedding-3-small"
            )

            # Reinitialize the appropriate RAG model based on rag_type
            rag_type = state["rag_type"]

            if rag_type == "RagKMajority":
                self.rag = load_kmajority(
                    top_k=top_k,
                    use_openai=use_openai,
                    embedding_model=openai_model if use_openai else sbert_model,
                    config=config,
                )
            elif rag_type == "CentroidNN":
                self.rag = load_centroid(use_openai=use_openai)
            elif rag_type == "RagLLM":
                # Strictly use the LLM model from the active run config.
                # We ignore any value saved in the pickle and do not fall back
                # to local defaults – this keeps behaviour tied to the config
                # you pass (DATASET/VARIANT/CONFIG_FILE).
                resolved_cfg = config.get("resolved", {}) or {}
                llm_model = resolved_cfg.get("llm_model")
                if not llm_model:
                    raise RuntimeError(
                        "RagLLM restore failed: 'resolved.llm_model' is not set in the "
                        "active configuration. Please ensure your experiment config "
                        "specifies model.llm_backend and that providers.yaml defines "
                        "the corresponding default model."
                    )

                logger.info(f"Using LLM model from active config: {llm_model}")

                # Get min_labels from state (new parameter in new implementation)
                min_labels = state.get("min_labels", 4)
                # Get prompt_style from state or config
                prompt_style = state.get("prompt_style") or config.get("model", {}).get(
                    "prompt_style", "default"
                )

                # New implementation uses TF-IDF retrieval, no embedder needed
                # The use_openai parameter is kept for compatibility but not used
                self.rag = load_llm(
                    top_k=top_k,
                    model=llm_model,
                    use_openai=use_openai,  # Kept for compatibility, but new impl doesn't use it
                    min_labels=min_labels,
                    prompt_style=prompt_style,
                )

            # Update rag_clf for consistency
            self.rag_clf = self.rag
