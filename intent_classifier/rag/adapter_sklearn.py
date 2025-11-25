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
    >>> from rag.rag_llm import RagLLM
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

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RagSklearnAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, rag_clf, **_):
        self.rag = rag_clf
        self.rag_clf = rag_clf  # Add this for compatibility with clone()

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, list):
            return self.rag.predict(X)
        elif isinstance(X, np.ndarray):
            return self.rag.predict(X.tolist())
        else:
            raise ValueError(f"Input must be a list or numpy array, got {type(X)}")

    def predict_proba(self, X):
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

    def get_params(self, deep=True):
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
    def _get_config():
        """Get configuration from standard locations.

        This helper method tries to load the project configuration from standard
        locations, falling back to empty dict if not found.

        Returns:
            dict: Configuration dictionary
        """
        try:
            from pathlib import Path

            import yaml

            # Try to find repo root
            current_file = Path(__file__).resolve()
            for parent in [current_file.parent.parent.parent, current_file.parent.parent]:
                config_path = parent / "config" / "config.yaml"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        return yaml.safe_load(f)

            return {}
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

        return state

    def __setstate__(self, state):
        """Custom deserialization that reinitializes the RAG component.

        This method is called when unpickling the object.
        """
        # Initialize empty attributes
        self.rag = None
        self.rag_clf = None

        # Only attempt reinitialization if we have a rag_type
        if "rag_type" in state and state["rag_type"]:
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
                # For LLM-based RAG
                if use_openai:
                    from intent_classifier.embeddings.openai_embedder import OpenAIEmbedder

                    embedder = OpenAIEmbedder(model=openai_model, batch_size=50)
                    self.rag = load_llm(
                        top_k=top_k, model="ollama/llama3.1:8b", embedder=embedder, use_openai=True
                    )
                else:
                    # For local embeddings
                    from intent_classifier.rag.vector_store import VectorStore

                    def embedder(texts):
                        return VectorStore.embed(sbert_model, texts)

                    self.rag = load_llm(
                        top_k=top_k, model="ollama/llama3.1:8b", embedder=embedder, use_openai=False
                    )

            # Update rag_clf for consistency
            self.rag_clf = self.rag
