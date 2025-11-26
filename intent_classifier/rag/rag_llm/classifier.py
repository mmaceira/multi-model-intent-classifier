"""Main RAG LLM classifier implementation using the new standalone implementation."""

from __future__ import annotations

import logging

# Import the new implementation
import sys
from pathlib import Path as PathLib
from typing import List, Optional, Sequence

import numpy as np

from ...utils.method_logger import get_logger as get_method_logger, log_method
from ..classifier_base import RagClassifierBase

# Add project root to path to import rag_llm
repo_root = PathLib(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import from the standalone rag_llm module
try:
    from rag_llm import Example, Retriever, _load_examples, classify_single  # noqa: E402
except ImportError as err:
    # Fallback: try importing from the module directly
    import importlib.util

    rag_llm_path = repo_root / "rag_llm.py"
    if rag_llm_path.exists():
        spec = importlib.util.spec_from_file_location("rag_llm", rag_llm_path)
        rag_llm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_llm_module)
        Example = rag_llm_module.Example
        Retriever = rag_llm_module.Retriever
        classify_single = rag_llm_module.classify_single
        _load_examples = rag_llm_module._load_examples
    else:
        raise ImportError(f"Could not find rag_llm.py at {rag_llm_path}") from err

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RagLLM(RagClassifierBase):
    """RAG-style intent classifier using the new implementation.

    This classifier uses TF-IDF retrieval with LLM-based classification.
    It wraps the new standalone rag_llm implementation for pipeline compatibility.

    Attributes:
        retriever: TF-IDF retriever component
        model: The LLM model identifier to use for classification
        top_k: Number of similar examples to retrieve for context (maps to k parameter)
        min_labels: Minimum number of distinct labels in retrieved set (maps to m parameter)
        examples: Training examples used for retrieval
        label_defs: Label definitions dictionary
    """

    def __init__(
        self,
        retriever: Retriever,
        labels: Sequence[str],
        *,
        model: str = "ollama/llama3.1:8b",
        top_k: int = 10,
        min_labels: int = 4,
        label_defs: Optional[dict[str, str]] = None,
    ):
        """Initialize the RAG LLM classifier.

        Args:
            retriever: The TF-IDF retriever component
            labels: List of valid classification labels
            model: LLM model identifier (default: "ollama/llama3.1:8b")
            top_k: Number of similar examples to retrieve (default: 10, maps to k)
            min_labels: Minimum distinct labels in retrieved set (default: 4, maps to m)
            label_defs: Optional label definitions dictionary
        """
        super().__init__(labels)
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.min_labels = min_labels
        self.examples = retriever.examples
        self.label_defs = label_defs or {}

    @classmethod
    def load_default(
        cls,
        *,
        use_openai: bool = False,
        model: str = "ollama/llama3.1:8b",
        top_k: int = 10,
        min_labels: int = 4,
        **kwargs,
    ):
        """Create a classifier with default configuration.

        This factory method loads training data and creates a retriever.

        Args:
            use_openai: Whether to use OpenAI embeddings (not used in new impl,
                kept for compatibility)
            model: LLM model identifier (default: "ollama/llama3.1:8b")
            top_k: Number of similar examples to retrieve (default: 10)
            min_labels: Minimum distinct labels in retrieved set (default: 4)
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            RagLLM: A configured classifier instance ready for prediction
        """
        # Load training examples
        examples, label_defs = _load_examples()

        # Create TF-IDF retriever
        retriever = Retriever(examples)

        # Extract labels from examples
        labels = sorted({ex.label for ex in examples})

        return cls(
            retriever=retriever,
            labels=labels,
            model=model,
            top_k=top_k,
            min_labels=min_labels,
            label_defs=label_defs,
        )

    @log_method
    def predict(self, docs: Sequence[str], **kwargs) -> List[str]:
        """Classify multiple documents.

        Args:
            docs: Sequence of document texts to classify
            **kwargs: Additional arguments (ignored)

        Returns:
            List[str]: Predicted labels for each document
        """
        method_logger = get_method_logger()
        results = []
        for doc in docs:
            try:
                result = classify_single(
                    model=self.model,
                    query=doc,
                    retriever=self.retriever,
                    label_defs=self.label_defs,
                    k=self.top_k,
                    m=self.min_labels,
                )
                prediction = result["label"]
                # Log each individual prediction
                method_logger.log_prediction(
                    input=doc,
                    output=prediction,
                    metadata={"confidence": result.get("confidence"), "model": self.model},
                )
                results.append(prediction)
            except Exception as e:
                logger.error(f"Error classifying document '{doc[:50]}...': {e}")
                # Fallback to first label if classification fails
                results.append(self.labels[0] if self.labels else "unknown")

        return results

    @log_method
    def predict_proba(self, docs: Sequence[str], **kwargs) -> np.ndarray:
        """Generate probability estimates for each class.

        This method uses similarity-based probability estimation from retrieved examples.

        Args:
            docs: Sequence of document texts to classify
            **kwargs: Additional arguments (ignored)

        Returns:
            np.ndarray: An array of shape (n_samples, n_classes) containing
                        probability estimates for each class.
        """
        sorted_labels = sorted(self.labels)
        label_to_idx = {label: i for i, label in enumerate(sorted_labels)}
        all_probas = np.zeros((len(docs), len(sorted_labels)))

        for i, doc in enumerate(docs):
            try:
                # Retrieve examples for this document
                retrieved = self.retriever.select_topk_with_min_labels(
                    doc, k=self.top_k, m=self.min_labels
                )

                # Count label occurrences in retrieved examples
                label_counts = {}
                for ex, _ in retrieved:
                    label = ex.label
                    label_counts[label] = label_counts.get(label, 0) + 1

                # Convert to probabilities
                total = len(retrieved) if retrieved else 1
                for label, count in label_counts.items():
                    if label in label_to_idx:
                        all_probas[i, label_to_idx[label]] = count / total

                # Normalize to ensure sum is 1.0
                row_sum = all_probas[i, :].sum()
                if row_sum > 0:
                    all_probas[i, :] /= row_sum
                else:
                    # If no matches, assign equal probabilities
                    all_probas[i, :] = 1.0 / len(sorted_labels)

            except Exception as e:
                logger.error(f"Error computing probabilities for document '{doc[:50]}...': {e}")
                # Assign equal probabilities on error
                all_probas[i, :] = 1.0 / len(sorted_labels)

        return all_probas
