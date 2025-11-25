"""Main RAG LLM classifier implementation."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Callable, Iterable, List, Sequence

import numpy as np

from ..classifier_base import RagClassifierBase
from ..retrieval import Retriever
from ..vector_store import VectorStore
from .client import LLMClient
from .prompt_builder import build_classification_prompt
from .response_parser import (
    extract_labels_from_response,
    normalize_label_count,
    parse_json_response,
    validate_and_normalize_labels,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maximum retries for invalid responses
MAX_RETRIES_FOR_INVALID = 2


class RagLLM(RagClassifierBase):
    """RAG-style intent classifier with production-ready features.

    This classifier combines retrieval-augmented generation with LLM-based classification.
    It includes robust error handling, rate limiting, and optimized batch processing
    to balance throughput with API usage.

    Attributes:
        retriever: Component that retrieves semantically similar examples
        model: The LLM model identifier to use for classification (default: "ollama/llama3.1:8b")
        top_k: Number of similar examples to retrieve for context
        batch_size: Number of documents to classify in a single LLM call
        max_retries: Maximum number of retry attempts for failed API calls
        embedder: Optional custom embedding function for document vectorization
    """

    def __init__(
        self,
        retriever: Retriever,
        labels: Sequence[str],
        *,
        model: str = "ollama/llama3.1:8b",
        top_k: int = 5,
        batch_size: int = 8,
        max_concurrency: int = 2,
        rate_limit_per_minute: int = 120,
        embedder: Callable[[Sequence[str]], Iterable[Sequence[float]]] | None = None,
        max_retries: int = 5,
    ):
        """Initialize the RAG LLM classifier.

        Args:
            retriever: The retrieval component that finds similar examples
            labels: List of valid classification labels
            model: LLM model identifier (default: "ollama/llama3.1:8b")
            top_k: Number of similar examples to retrieve for context (default: 5)
            batch_size: Number of documents to process in a single API call (default: 8)
            max_concurrency: Maximum number of concurrent API requests (default: 2)
            rate_limit_per_minute: Maximum API calls per minute (default: 120)
            embedder: Optional custom embedding function (default: None, uses built-in)
            max_retries: Maximum number of retry attempts for failed API calls (default: 5)
        """
        super().__init__(labels)
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size
        self._sem = asyncio.Semaphore(max_concurrency)
        self.embedder = embedder
        self.max_retries = max_retries
        self._llm_client = LLMClient(
            model=model,
            max_retries=max_retries,
            rate_limit_per_minute=rate_limit_per_minute,
        )

    @classmethod
    def load_default(cls, *, use_openai: bool = False, **kwargs):
        """Create a classifier with default configuration.

        This factory method simplifies creation with sensible defaults,
        automatically loading the retriever and extracting labels.

        Args:
            use_openai: Whether to use OpenAI's embedding API instead of local models
                (default: False)
            **kwargs: Additional parameters to override default RagLLM initialization

        Returns:
            RagLLM: A configured classifier instance ready for prediction

        Example:
            ```python
            # Basic usage with defaults
            classifier = RagLLM.load_default()

            # With custom parameters
            classifier = RagLLM.load_default(
                use_openai=True,
                model="gpt-4o",
                batch_size=16
            )
            ```
        """
        retriever = Retriever.from_default(use_openai=use_openai)
        labels = sorted({str(m["label"]) for m in retriever.store.meta})
        return cls(retriever, labels, **kwargs)

    def _fallback_to_kmajority(
        self, docs: List[str], neighbors_list: List[List[dict]]
    ) -> List[str]:
        """Fallback to per-document kMajority when LLM classification fails.

        Args:
            docs: List of documents
            neighbors_list: List of neighbor lists for each document

        Returns:
            List of labels using majority vote from neighbors for each document
        """
        labels = []
        for i, (_doc, neighbors) in enumerate(zip(docs, neighbors_list, strict=False)):
            if neighbors:
                # Get majority label from neighbors
                neighbor_labels = [n["label"] for n in neighbors]
                if neighbor_labels:
                    majority_label = Counter(neighbor_labels).most_common(1)[0][0]
                    if majority_label in self.labels:
                        labels.append(majority_label)
                        logger.info(
                            f"Document {i+1}: Using kMajority fallback label '{majority_label}' "
                            f"from {len(neighbors)} neighbors"
                        )
                    else:
                        # If majority label not in valid labels, use first valid neighbor label
                        valid_neighbor_labels = [
                            n["label"] for n in neighbors if n["label"] in self.labels
                        ]
                        if valid_neighbor_labels:
                            fallback_label = Counter(valid_neighbor_labels).most_common(1)[0][0]
                            labels.append(fallback_label)
                            logger.warning(
                                f"Document {i+1}: Majority label '{majority_label}' not valid. "
                                f"Using fallback '{fallback_label}' from neighbors"
                            )
                        else:
                            labels.append(self.labels[0])
                            logger.warning(
                                f"Document {i+1}: No valid neighbor labels. "
                                f"Using first label '{self.labels[0]}'"
                            )
                else:
                    labels.append(self.labels[0])
                    logger.warning(
                        f"Document {i+1}: No neighbor labels available. "
                        f"Using first label '{self.labels[0]}'"
                    )
            else:
                labels.append(self.labels[0])
                logger.warning(
                    f"Document {i+1}: No neighbors available. "
                    f"Using first label '{self.labels[0]}'"
                )
        return labels

    def _embed(self, docs: Sequence[str]) -> np.ndarray:
        """Generate embeddings for input documents.

        This method provides flexible embedding options:
        1. Custom embedder provided at initialization
        2. Fallback to default sentence transformer model

        Args:
            docs: List of text documents to embed

        Returns:
            np.ndarray: Matrix of document embeddings with shape (n_docs, embedding_dim)
        """
        if self.embedder is None:
            return VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        if hasattr(self.embedder, "encode"):
            # encode() now returns numpy array directly
            return self.embedder.encode(list(docs)).astype("float32")
        return np.array(self.embedder(list(docs)), dtype="float32")

    async def _classify_batch(
        self,
        docs: List[str],
        vectors: List[np.ndarray],
        return_probas: bool = False,
    ) -> List[str]:
        """Classify a batch of documents using the LLM with retrieved context.

        This method implements the core RAG classification logic:
        1. Retrieve similar examples for each document
        2. Build a prompt with documents and their contexts
        3. Call the LLM to classify all documents in one batch
        4. Parse and validate the response with fallbacks

        Args:
            docs: List of documents to classify
            vectors: List of document embedding vectors
            return_probas: Whether to return probability estimates

        Returns:
            List[str]: Predicted labels for each document in the batch

        Note:
            This method includes extensive error handling to ensure robustness
            in production. If classification fails, it falls back to using the
            first available label for all documents.
        """
        # Prepare contexts and store neighbors for fallback
        contexts: List[str] = []
        neighbors_list: List[List[dict]] = []
        for vec in vectors:
            try:
                # Pass the numpy array directly to the retriever
                neigh = self.retriever.top_k(vec, self.top_k)
                neighbors_list.append(neigh)
                # Include both text and label in context to show exact label format
                context_parts = []
                for n in neigh:
                    context_parts.append(f"Example: \"{n['text']}\" → Label: \"{n['label']}\"")
                contexts.append("\n".join(context_parts))
            except Exception as e:
                logger.error(f"Error retrieving neighbors: {e}")
                neighbors_list.append([])
                contexts.append("")  # Use empty context if retrieval fails

        # Build prompt
        messages = build_classification_prompt(docs, contexts, list(self.labels))

        # Retry loop for invalid responses
        text = None  # Initialize to avoid UnboundLocalError in exception handlers
        for retry_attempt in range(MAX_RETRIES_FOR_INVALID + 1):
            try:
                resp = await self._llm_client.chat_with_retry(messages)
                text = resp.choices[0].message.content.strip()

                # Debug logging: log the raw response
                if logger.isEnabledFor(logging.DEBUG) or len(docs) <= 3:
                    logger.debug(f"Raw LLM response (first 500 chars):\n{text[:500]}")

                # Parse JSON response
                try:
                    response_data = parse_json_response(text)
                except Exception as e:
                    if retry_attempt < MAX_RETRIES_FOR_INVALID:
                        logger.warning(
                            f"⚠️  JSON Parse Error (attempt {retry_attempt + 1}/"
                            f"{MAX_RETRIES_FOR_INVALID + 1}): {e}. "
                            f"Response preview: {text[:200] if text else 'N/A'}... Retrying..."
                        )
                        # Add stricter instruction
                        messages[-1]["content"] = (
                            messages[-1]["content"]
                            + "\n\n⚠️ RETRY: Previous response was not valid JSON. "
                            "You MUST return ONLY a valid JSON object, nothing else."
                        )
                        continue  # Retry
                    else:
                        logger.error(
                            f"❌ JSON Parse Failure: Failed to parse response after "
                            f"{MAX_RETRIES_FOR_INVALID + 1} attempts. "
                            f"Error: {e}. Response: {text[:500] if text else 'N/A'}"
                        )
                        break  # Exit retry loop

                # Extract labels from response
                labels_out = extract_labels_from_response(response_data, len(docs))

                # Handle extraction failures
                if labels_out is None:
                    # Check if we got a single label that we can use for all docs
                    if isinstance(response_data, dict):
                        for key, value in response_data.items():
                            if isinstance(value, str) and value in self.labels:
                                logger.warning(
                                    f"Found single label '{value}' at key '{key}', "
                                    f"using for all documents"
                                )
                                return [value] * len(docs)

                    logger.error(
                        f"No valid labels found in response: {response_data}. "
                        f"Falling back to per-document kMajority from neighbors."
                    )
                    if retry_attempt < MAX_RETRIES_FOR_INVALID:
                        continue
                    # Fallback to per-document kMajority using neighbors_list
                    return self._fallback_to_kmajority(docs, neighbors_list)

                # Normalize label count
                labels_out = normalize_label_count(labels_out, len(docs))

                # Validate and normalize labels
                validated_labels, invalid_count = validate_and_normalize_labels(
                    labels_out, list(self.labels), neighbors_list
                )

                # Check if too many invalid labels, retry the request
                if invalid_count > len(docs) * 0.3 and retry_attempt < MAX_RETRIES_FOR_INVALID:
                    logger.warning(
                        f"Too many invalid labels ({invalid_count}/{len(docs)}). "
                        f"Retrying request (attempt {retry_attempt + 1}/"
                        f"{MAX_RETRIES_FOR_INVALID + 1})"
                    )
                    # Add a more strict instruction to the prompt
                    messages[-1]["content"] = (
                        messages[-1]["content"]
                        + "\n\n⚠️ RETRY: Previous response had invalid labels. "
                        "You MUST use EXACT labels from the list. "
                        "Copy them character-by-character."
                    )
                    continue  # Retry the request

                # Ensure we have the correct number of labels
                if len(validated_labels) != len(docs):
                    logger.warning(
                        f"Validated labels count ({len(validated_labels)}) "
                        f"doesn't match docs count ({len(docs)}). Adjusting..."
                    )
                    if len(validated_labels) < len(docs):
                        # Extend with last valid label or first label
                        extension_label = (
                            validated_labels[-1] if validated_labels else self.labels[0]
                        )
                        validated_labels.extend(
                            [extension_label] * (len(docs) - len(validated_labels))
                        )
                    else:
                        # Truncate if too long
                        validated_labels = validated_labels[: len(docs)]

                return validated_labels

            except Exception as e:
                if retry_attempt < MAX_RETRIES_FOR_INVALID:
                    logger.warning(
                        f"Unexpected error (attempt {retry_attempt + 1}/"
                        f"{MAX_RETRIES_FOR_INVALID + 1}): {e}. Retrying..."
                    )
                    continue  # Retry
                else:
                    logger.error(
                        f"Unexpected error handling batch response after "
                        f"{MAX_RETRIES_FOR_INVALID + 1} attempts: "
                        f"{text if text else 'N/A'} | Error: {e}"
                    )
                    break  # Exit retry loop

        # Final fallback: use per-document kMajority from neighbors
        logger.error(
            "All retry attempts exhausted for batch classification. "
            "Falling back to per-document kMajority from neighbors."
        )
        return self._fallback_to_kmajority(docs, neighbors_list)

    def predict(self, docs: Sequence[str]) -> List[str]:
        """Classify multiple documents with batched processing.

        This is the main public API method for classification. It:
        1. Embeds all documents
        2. Divides them into batches for efficient processing
        3. Classifies each batch with concurrency control
        4. Handles asyncio runtime details to work in any environment

        Args:
            docs: Sequence of document texts to classify

        Returns:
            List[str]: Predicted labels for each document

        Example:
            ```python
            classifier = RagLLM.load_default()
            results = classifier.predict([
                "OPEC agreed to cut oil production by 2 million barrels per day",
                "Scientists discover new species of deep-sea fish"
            ])
            ```
        """

        async def _run_all() -> List[str]:
            """Internal async implementation of batch prediction."""
            logger.info(
                f"Starting prediction for {len(docs)} documents with batch size {self.batch_size}"
            )
            embeddings = self._embed(docs)
            batches = [docs[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)]
            vec_batches = [
                embeddings[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)
            ]
            logger.info(f"Split into {len(batches)} batches")
            results: List[str] = []
            for i, (docs_batch, vec_batch) in enumerate(zip(batches, vec_batches, strict=False), 1):
                logger.info(f"Processing batch {i}/{len(batches)} with {len(docs_batch)} documents")
                async with self._sem:
                    preds = await self._classify_batch(
                        docs_batch, list(vec_batch), return_probas=False
                    )
                results.extend(preds)
            return results

        # Run async function in appropriate environment
        try:
            # Standard approach - works in most cases
            return asyncio.run(_run_all())
        except RuntimeError:  # Handle "event loop is already running" error
            # Solution for environments like Jupyter notebooks where an event loop
            # is already running
            import nest_asyncio

            nest_asyncio.apply()  # Patch the running loop
            # Get the current event loop instead of creating a new thread
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run_all())

    def predict_proba(self, docs: Sequence[str]) -> np.ndarray:
        """Generate probability estimates for each class.

        This method returns probability estimates for each document across all classes
        by using a similarity-based approach with the retrieved examples.

        Args:
            docs: Sequence of document texts to classify

        Returns:
            np.ndarray: An array of shape (n_samples, n_classes) containing
                        probability estimates for each class.

        Example:
            ```python
            classifier = RagLLM.load_default()
            probas = classifier.predict_proba([
                "OPEC agreed to cut oil production by 2 million barrels per day"
            ])
            # Returns array of shape (1, n_classes) with probabilities for each class
            ```
        """

        async def _run_all_proba() -> np.ndarray:
            """Internal async implementation for probability prediction."""
            logger.info(f"Starting probability prediction for {len(docs)} documents")
            embeddings = self._embed(docs)
            batches = [docs[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)]
            vec_batches = [
                embeddings[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)
            ]
            logger.info(f"Split into {len(batches)} batches")

            # Initialize the results array
            sorted_labels = sorted(self.labels)
            label_to_idx = {label: i for i, label in enumerate(sorted_labels)}
            all_probas = np.zeros((len(docs), len(sorted_labels)))

            # Process each batch
            batch_start = 0
            for i, (docs_batch, vec_batch) in enumerate(zip(batches, vec_batches, strict=False), 1):
                logger.info(f"Processing batch {i}/{len(batches)} with {len(docs_batch)} documents")
                async with self._sem:
                    # For similarity-based probability estimation
                    for j, (_doc, vec) in enumerate(zip(docs_batch, vec_batch, strict=False)):
                        try:
                            # Pass the numpy array directly to the retriever
                            neighbors = self.retriever.top_k(vec, self.top_k)
                            # Count label occurrences
                            label_counts = {}
                            for neighbor in neighbors:
                                label = neighbor["label"]
                                label_counts[label] = label_counts.get(label, 0) + 1

                            # Convert to probabilities
                            for label, count in label_counts.items():
                                if label in label_to_idx:
                                    all_probas[batch_start + j, label_to_idx[label]] = (
                                        count / self.top_k
                                    )
                        except Exception as e:
                            logger.error(f"Error processing document {j} in batch {i}: {e}")
                            # If there's an error, assign equal probabilities to all classes
                            all_probas[batch_start + j, :] = 1.0 / len(sorted_labels)

                batch_start += len(docs_batch)

            return all_probas

        # Run async function in appropriate environment
        try:
            # Standard approach - works in most cases
            return asyncio.run(_run_all_proba())
        except RuntimeError:  # Handle "event loop is already running" error
            # Solution for environments like Jupyter notebooks where an event loop
            # is already running
            import nest_asyncio

            nest_asyncio.apply()  # Patch the running loop
            # Get the current event loop instead of creating a new thread
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run_all_proba())
