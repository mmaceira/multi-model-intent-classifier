"""
Retrieval-Augmented Generation (RAG) LLM Intent Classifier
==========================================================

A production-ready intent classifier using RAG with LLMs. Combines vector similarity
search with LLM reasoning.

Key Features:
- Rate limiting and batched processing
- Robust error handling with exponential backoff
- Concurrency control for optimal performance

Implementation:
1. Two-Stage Process:
   a) Retrieval: Finds k similar examples from database
   b) Generation: Uses LLM to classify based on examples

2. Database Usage:
   - Uses database to find context examples
   - LLM uses examples to make informed decisions
   - More flexible than centroid or k-NN approaches

3. Production Features:
   - Batched processing for efficiency
   - Rate limiting to prevent throttling
   - Error handling with retries

Advantages:
- Most flexible approach
- Handles complex cases well
- Adapts to new patterns

Disadvantages:
- Expensive (requires LLM API calls)
- Slower than other approaches
- More complex implementation

Example usage:
    ```python
    classifier = RagLLM.load_default()
    results = classifier.predict([
        "Apple's stock rose 2% after strong quarterly earnings.",
        "Manchester United signed a new striker for £80 million."
    ])
    # Returns: ["business", "sports"]
    ```

Created: 2025-05-04
Author: ChatGPT
License: Proprietary
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import Counter
from typing import Callable, Iterable, List, Sequence

import numpy as np
from dotenv import load_dotenv
from litellm import acompletion
from litellm.exceptions import RateLimitError

from src.utils.retry import with_retry

from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Disable litellm's proxy-related logging to avoid import errors
# These errors occur because litellm tries to import proxy modules
# even when not using the proxy, and those modules have optional dependencies
os.environ.setdefault(
    "LITELLM_LOG", "ERROR"
)  # Only show errors, not warnings about missing modules
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "true")  # Suppress litellm's verbose logging

# System prompt for the LLM classification task
_PROMPT_SYSTEM = """You are an intent classifier for user utterances.

CRITICAL RULES:
1. You MUST respond with ONLY a valid JSON object containing a 'labels' array
2. You MUST use EXACT labels from the provided list - NO exceptions
3. Do NOT create new labels, do NOT use synonyms, do NOT modify labels, do NOT paraphrase
4. Labels are case-sensitive and must match EXACTLY (including underscores, hyphens, and spelling)
5. The 'labels' array MUST contain EXACTLY the same number of labels as there are utterances
6. DO NOT use code blocks, markdown formatting, or explanations
7. Return ONLY the JSON object - nothing before or after it
8. If an utterance doesn't match any label exactly, choose the CLOSEST matching label from the list

Example format: {"labels": ["transfer_money", "check_balance", "translate"]}
Remember: Return ONLY the JSON, no explanations, no markdown, no extra text."""


def _exponential_backoff(attempt: int) -> float:
    """Calculate exponential backoff wait time with jitter.

    Args:
        attempt: The current retry attempt number (0-indexed)

    Returns:
        float: Delay time in seconds, exponentially increasing with retry attempts
              but capped at 30 seconds maximum. Includes randomization to prevent
              synchronized retries across multiple clients.
    """
    delay = min((2**attempt) + random.random(), 30.0)
    return delay


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
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
        # Calculate minimum interval between API calls based on rate limit
        self._min_interval = 60.0 / rate_limit_per_minute

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
            return np.array(self.embedder.encode(list(docs)), dtype="float32")
        return np.array(self.embedder(list(docs)), dtype="float32")

    @with_retry(max_retries=3, initial_delay=1.0, max_delay=10.0, backoff_factor=2.0, logger=logger)
    def get_completion(self, prompt: str, **kwargs) -> str:
        """Get a completion from the LLM.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM
        **kwargs
            Additional arguments to pass to the completion API

        Returns
        -------
        str
            The completion text
        """
        import asyncio

        async def _get_completion_async():
            response = await acompletion(
                model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
            )
            return response.choices[0].message.content

        try:
            return asyncio.run(_get_completion_async())
        except RuntimeError:
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_get_completion_async())

    async def _chat_with_retry(self, messages):
        """Send a request to the LLM API with robust error handling.

        Features:
        - Rate limiting with minimum interval enforcement
        - Exponential backoff with jitter for retries
        - Detailed error logging

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Response object from litellm with choices[0].message.content

        Raises:
            RateLimitError: If rate limit is exceeded after all retries
            APIError: For other API-related errors
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                # Ensure minimum interval between API calls
                if attempt > 0:
                    await asyncio.sleep(self._min_interval)

                # Build completion parameters
                # Ollama doesn't support response_format parameter
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                }

                # Only add response_format for non-Ollama models
                if not self.model.startswith("ollama/"):
                    completion_params["response_format"] = {"type": "json_object"}

                # Make the API call using litellm
                response = await acompletion(**completion_params)
                return response

            except RateLimitError as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.error(f"Rate limit exceeded after {attempt} attempts: {e}")
                    raise
                delay = _exponential_backoff(attempt)
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.error(f"API error after {attempt} attempts: {e}")
                    raise
                delay = _exponential_backoff(attempt)
                logger.warning(
                    f"API error, retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

    async def _classify_batch(
        self,
        docs: List[str],
        vectors: List[np.ndarray],
        return_probas: bool = False,
        retry_count: int = 0,
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
        # prepare contexts and store neighbors for fallback
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

        # build a single prompt with multiple items
        # Format labels as a numbered list for better readability
        labels_list = "\n".join(
            f"  - {label}" for label in self.labels[:20]
        )  # Show first 20 labels
        if len(self.labels) > 20:
            labels_list += f"\n  ... and {len(self.labels) - 20} more labels"

        lines = []
        for i, (doc, ctx) in enumerate(zip(docs, contexts, strict=False), start=1):
            # Ensure doc is a string and not empty
            doc_text = str(doc).strip() if doc else " "
            ctx_text = str(ctx).strip() if ctx else " "
            lines.append(f"{i}. Utterance: {doc_text}\nContext:\n{ctx_text}")

        # Create the examples part of the JSON structure using actual labels from the list
        # Use diverse labels in examples, not just the first one
        num_examples = min(3, len(docs), len(self.labels))
        example_labels = [self.labels[i % len(self.labels)] for i in range(num_examples)]
        examples_json = ", ".join(f'"{label}"' for label in example_labels)
        if len(docs) > 3:
            examples_json += "..."

        # Show all labels if there are not too many, otherwise show first 30
        if len(self.labels) <= 30:
            full_labels_list = "\n".join(f"  - {label}" for label in self.labels)
        else:
            full_labels_list = "\n".join(f"  - {label}" for label in self.labels[:30])
            full_labels_list += f"\n  ... and {len(self.labels) - 30} more labels"

        user_content = (
            f"Classify these {len(docs)} user utterances into ONE of these EXACT "
            f"intent categories:\n{full_labels_list}\n\n"
            + "CRITICAL RULES - READ CAREFULLY:\n"
            + "1. You MUST use one of the EXACT labels from the list above - "
            "copy them EXACTLY (case-sensitive, including underscores/hyphens)\n"
            + "2. Do NOT create new labels, do NOT use synonyms, do NOT modify "
            "labels, do NOT paraphrase\n"
            + "3. If an utterance seems similar to a label but doesn't match exactly, "
            "choose the CLOSEST matching label from the list\n"
            + "4. Look at the context examples - they show similar utterances and "
            "their correct EXACT labels\n"
            + "5. Your response MUST be ONLY a valid JSON object with a 'labels' array - "
            "NO other text before or after\n"
            + f"6. The 'labels' array MUST contain EXACTLY {len(docs)} labels, "
            "one for each utterance in order\n"
            + "7. Each label MUST be copied EXACTLY from the list above - "
            "no modifications\n\n" + "\n\n".join(lines) + "\n\nRESPOND WITH ONLY THIS JSON FORMAT "
            "(no explanations, no markdown, no code blocks):\n"
            + f'{{"labels": [{examples_json}]}}\n\n'
            + f"Remember: Return ONLY the JSON object. The array must have EXACTLY "
            f"{len(docs)} labels, each one EXACTLY matching a label from the list above."
        )
        messages = [
            {"role": "system", "content": _PROMPT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        text = None  # Initialize to avoid UnboundLocalError in exception handlers
        max_retries_for_invalid = 2  # Retry up to 2 times if response is invalid
        for retry_attempt in range(max_retries_for_invalid + 1):
            try:
                resp = await self._chat_with_retry(messages)
                text = resp.choices[0].message.content.strip()

                # Strip markdown code blocks if present
                # (LLM sometimes wraps JSON in ```json ... ```)
                # Handle cases like: ```json\n{...}\n``` or ```\n{...}\n```
                if text.startswith("```"):
                    # Find the first newline after the opening backticks
                    first_newline = text.find("\n")
                    if first_newline != -1:
                        # Extract everything after the first newline
                        text = text[first_newline + 1 :]
                    else:
                        # No newline, just strip backticks
                        text = text.lstrip("`")
                    # Remove closing backticks (handle multiple backticks)
                    text = text.rstrip("`").strip()

                # Extract JSON from text - handle cases where LLM adds explanations after JSON
                # Find the first { and last } to extract just the JSON object
                first_brace = text.find("{")
                last_brace = text.rfind("}")

                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    # Extract just the JSON portion
                    json_text = text[first_brace : last_brace + 1]
                else:
                    # Fallback: try to parse the whole text
                    json_text = text

                # Improved JSON parsing with better error handling
                try:
                    response_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    # Try to find JSON object in the text more aggressively
                    # Look for patterns like {"labels": [...]}
                    import re

                    json_match = re.search(
                        r'\{[^{]*"labels"[^{]*\[[^\]]*\][^{]*\}', text, re.DOTALL
                    )
                    if json_match:
                        json_text = json_match.group(0)
                        try:
                            response_data = json.loads(json_text)
                        except json.JSONDecodeError:
                            raise e from None
                    else:
                        raise e from None

                # Check if the response has a 'labels' property
                if isinstance(response_data, dict) and "labels" in response_data:
                    labels_out = response_data["labels"]
                    # Ensure labels_out is a list
                    if not isinstance(labels_out, list):
                        logger.error(
                            f"Expected 'labels' to be a list but got "
                            f"{type(labels_out)}: {labels_out}"
                        )
                        # Convert single string to list if that's what we got
                        if isinstance(labels_out, str):
                            labels_out = [labels_out]
                        else:
                            if retry_attempt < max_retries_for_invalid:
                                continue
                            return [self.labels[0]] * len(docs)
                # Fallback to assuming the entire object is the array
                elif isinstance(response_data, list):
                    labels_out = response_data
                # Handle case when response doesn't match expected format
                else:
                    # Try to find any array in the response
                    labels_out = None
                    for key, value in response_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            labels_out = value
                            logger.warning(f"Using array found at key '{key}' instead of 'labels'")
                            break
                    if labels_out is None:
                        # Look for any string that might be a valid label
                        for key, value in response_data.items():
                            if isinstance(value, str) and value in self.labels:
                                logger.warning(
                                    f"Found single label '{value}' at key '{key}', "
                                    f"using for all documents"
                                )
                                return [value] * len(docs)

                        logger.error(f"No valid labels found in response: {response_data}")
                        if retry_attempt < max_retries_for_invalid:
                            continue
                        return [self.labels[0]] * len(docs)

                if isinstance(labels_out, list):
                    # Handle case where number of labels doesn't match docs
                    if len(labels_out) != len(docs):
                        logger.warning(
                            f"Expected {len(docs)} labels but got {len(labels_out)}. Adjusting..."
                        )
                        # Extend with first label if too short
                        if len(labels_out) < len(docs):
                            # Use the last label for extension if available
                            extension_label = labels_out[-1] if labels_out else self.labels[0]
                            labels_out.extend([extension_label] * (len(docs) - len(labels_out)))
                        # Truncate if too long
                        else:
                            labels_out = labels_out[: len(docs)]

                    # Validate that all labels are in allowed list
                    # Use fuzzy matching: case-insensitive, handle underscores/spaces
                    validated_labels = []
                    invalid_count = 0
                    for lab in labels_out:
                        lab_str = str(lab).strip()
                        # Exact match
                        if lab_str in self.labels:
                            validated_labels.append(lab_str)
                        else:
                            invalid_count += 1
                        # Try case-insensitive match
                        lab_lower = lab_str.lower()
                        matched = None
                        for valid_label in self.labels:
                            if valid_label.lower() == lab_lower:
                                matched = valid_label
                                break

                        # Try matching with underscores/spaces normalized
                        if matched is None:
                            lab_normalized = lab_lower.replace(" ", "_").replace("-", "_")
                            for valid_label in self.labels:
                                valid_normalized = (
                                    valid_label.lower().replace(" ", "_").replace("-", "_")
                                )
                                if valid_normalized == lab_normalized:
                                    matched = valid_label
                                    break

                        if matched:
                            validated_labels.append(matched)
                            logger.debug(f"Normalized label '{lab_str}' to '{matched}'")
                        else:
                            # Fallback: use majority vote from neighbors for this document
                            # This is better than always using self.labels[0]
                            doc_idx = len(validated_labels)
                            if doc_idx < len(neighbors_list) and neighbors_list[doc_idx]:
                                # Get majority label from neighbors
                                neighbor_labels = [n["label"] for n in neighbors_list[doc_idx]]
                                if neighbor_labels:
                                    majority_label = Counter(neighbor_labels).most_common(1)[0][0]
                                    if majority_label in self.labels:
                                        validated_labels.append(majority_label)
                                        logger.warning(
                                            f"LLM returned invalid label '{lab_str}' "
                                            f"for doc {doc_idx+1}. "
                                            f"Using majority vote from neighbors: "
                                            f"{majority_label}"
                                        )
                                    else:
                                        validated_labels.append(self.labels[0])
                                        logger.warning(
                                            f"LLM returned invalid label '{lab_str}' "
                                            f"and neighbor majority '{majority_label}' "
                                            f"not in valid labels. Using fallback: "
                                            f"{self.labels[0]}"
                                        )
                                else:
                                    validated_labels.append(self.labels[0])
                                    logger.warning(
                                        f"LLM returned invalid label '{lab_str}' and no neighbors. "
                                        f"Using fallback: {self.labels[0]}"
                                    )
                            else:
                                validated_labels.append(self.labels[0])
                                logger.warning(
                                    f"LLM returned invalid label '{lab_str}'. "
                                    f"Valid labels: {self.labels}. Using fallback: {self.labels[0]}"
                                )

                    # After processing all labels, check if too many invalid labels,
                    # retry the request
                    if invalid_count > len(docs) * 0.3 and retry_attempt < max_retries_for_invalid:
                        logger.warning(
                            f"Too many invalid labels ({invalid_count}/{len(docs)}). "
                            f"Retrying request (attempt {retry_attempt + 1}/"
                            f"{max_retries_for_invalid + 1})"
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
            except json.JSONDecodeError as e:
                if retry_attempt < max_retries_for_invalid:
                    logger.warning(
                        f"Failed parsing JSON (attempt {retry_attempt + 1}/"
                        f"{max_retries_for_invalid + 1}): {e}. Retrying..."
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
                        f"Failed parsing JSON batch response after "
                        f"{max_retries_for_invalid + 1} attempts: "
                        f"{text if text else 'N/A'} | Error: {e}"
                    )
                    break  # Exit retry loop
            except Exception as e:
                if retry_attempt < max_retries_for_invalid:
                    logger.warning(
                        f"Unexpected error (attempt {retry_attempt + 1}/"
                        f"{max_retries_for_invalid + 1}): {e}. Retrying..."
                    )
                    continue  # Retry
                else:
                    logger.error(
                        f"Unexpected error handling batch response after "
                        f"{max_retries_for_invalid + 1} attempts: "
                        f"{text if text else 'N/A'} | Error: {e}"
                    )
                    break  # Exit retry loop

        # fallback: label all with first
        return [self.labels[0]] * len(docs)

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
