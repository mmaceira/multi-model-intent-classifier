"""OpenAIEmbeddings module.

Provides a thin wrapper around the OpenAI embeddings endpoint with transparent batching.

Classes:
- OpenAIEmbedder

Functions:
- chunk_iterable

Created: 2025-05-03
"""

from __future__ import annotations

import itertools
import logging
import os
import random
import time
from typing import Iterable, List

import numpy as np

from src.utils.embeddings import EmbeddingGenerator

try:
    import openai
    from openai import RateLimitError
except ImportError as e:
    raise ImportError("openai package required. Install with `pip install openai`.") from e

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def chunk_iterable(iterable: Iterable[str], batch_size: int = 100):
    """Yield successive batches of size *batch_size* from *iterable*."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


class OpenAIEmbedder:
    """Encode text using OpenAI's cheapest embedding model (``text-embedding-3-small``).

    Parameters
    ----------
    model:
        Embedding model name. Default: ``text-embedding-3-small`` (currently the lowest‑cost).
    batch_size:
        How many texts to send per request (max 2048 tokens total → stay conservative at 100 examples).
    api_key:
        If ``None``, the class falls back to the ``OPENAI_API_KEY`` environment variable.
    organization:
        Optional OpenAI organization ID.
    max_retries:
        Maximum number of retries for rate-limited API calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        """Initialize the OpenAI embedder.

        Parameters
        ----------
        api_key : str | None
            OpenAI API key. If None, falls back to OPENAI_API_KEY environment variable.
        model : str, optional
            Model to use for embeddings, by default "text-embedding-3-small"
        batch_size : int, optional
            Number of texts to process in each batch, by default 100
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided and OPENAI_API_KEY environment variable is not set"
                )

        self.embedder = EmbeddingGenerator(
            api_key=api_key, model=model, batch_size=batch_size, max_retries=3
        )

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to embed

        Returns
        -------
        np.ndarray
            Array of embeddings
        """
        return self.embedder.generate_embeddings(texts)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors aligned with *texts*."""
        # Start timing the entire encoding process
        start_time = time.time()

        # Calculate total number of batches
        total_batches = (len(texts) + self.embedder.batch_size - 1) // self.embedder.batch_size
        logger.info(f"Starting OpenAI embedding for {len(texts)} texts in {total_batches} batches")

        embeddings: List[List[float]] = []
        batch_count = 0
        total_token_count = 0  # This is just an estimate

        for batch in chunk_iterable(texts, self.embedder.batch_size):
            batch_count += 1
            batch_size = len(batch)
            batch_start_time = time.time()

            logger.info(f"Processing batch {batch_count}/{total_batches} ({batch_size} texts)")

            # Rough token estimation (4 chars ~= 1 token)
            batch_token_estimate = sum(len(text) // 4 for text in batch)
            total_token_count += batch_token_estimate

            try:
                # Use the retry-enabled method instead of direct API call
                response = self.get_embeddings(batch)

                # Get actual token usage if available in the response
                if hasattr(response, "usage") and hasattr(response.usage, "total_tokens"):
                    actual_tokens = response.usage.total_tokens

                # Using .data list ensures order preserved
                embeddings.extend([list(embedding) for embedding in response])

                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time

                # Add a small delay between batches to avoid rate limits
                if batch_count < total_batches:
                    delay = random.uniform(0.2, 0.5)
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Error in batch {batch_count}: {str(e)}")
                # If we've completely failed after retries, raise the exception
                raise

        # Calculate and log total embedding time
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(
            f"Completed {total_batches} batches in {total_duration:.2f}s (avg {total_duration/len(texts):.4f}s per text)"
        )

        return embeddings
