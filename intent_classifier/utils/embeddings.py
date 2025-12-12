"""Embedding generation utilities.

This module provides functionality for generating text embeddings using OpenAI's
API. It includes batch processing capabilities and retry logic for handling
API rate limits and temporary failures.

Classes:
- EmbeddingGenerator: Handles the generation of text embeddings using OpenAI's API
"""

import logging
from typing import List

import numpy as np
from openai import OpenAI

from intent_classifier.utils.retry import with_retry

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """A class for generating text embeddings using OpenAI's API.

    This class provides methods for generating embeddings for text data,
    with support for batch processing and automatic retries for failed
    API calls. It uses exponential backoff to handle rate limits and
    temporary failures.

    Attributes
    ----------
    client : OpenAI
        OpenAI client instance for making API calls.

    model : str
        Name of the embedding model to use.

    batch_size : int
        Number of texts to process in each batch.

    max_retries : int
        Maximum number of retry attempts for API calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """Initialize the embedding generator.

        Parameters
        ----------
        api_key : str
            OpenAI API key for authentication.

        model : str, default="text-embedding-3-small"
            Name of the embedding model to use. Must be a valid OpenAI
            embedding model name.

        batch_size : int, default=100
            Number of texts to process in each batch. Larger batch sizes
            can improve throughput but may increase the risk of rate
            limiting.

        max_retries : int, default=3
            Maximum number of retry attempts for failed API calls.

        Examples
        --------
        >>> generator = EmbeddingGenerator(api_key="your-api-key")
        >>> embeddings = generator.generate_embeddings(["Hello, world!"])
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    @with_retry(max_retries=3, initial_delay=1.0, max_delay=10.0, backoff_factor=2.0, logger=logger)
    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts.

        This method processes a single batch of texts through the OpenAI
        API, with retry logic for handling temporary failures. It includes
        input validation and cleaning to ensure the API call succeeds.

        Parameters
        ----------
        texts : List[str]
            List of texts to generate embeddings for.

        Returns
        -------
        np.ndarray
            Array of embeddings, one for each input text.

        Raises
        ------
        Exception
            If the API call fails after all retry attempts.

        Examples
        --------
        >>> generator = EmbeddingGenerator(api_key="your-api-key")
        >>> embeddings = generator._get_embeddings_batch(["Hello", "World"])
        >>> embeddings.shape
        (2, 1536)  # For text-embedding-3-small model
        """
        # Ensure texts is a list of strings and clean them
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [str(texts)]

        # Clean and validate texts
        cleaned_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            # Remove any problematic characters and ensure text is not empty
            text = text.strip()
            if not text:
                text = " "  # Use space for empty texts
            cleaned_texts.append(text)

        try:
            response = self.client.embeddings.create(model=self.model, input=cleaned_texts)
            return np.array([data.embedding for data in response.data])
        except Exception as e:
            logger.error(f"Error in _get_embeddings_batch: {str(e)}")
            logger.error(f"Input texts: {cleaned_texts}")
            raise

    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts.

        This method processes a list of texts in batches, generating
        embeddings for each text using the OpenAI API. It includes
        progress tracking and error handling.

        Parameters
        ----------
        texts : List[str]
            List of texts to generate embeddings for.

        show_progress : bool, default=True
            Whether to log progress information during processing.

        Returns
        -------
        np.ndarray
            Array of embeddings, one for each input text.

        Raises
        ------
        Exception
            If any batch fails after all retry attempts.

        Examples
        --------
        >>> generator = EmbeddingGenerator(api_key="your-api-key")
        >>> texts = ["First text", "Second text", "Third text"]
        >>> embeddings = generator.generate_embeddings(texts)
        >>> embeddings.shape
        (3, 1536)  # For text-embedding-3-small model
        """
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                batch_embeddings = self._get_embeddings_batch(batch)
                embeddings.extend(batch_embeddings)

                if show_progress:
                    current_batch = (i // self.batch_size) + 1
                    # Only log batch completion with minimal information
                    logger.info(f"Processed batch {current_batch}/{total_batches}")

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                raise

        return np.array(embeddings)

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Alias for generate_embeddings for compatibility with SentenceTransformer API.

        This method provides compatibility with code that expects a .encode() method
        like SentenceTransformer models have.

        Parameters
        ----------
        texts : List[str]
            List of texts to generate embeddings for.
        **kwargs
            Additional arguments (ignored, kept for compatibility)

        Returns
        -------
        np.ndarray
            Array of embeddings, one for each input text.
        """
        return self.generate_embeddings(texts, show_progress=kwargs.get("show_progress_bar", False))
