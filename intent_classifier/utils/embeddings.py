"""Embedding generation utilities.

This module provides functionality for generating text embeddings using:

- OpenAI's API (via the official SDK)
- Ollama-hosted embedding models (via the native HTTP API, treated as an embedding backend)

It includes batch processing capabilities and retry logic for handling
API rate limits and temporary failures where appropriate.

Classes
-------
- EmbeddingGenerator: Handles the generation of text embeddings using OpenAI's API
- LitellmOllamaEmbedder: SentenceTransformer-compatible wrapper around the Ollama
  HTTP embeddings API (e.g. qwen3-embedding:latest)
"""

import logging
import os
from typing import Any

import numpy as np
from openai import OpenAI

from intent_classifier.utils.retry import with_retry

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Error raised when an external embedding service (e.g. Ollama) is unavailable.

    This is used to signal *recoverable* infrastructure issues (HTTP 4xx/5xx,
    connection errors, timeouts, etc.) so that higher-level pipelines can
    decide to **skip** the affected model without failing the entire run.
    """


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
    def _get_embeddings_batch(self, texts: list[str]) -> np.ndarray:
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
            logger.error(f"Error in _get_embeddings_batch: {e!s}")
            logger.error(f"Input texts: {cleaned_texts}")
            raise

    def generate_embeddings(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
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
                logger.error(f"Error processing batch: {e!s}")
                raise

        return np.array(embeddings)

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
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


class LitellmOllamaEmbedder:
    """Embedder for Ollama-hosted embedding models.

    This wrapper exposes a SentenceTransformer-compatible ``encode`` method so it can
    be dropped into places that expect a local encoder, while internally delegating
    to Ollama's native HTTP ``/api/embeddings`` endpoint.

    Parameters
    ----------
    model : str
        Embedding model identifier (e.g. ``\"qwen3-embedding:latest\"`` or
        ``\"ollama/qwen3-embedding:latest\"``). Any optional ``\"ollama/\"`` prefix
        is stripped before calling the Ollama API.
    base_url : str | None, optional
        Base URL for the Ollama HTTP endpoint. If ``None``, will rely on litellm's
        default resolution (including ``OLLAMA_API_BASE`` / ``OLLAMA_HOST``).
    batch_size : int, default=32
        Number of texts to send per embedding request.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        batch_size: int = 32,
    ) -> None:
        # Normalise model name for Ollama API (strip optional "ollama/" prefix)
        self.model = model.replace("ollama/", "")
        # Resolve base URL: explicit arg > standard env vars > config‑derived env > localhost
        # Priority:
        #   1) Explicit base_url argument
        #   2) OLLAMA_API_BASE (standard Ollama env var)
        #   3) OLLAMA_HOST (alternate standard env var)
        #   4) MODEL_OLLAMA_ENDPOINT (exported from llm_config + dataset config)
        #   5) Fallback: http://localhost:11434
        resolved_base = (
            base_url
            or os.getenv("OLLAMA_API_BASE")
            or os.getenv("OLLAMA_HOST")
            or os.getenv("MODEL_OLLAMA_ENDPOINT")
        )
        if resolved_base is None:
            resolved_base = "http://localhost:11434"
        if not resolved_base.startswith("http://") and not resolved_base.startswith("https://"):
            resolved_base = f"http://{resolved_base}"
        self.base_url = resolved_base.rstrip("/")
        self.batch_size = batch_size

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        clean_texts: list[str] = []
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            t = t.strip()
            clean_texts.append(t or " ")

        # Call Ollama embeddings API once per text (API currently expects a single prompt)
        import json
        from urllib import error, request

        url = f"{self.base_url}/api/embeddings"
        vectors: list[np.ndarray] = []

        for text in clean_texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with request.urlopen(req, timeout=600) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
            except error.HTTPError as exc:
                # HTTP-level errors (e.g. 404 when /api/embeddings is not available)
                logger.error("HTTP error calling Ollama embeddings at %s: %s", url, exc)
                raise EmbeddingServiceError(f"Ollama embeddings HTTP error: {exc}") from exc
            except error.URLError as exc:
                # Connection issues, DNS failures, refusals, etc.
                logger.error("Error calling Ollama embeddings at %s: %s", url, exc)
                raise EmbeddingServiceError(f"Ollama embeddings connection error: {exc}") from exc
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unexpected response from Ollama embeddings at %s: %s", url, exc)
                raise EmbeddingServiceError(f"Ollama embeddings unexpected error: {exc}") from exc

            if "embedding" not in data:
                logger.error("Ollama embedding response missing 'embedding' field: %r", data)
                raise EmbeddingServiceError(
                    "Invalid Ollama embeddings response (no 'embedding' field)"
                )

            vectors.append(np.asarray(data["embedding"], dtype="float32"))

        return np.vstack(vectors)

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool | None = None,  # kept for API compatibility, currently ignored
        **_: Any,
    ) -> np.ndarray:
        """Encode a list of texts into embeddings.

        The signature mirrors SentenceTransformer.encode so existing call sites can
        pass ``batch_size`` and ``show_progress_bar`` without errors.
        """
        if not texts:
            return np.empty((0, 0), dtype="float32")

        bs = batch_size or self.batch_size
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            all_vecs.append(self._embed_batch(batch))

        # Ensure we return a single 2D array
        return np.vstack(all_vecs)
