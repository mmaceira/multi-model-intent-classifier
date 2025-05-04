"""OpenAIEmbeddings module.

Provides a thin wrapper around the OpenAI embeddings endpoint with transparent batching.

Classes:
- OpenAIEmbedder

Functions:
- chunk_iterable

Created: 2025-05-03
"""

from __future__ import annotations
import os
import itertools
import time
import logging
import random
import backoff
from typing import Iterable, List

try:
    import openai
    from openai import RateLimitError
except ImportError as e:
    raise ImportError("openai package required. Install with `pip install openai`.") from e

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: str | None = None,
        organization: str | None = None,
        max_retries: int = 5,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if organization:
            openai.organization = organization
        self._client = openai

    def _backoff_hdlr(self, details):
        """Handler called on backoff before each retry."""
        logger.warning(
            f"OpenAI API embeddings call failed. Retrying in {details['wait']:.2f}s after {details['tries']} tries. "
            f"Error: {details['exception']}"
        )

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, openai.RateLimitError),
        max_tries=5,  # Max retries
        factor=1.5,   # Exponential backoff factor
        jitter=backoff.full_jitter,  # Add jitter for distributed systems
    )
    def _create_embeddings_with_retry(self, batch):
        """Make an OpenAI embeddings API call with retry logic for rate limits"""
        try:
            return self._client.embeddings.create(input=batch, model=self.model)
        except (RateLimitError, openai.RateLimitError) as e:
            # Extract wait time from error message if available
            wait_time = 0.5  # Default wait time if we can't parse the message
            try:
                # Try to extract wait time from error message
                if hasattr(e, 'message') and 'try again in' in e.message.lower():
                    import re
                    match = re.search(r'try again in (\d+)ms', e.message.lower())
                    if match:
                        wait_ms = int(match.group(1))
                        wait_time = wait_ms / 1000.0 + 0.1  # Add a small buffer
            except:
                pass
            
            logger.warning(f"Rate limit exceeded for embeddings. Waiting for {wait_time:.2f}s before retry")
            time.sleep(wait_time)
            raise  # Re-raise to be caught by backoff

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors aligned with *texts*."""
        # Start timing the entire encoding process
        start_time = time.time()
        logger.info(f"Starting OpenAI embedding generation for {len(texts)} texts with model {self.model}")
        
        embeddings: List[List[float]] = []
        batch_count = 0
        total_token_count = 0  # This is just an estimate
        
        for batch in chunk_iterable(texts, self.batch_size):
            batch_count += 1
            batch_size = len(batch)
            batch_start_time = time.time()
            
            logger.info(f"Processing embedding batch {batch_count} with {batch_size} texts")
            
            # Rough token estimation (4 chars ~= 1 token)
            batch_token_estimate = sum(len(text) // 4 for text in batch)
            total_token_count += batch_token_estimate
            
            try:
                # Use the retry-enabled method instead of direct API call
                response = self._create_embeddings_with_retry(batch)
                
                # Get actual token usage if available in the response
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    actual_tokens = response.usage.total_tokens
                    logger.info(f"Batch {batch_count} used {actual_tokens} tokens")
                
                # Using .data list ensures order preserved
                embeddings.extend([d.embedding for d in response.data])
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                logger.info(f"Batch {batch_count} completed in {batch_duration:.2f} seconds")
                
                if batch_size > 0:
                    logger.info(f"Average time per text in batch: {batch_duration/batch_size:.4f} seconds")
                
                # Add a small delay between batches to avoid rate limits
                if batch_count < (len(texts) + self.batch_size - 1) // self.batch_size:
                    delay = random.uniform(0.2, 0.5)
                    logger.info(f"Adding delay of {delay:.2f}s before next batch")
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_count}: {str(e)}")
                # If we've completely failed after retries, raise the exception
                raise
        
        # Calculate and log total embedding time
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Total OpenAI embedding time for {len(texts)} texts: {total_duration:.2f} seconds")
        
        if len(texts) > 0:
            logger.info(f"Average time per text: {total_duration/len(texts):.4f} seconds")
        
        logger.info(f"Processed {batch_count} batches with approximately {total_token_count} tokens")
        
        return embeddings
