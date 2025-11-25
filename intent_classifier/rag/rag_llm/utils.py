"""Utility functions for RAG LLM classifier."""

import random


def exponential_backoff(attempt: int) -> float:
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
