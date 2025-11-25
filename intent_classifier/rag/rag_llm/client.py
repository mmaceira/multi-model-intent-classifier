"""LLM API client with retry logic and rate limiting."""

import asyncio
import logging

from litellm import acompletion
from litellm.exceptions import RateLimitError

from .utils import exponential_backoff

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for making LLM API calls with robust error handling.

    Features:
    - Rate limiting with minimum interval enforcement
    - Exponential backoff with jitter for retries
    - Detailed error logging
    """

    def __init__(
        self,
        model: str,
        max_retries: int = 5,
        rate_limit_per_minute: int = 120,
    ):
        """Initialize the LLM client.

        Args:
            model: LLM model identifier
            max_retries: Maximum number of retry attempts for failed API calls
            rate_limit_per_minute: Maximum API calls per minute
        """
        self.model = model
        self.max_retries = max_retries
        # Calculate minimum interval between API calls based on rate limit
        self._min_interval = 60.0 / rate_limit_per_minute

    async def chat_with_retry(self, messages):
        """Send a request to the LLM API with robust error handling.

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
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                }

                # Handle JSON format based on model type
                if self.model.startswith("ollama/"):
                    # Ollama uses format="json" parameter for JSON mode
                    completion_params["format"] = "json"
                else:
                    # Other models use response_format
                    completion_params["response_format"] = {"type": "json_object"}

                # Make the API call using litellm
                response = await acompletion(**completion_params)
                return response

            except RateLimitError as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.error(
                        f"❌ Rate Limit Error: Exceeded after {attempt} attempts. "
                        f"Model: {self.model}. Error: {e}"
                    )
                    raise
                delay = exponential_backoff(attempt)
                logger.warning(
                    f"⚠️  Rate Limit Hit: Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{self.max_retries}). Model: {self.model}"
                )
                await asyncio.sleep(delay)

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.error(
                        f"❌ LLM API Error: Failed after {attempt} attempts. "
                        f"Model: {self.model}. Error type: {type(e).__name__}. Error: {e}"
                    )
                    raise
                delay = exponential_backoff(attempt)
                logger.warning(
                    f"⚠️  LLM API Error: Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{self.max_retries}). Model: {self.model}. "
                    f"Error: {type(e).__name__}: {str(e)[:100]}"
                )
                await asyncio.sleep(delay)
