"""Retry utilities for handling API calls and other operations.

This module provides a decorator for adding retry logic to functions that may
fail due to temporary issues like network problems or rate limiting. It
implements exponential backoff to prevent overwhelming the target system.

Functions:
- with_retry: Decorator that adds retry logic to a function
"""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that adds retry logic to a function.

    This decorator implements exponential backoff retry logic for functions
    that may fail due to temporary issues. It catches specified exceptions,
    waits for an increasing amount of time between retries, and logs the
    retry attempts if a logger is provided.

    Parameters
    ----------
    max_retries : int, default=3
        Maximum number of retry attempts before giving up.

    initial_delay : float, default=1.0
        Initial delay between retries in seconds.

    max_delay : float, default=10.0
        Maximum delay between retries in seconds. The delay will not exceed
        this value even with exponential backoff.

    backoff_factor : float, default=2.0
        Factor by which the delay increases after each retry. For example,
        with initial_delay=1 and backoff_factor=2, the delays will be:
        1s, 2s, 4s, 8s, etc. (capped at max_delay).

    exceptions : Union[Type[Exception], Tuple[Type[Exception], ...]], default=Exception
        Exception type(s) to catch and retry on. Can be a single exception
        type or a tuple of exception types.

    logger : Optional[logging.Logger], default=None
        Logger instance for logging retry attempts. If None, no logging
        will be performed.

    Returns
    -------
    Callable
        Decorated function with retry logic.

    Examples
    --------
    >>> import logging
    >>> logger = logging.getLogger(__name__)

    >>> @with_retry(max_retries=2, logger=logger)
    ... def unreliable_function():
    ...     if random.random() < 0.5:
    ...         raise ConnectionError("Temporary failure")
    ...     return "Success!"

    >>> @with_retry(
    ...     max_retries=3,
    ...     initial_delay=0.1,
    ...     exceptions=(ConnectionError, TimeoutError)
    ... )
    ... def api_call():
    ...     # Make API call here
    ...     pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        if logger:
                            logger.error(
                                f"Function {func.__name__} failed after {max_retries} retries. "
                                f"Last error: {e!s}"
                            )
                        raise

                    if logger:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries + 1} for {func.__name__}: {e!s}"
                        )

                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            if last_exception is not None:
                raise last_exception  # This should never be reached due to the raise in the loop
            raise RuntimeError("Unexpected: retry loop completed without exception or return")

        return wrapper

    return decorator
