"""Retry utilities for handling API calls and other operations."""

import time
import logging
from functools import wraps
from typing import Callable, Type, Union, Tuple, Optional

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """Decorator that adds retry logic to a function.
    
    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retry attempts, by default 3
    initial_delay : float, optional
        Initial delay between retries in seconds, by default 1.0
    max_delay : float, optional
        Maximum delay between retries in seconds, by default 10.0
    backoff_factor : float, optional
        Factor by which the delay increases after each retry, by default 2.0
    exceptions : Union[Type[Exception], Tuple[Type[Exception], ...]], optional
        Exception type(s) to catch and retry on, by default Exception
    logger : Optional[logging.Logger], optional
        Logger instance for logging retry attempts, by default None
        
    Returns
    -------
    Callable
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        if logger:
                            logger.error(
                                f"Function {func.__name__} failed after {max_retries} retries. "
                                f"Last error: {str(e)}"
                            )
                        raise
                    
                    if logger:
                        logger.warning(
                            f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                        )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception  # This should never be reached due to the raise in the loop
        
        return wrapper
    return decorator 