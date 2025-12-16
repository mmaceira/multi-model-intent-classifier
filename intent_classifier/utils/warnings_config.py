"""Utility module for configuring warnings and logging globally."""

import logging
import os
import warnings
from collections.abc import Callable
from typing import Any, cast

# Suppress Pydantic warnings at the module level before any imports
# This must be done before pydantic is imported
os.environ.setdefault("PYDANTIC_WARNINGS", "none")

_ORIG_HANDLER: Callable[..., Any] | None = None


def suppress_pydantic_warnings() -> None:
    """Suppress verbose Pydantic serialization warnings.

    These warnings are not serious - they occur when Pydantic deserializes
    API responses that don't exactly match the expected schema, but the
    code still works correctly. They're just very verbose.

    The warnings come from litellm/openai client libraries when they deserialize
    API responses. We suppress them at multiple levels to ensure they're caught.
    """
    global _ORIG_HANDLER
    if _ORIG_HANDLER is None:
        _ORIG_HANDLER = warnings.showwarning

    def _filtered_showwarning(
        message: Any,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: Any = None,
    ) -> None:
        """Custom warning handler that filters out Pydantic serialization warnings."""
        # Check if this is a Pydantic warning we want to suppress
        if issubclass(category, UserWarning):
            msg_str = str(message)
            filename_str = str(filename) if filename else ""

            # Suppress if it's from pydantic or contains our target messages
            if (
                "pydantic" in filename_str.lower()
                or "PydanticSerializationUnexpectedValue" in msg_str
                or "Expected `Usage`" in msg_str
                or "serialized value may not be as expected" in msg_str
            ):
                return  # Suppress this warning

        # For all other warnings, use the original handler
        original_handler = cast(
            Callable[..., Any],
            _ORIG_HANDLER if _ORIG_HANDLER is not None else warnings.showwarning,
        )
        original_handler(message, category, filename, lineno, file, line)

    # Install our custom warning handler
    warnings.showwarning = _filtered_showwarning

    # Also set up filterwarnings as backup
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal")
    warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
    warnings.filterwarnings("ignore", message=".*Expected `Usage`.*")
    warnings.filterwarnings("ignore", message=".*serialized value may not be as expected.*")

    # Set environment variable as additional measure
    os.environ["PYDANTIC_WARNINGS"] = "none"


def configure_logging(level: str = "INFO", suppress_warnings: bool = True) -> None:
    """Configure logging with sensible defaults.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        suppress_warnings: Whether to suppress Pydantic warnings
    """
    if suppress_warnings:
        suppress_pydantic_warnings()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configuration
    )

    # Set specific loggers to be less verbose
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
