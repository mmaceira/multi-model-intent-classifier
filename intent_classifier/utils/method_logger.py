"""
Method Logger Module

This module provides simple disk-based logging for all methods across the repository.
It logs method inputs, outputs (predictions), and errors to structured JSON files.

Key Features:
- Structured JSON logging to disk
- Automatic method call logging via decorators
- Simple logging of inputs, outputs, and errors
- Integration with existing config system
- Thread-safe logging

Classes:
- MethodLogger: Main logger class

Functions:
- log_method: Decorator for automatic method logging
- get_logger: Factory function to get configured logger instance

Example Usage:
    >>> from intent_classifier.utils.method_logger import log_method, get_logger
    >>>
    >>> # Use as decorator
    >>> @log_method
    >>> def predict(self, docs):
    ...     return self.model.predict(docs)
    >>>
    >>> # Manual logging
    >>> logger = get_logger()
    >>> logger.log_prediction(input="test query", output="label1")
"""

import json
import logging
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Module-level logger for this module itself
_module_logger = logging.getLogger(__name__)


class MethodLogger:
    """
    Simple logger for method inputs, outputs, and errors.

    Logs are written as JSON lines (JSONL format) for easy parsing.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, log_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize the method logger.

        Args:
            log_dir: Directory where logs will be written. If None, attempts to
                    determine from config.
            enabled: Whether logging is enabled. If False, all logging operations
                    are no-ops.
        """
        self.enabled = enabled
        self.log_dir = log_dir or self._get_log_dir_from_config()
        self._file_locks: Dict[str, threading.Lock] = {}

        if self.enabled and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            _module_logger.info(f"Method Logger initialized. Log directory: {self.log_dir}")
        elif not self.enabled:
            _module_logger.info("Method Logger disabled")
        else:
            _module_logger.warning("Method Logger: No log directory configured, logging disabled")

    @classmethod
    def get_instance(cls, log_dir: Optional[Path] = None, enabled: bool = True) -> "MethodLogger":
        """
        Get or create the singleton logger instance.

        Args:
            log_dir: Directory where logs will be written. Only used on first call.
            enabled: Whether logging is enabled. Only used on first call.

        Returns:
            MethodLogger: The singleton logger instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(log_dir=log_dir, enabled=enabled)
        return cls._instance

    @staticmethod
    def _get_log_dir_from_config() -> Optional[Path]:
        """Get log directory from config for saving logs."""
        try:
            # Try to get predictions_dir from config
            try:
                from intent_classifier.rag.adapter_sklearn import RagSklearnAdapter

                log_dir = RagSklearnAdapter._get_log_dir_from_config(
                    RagSklearnAdapter._get_config()
                )
                if log_dir:
                    return Path(log_dir)
            except Exception:
                pass

            # Fallback: try to get from config_vars
            try:
                from config.notebook_setup import config_vars

                predictions_dir = config_vars.get("PATHS_PREDICTIONS_DIR")
                if predictions_dir:
                    return Path(predictions_dir) / "method_logs"
            except Exception:
                pass

            # Final fallback: use current directory
            return Path.cwd() / "method_logs"
        except Exception as e:
            _module_logger.warning(f"Could not determine log directory: {e}")
            return None

    def _get_log_file(self, log_type: str) -> Path:
        """
        Get the log file path for a given log type.

        Args:
            log_type: Type of log ('method_calls', 'predictions', 'errors')

        Returns:
            Path: Path to the log file
        """
        if not self.log_dir:
            raise ValueError("Log directory not configured")

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{log_type}_{timestamp}.jsonl"
        return self.log_dir / filename

    def _write_log_entry(self, log_type: str, entry: Dict[str, Any]) -> None:
        """
        Write a log entry to the appropriate log file.

        Args:
            log_type: Type of log ('method_calls', 'predictions', 'errors')
            entry: Dictionary containing log data
        """
        if not self.enabled or not self.log_dir:
            return

        try:
            log_file = self._get_log_file(log_type)

            # Get or create file lock for this log type
            if log_type not in self._file_locks:
                with self._lock:
                    if log_type not in self._file_locks:
                        self._file_locks[log_type] = threading.Lock()

            # Write entry (thread-safe)
            with self._file_locks[log_type]:
                with open(log_file, "a", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False, default=self._json_serializer)
                    f.write("\n")
        except Exception as e:
            _module_logger.error(f"Error writing log entry: {e}", exc_info=True)

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy arrays and other non-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)

    def log_method_call(
        self,
        method_name: str,
        class_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        output: Optional[Any] = None,
        duration: Optional[float] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Log a method call with inputs and output.

        Args:
            method_name: Name of the method
            class_name: Name of the class (if applicable)
            inputs: Dictionary of input parameters
            output: Return value
            duration: Time taken in seconds
            error: Exception if the call failed
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
        }

        if class_name:
            entry["class"] = class_name

        if inputs:
            entry["inputs"] = {k: self._json_serializer(v) for k, v in inputs.items()}

        if output is not None:
            entry["output"] = self._json_serializer(output)

        if duration is not None:
            entry["duration_seconds"] = duration

        if error:
            entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            # Log errors to separate error log file
            self._write_log_entry("errors", entry)
        else:
            # Log successful calls
            self._write_log_entry("method_calls", entry)

    def log_prediction(
        self,
        input: Any,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a prediction (input and output).

        Args:
            input: Input to the prediction (e.g., query text, document)
            output: Prediction output (e.g., label, probabilities)
            metadata: Optional additional metadata
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": self._json_serializer(input),
            "output": self._json_serializer(output),
        }

        if metadata:
            entry["metadata"] = {k: self._json_serializer(v) for k, v in metadata.items()}

        self._write_log_entry("predictions", entry)

    def log_error(
        self,
        method_name: str,
        error: Exception,
        inputs: Optional[Dict[str, Any]] = None,
        class_name: Optional[str] = None,
    ) -> None:
        """
        Log an error.

        Args:
            method_name: Name of the method where error occurred
            error: The exception that was raised
            inputs: Optional input parameters when error occurred
            class_name: Optional class name
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

        if class_name:
            entry["class"] = class_name

        if inputs:
            entry["inputs"] = {k: self._json_serializer(v) for k, v in inputs.items()}

        self._write_log_entry("errors", entry)


def log_method(logger: Optional[MethodLogger] = None):
    """
    Decorator to automatically log method calls (inputs, outputs, errors).

    Usage:
        @log_method
        def predict(self, docs):
            return self.model.predict(docs)

        @log_method(logger=my_logger)
        def custom_method(self, arg1, arg2):
            return result

    Args:
        logger: Optional logger instance. If None, uses singleton instance.

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = MethodLogger.get_instance()

            if not logger.enabled:
                return func(*args, **kwargs)

            # Extract class name if method is bound
            class_name = None
            if args and hasattr(args[0], "__class__"):
                class_name = args[0].__class__.__name__

            # Prepare inputs dictionary
            inputs = {}
            if args:
                # Skip 'self' if bound method, otherwise include all args
                if class_name:
                    # For bound methods, skip self and name remaining args
                    for i, arg in enumerate(args[1:], 1):
                        inputs[f"arg{i}"] = arg
                else:
                    # For unbound functions, include all args
                    for i, arg in enumerate(args):
                        inputs[f"arg{i}"] = arg

            if kwargs:
                inputs.update(kwargs)

            start_time = time.time()
            error = None
            output = None

            try:
                output = func(*args, **kwargs)
                return output
            except Exception as e:
                error = e
                raise
            finally:
                duration = time.time() - start_time
                logger.log_method_call(
                    method_name=func.__name__,
                    class_name=class_name,
                    inputs=inputs,
                    output=output,
                    duration=duration,
                    error=error,
                )

        return wrapper

    # Support both @log_method and @log_method() usage
    # If called without parentheses, func is passed directly
    import inspect

    if inspect.isfunction(logger) or inspect.ismethod(logger):
        # Called as @log_method (without parentheses)
        func = logger
        logger = None
        return decorator(func)
    else:
        # Called as @log_method() or @log_method(logger=...)
        return decorator


def get_logger(log_dir: Optional[Path] = None, enabled: bool = True) -> MethodLogger:
    """
    Get or create a method logger instance.

    Args:
        log_dir: Optional log directory. If None, attempts to determine from config.
        enabled: Whether logging is enabled.

    Returns:
        MethodLogger: Logger instance
    """
    return MethodLogger.get_instance(log_dir=log_dir, enabled=enabled)
