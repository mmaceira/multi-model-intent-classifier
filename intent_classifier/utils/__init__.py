"""Utility functions for the CLINC150 RAG Classifier project."""

# Expose key functions at the module level for easier imports
from intent_classifier.utils.file_ops import ensure_dir  # noqa: F401
from intent_classifier.utils.method_logger import (  # noqa: F401
    MethodLogger,
    get_logger,
    log_method,
)


# Lazy import to avoid circular dependency
# model_loader imports algorithms which may import embeddings
def load_models_from_config(*args, **kwargs):
    """Lazy import wrapper to avoid circular dependencies."""
    from intent_classifier.utils.model_loader import (
        load_models_from_config as _load_models_from_config,
    )

    return _load_models_from_config(*args, **kwargs)


__all__ = ["ensure_dir", "load_models_from_config", "MethodLogger", "get_logger", "log_method"]
