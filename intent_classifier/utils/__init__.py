"""Utility functions for the CLINC150 RAG Classifier project."""

from typing import Any

# Expose key functions at the module level for easier imports
from intent_classifier.utils.file_ops import ensure_dir
from intent_classifier.utils.label_utils import (
    binarize_labels,
    is_multilabel,
    multilabel_predictions_from_binary,
    multilabel_predictions_from_proba,
    to_multilabel_format,
    to_singlelabel_format,
)
from intent_classifier.utils.method_logger import (
    MethodLogger,
    get_logger,
    log_method,
)


# Lazy import to avoid circular dependency
# model_loader imports algorithms which may import embeddings
def load_models_from_config(*args: Any, **kwargs: Any) -> Any:
    """Lazy import wrapper to avoid circular dependencies."""
    from intent_classifier.utils.model_loader import (
        load_models_from_config as _load_models_from_config,
    )

    return _load_models_from_config(*args, **kwargs)


__all__ = [
    "ensure_dir",
    "load_models_from_config",
    "MethodLogger",
    "get_logger",
    "log_method",
    "is_multilabel",
    "to_multilabel_format",
    "to_singlelabel_format",
    "binarize_labels",
    "multilabel_predictions_from_proba",
    "multilabel_predictions_from_binary",
]
