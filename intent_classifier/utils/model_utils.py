"""Model Utility Functions

This module provides utility functions for working with serialized models in different formats.
"""

from pathlib import Path
from typing import Any

from intent_classifier.utils.file_ops import sanitize_model_name

# Standard model file names to check
ALLOWED_MODEL_FILENAMES = [
    "model.pkl",
    "model.joblib",
    "classifier.pkl",
    "classifier.joblib",
    "pipeline.pkl",
    "pipeline.joblib",
]


def find_model_file(model_dir: str | Path, model_name: str) -> str | None:
    """Find a model file in the specified directory under the model name subdirectory.

    This function checks for various common model file formats (pkl, joblib) and
    returns the path to the first one found.

    Args:
        model_dir: Base directory containing model subdirectories
        model_name: Name of the model subdirectory (will be sanitized for filesystem use)

    Returns:
        Path to the model file as a string, or None if no model file is found

    Example:
        >>> path = find_model_file("models", "naive_bayes")
        >>> # Will look for models/naive_bayes/model.pkl, models/naive_bayes/model.joblib, etc.
    """
    # Sanitize model name to match how directories are created
    safe_dir_name = sanitize_model_name(model_name)
    base_dir = Path(model_dir) / safe_dir_name

    if not base_dir.exists():
        return None

    for filename in ALLOWED_MODEL_FILENAMES:
        file_path = base_dir / filename
        if file_path.exists():
            return str(file_path)

    return None


def load_model_paths(models: dict[str, Any], model_dir: str | Path) -> dict[str, str]:
    """Load model paths from model directory for all models in the provided dictionary.

    This function takes a dictionary of models (as returned by load_models_from_config)
    and finds the corresponding serialized model files in the model directory.

    Args:
        models: Dictionary of models with model names as keys
        model_dir: Base directory containing model subdirectories

    Returns:
        Dictionary mapping model names to their file paths

    Example:
        >>> from intent_classifier.utils.model_loader import load_models_from_config
        >>> models = load_models_from_config()
        >>> model_paths = load_model_paths(models, "models")
        >>> # Result can be used with run_prediction
    """
    model_paths = {}
    for name in models:
        path = find_model_file(model_dir, name)
        if path:
            model_paths[name] = path

    return model_paths
