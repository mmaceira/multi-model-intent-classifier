"""Model Utility Functions

This module provides utility functions for working with serialized models in different formats.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Standard model file names to check
ALLOWED_MODEL_FILENAMES = [
    "model.pkl",
    "model.joblib",
    "classifier.pkl",
    "classifier.joblib",
    "pipeline.pkl",
    "pipeline.joblib",
]


def find_model_file(model_dir: Union[str, Path], model_name: str) -> Optional[str]:
    """Find a model file in the specified directory under the model name subdirectory.

    This function checks for various common model file formats (pkl, joblib) and
    returns the path to the first one found.

    Args:
        model_dir: Base directory containing model subdirectories
        model_name: Name of the model subdirectory

    Returns:
        Path to the model file as a string, or None if no model file is found

    Example:
        >>> path = find_model_file("models", "naive_bayes")
        >>> # Will look for models/naive_bayes/model.pkl, models/naive_bayes/model.joblib, etc.
    """
    base_dir = Path(model_dir) / model_name

    if not base_dir.exists():
        return None

    for filename in ALLOWED_MODEL_FILENAMES:
        file_path = base_dir / filename
        if file_path.exists():
            return str(file_path)

    return None


def scan_model_directory(model_dir: Union[str, Path]) -> List[str]:
    """Scan a directory for model subdirectories containing model files.

    Args:
        model_dir: Base directory to scan

    Returns:
        List of model names (subdirectory names) that contain valid model files
    """
    base_dir = Path(model_dir)
    if not base_dir.exists():
        return []

    model_names = []

    for item in base_dir.iterdir():
        if item.is_dir():
            for filename in ALLOWED_MODEL_FILENAMES:
                if (item / filename).exists():
                    model_names.append(item.name)
                    break

    return model_names


def load_model_paths(models: Dict[str, Any], model_dir: Union[str, Path]) -> Dict[str, str]:
    """Load model paths from model directory for all models in the provided dictionary.

    This function takes a dictionary of models (as returned by load_models_from_config)
    and finds the corresponding serialized model files in the model directory.

    Args:
        models: Dictionary of models with model names as keys
        model_dir: Base directory containing model subdirectories

    Returns:
        Dictionary mapping model names to their file paths

    Example:
        >>> from src.utils.model_loader import load_models_from_config
        >>> models = load_models_from_config()
        >>> model_paths = load_model_paths(models, "models")
        >>> # Result can be used with run_prediction
    """
    model_paths = {}
    for name in models.keys():
        path = find_model_file(model_dir, name)
        if path:
            model_paths[name] = path

    return model_paths
