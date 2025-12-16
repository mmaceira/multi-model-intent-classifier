"""Common file-system utilities used across the project.

This module provides centralized file-system operations to ensure consistency
across the project and avoid circular imports. It includes functions for
directory management and file path handling.

Functions:
- ensure_dir: Create directory path if it doesn't exist
- sanitize_model_name: Convert model names to filesystem-safe directory names
"""

import os
from pathlib import Path


def sanitize_model_name(name: str) -> str:
    """Convert a model name to a filesystem-safe directory name.

    Replaces path separators and other problematic characters with underscores
    to prevent nested directory creation.

    Parameters
    ----------
    name : str
        The model name to sanitize.

    Returns
    -------
    str
        A sanitized version of the name safe for use as a directory name.

    Examples
    --------
    >>> sanitize_model_name("Embedding + LogReg (Qwen/Ollama)")
    'Embedding + LogReg (Qwen_Ollama)'
    >>> sanitize_model_name("RAG-LLM (TF-IDF, default prompt)")
    'RAG-LLM (TF-IDF, default prompt)'
    """
    # Replace path separators with underscores
    sanitized = name.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    return sanitized


def ensure_dir(path: str | Path) -> Path:
    """Create directory path (including parents) if it does not exist and return a Path object.

    This is a centralized implementation to avoid subtle inconsistencies and
    circular imports when each module defines its own helper. The function
    handles both string and Path inputs, and creates all necessary parent
    directories.

    Parameters
    ----------
    path : Union[str, Path]
        The directory path to ensure exists. Can be either a string or a Path object.

    Returns
    -------
    Path
        The Path object pointing to the directory. This can be used for further
        path operations.

    Raises
    ------
    OSError
        If the directory cannot be created due to permission issues or other
        system-level problems.

    Examples
    --------
    >>> ensure_dir("output/results")
    PosixPath('output/results')

    >>> ensure_dir(Path("output/results"))
    PosixPath('output/results')

    >>> path = ensure_dir("output/results")
    >>> path / "file.txt"
    PosixPath('output/results/file.txt')
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
