"""
Path Utilities

This module provides a single source of truth for all paths in the project.
It discovers the repository root and provides consistent path resolution
across the codebase.

Example:
    >>> from intent_classifier.utils.paths import get_repo_root, get_output_dir
    >>> repo_root = get_repo_root()
    >>> output_dir = get_output_dir("experiment_name")
"""

import subprocess
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory.

    Tries multiple methods to find the repo root:
    1. git rev-parse --show-toplevel (if git is available)
    2. Look for .git directory by walking up from current file
    3. Fallback to current working directory

    Returns:
        Path to repository root

    Raises:
        RuntimeError: If repository root cannot be determined
    """
    # Try git command first (most reliable)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        repo_root = Path(result.stdout.strip())
        if repo_root.exists():
            return repo_root.resolve()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: walk up from this file to find .git directory
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / ".git").exists():
            return parent

    # Last resort: use current working directory
    # This is less reliable but better than failing
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        return cwd.resolve()

    # If all else fails, assume we're in src/utils and go up 2 levels
    return current_file.parent.parent.parent.resolve()


# Cache the repo root
_REPO_ROOT: Path | None = None


def _get_repo_root_cached() -> Path:
    """Get cached repository root."""
    global _REPO_ROOT
    if _REPO_ROOT is None:
        _REPO_ROOT = get_repo_root()
    return _REPO_ROOT


def get_config_dir() -> Path:
    """Get the config directory path.

    Returns:
        Path to config directory
    """
    return _get_repo_root_cached() / "config"


def get_config_path(config_file: str | Path) -> Path:
    """Get the path to a config file.

    Args:
        config_file: Config file path. Must be:
                    - Absolute path, or
                    - Path starting with "config/" (relative to repo root)
                    Examples: "config/dataset/clinc150/tiny.yaml"

    Returns:
        Path to config file

    Raises:
        ValueError: If config_file doesn't start with "config/" and isn't absolute
    """
    config_file = Path(config_file)
    if config_file.is_absolute():
        return config_file

    config_file_str = str(config_file)
    if not config_file_str.startswith("config/"):
        raise ValueError(
            f"Config file path must start with 'config/' or be absolute. "
            f"Got: {config_file_str}. "
            f"Example: config/dataset/clinc150/tiny.yaml"
        )

    return _get_repo_root_cached() / config_file_str


def get_output_dir(experiment_name: str = "default") -> Path:
    """Get the output directory for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Path to experiment output directory
    """
    return _get_repo_root_cached() / "output" / experiment_name


def get_models_dir(experiment_name: str | None = None) -> Path:
    """Get the models directory path.

    Args:
        experiment_name: Optional experiment name. If provided, returns
                        models dir within that experiment's output directory.
                        If None, returns the base models directory.

    Returns:
        Path to models directory
    """
    if experiment_name:
        return get_output_dir(experiment_name) / "models"
    return _get_repo_root_cached() / "models"


def get_embeddings_dir(experiment_name: str | None = None) -> Path:
    """Get the embeddings directory path.

    Args:
        experiment_name: Optional experiment name. If provided, returns
                        embeddings dir within that experiment's output directory.
                        If None, returns the base embeddings directory.

    Returns:
        Path to embeddings directory
    """
    if experiment_name:
        return get_output_dir(experiment_name) / "embeddings"
    return _get_repo_root_cached() / "embeddings"
