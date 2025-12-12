"""
Centralized Configuration Loader

This module provides a single source of truth for loading and processing
YAML configuration files with variable substitution, validation, and caching.

The module uses TypedDict (ConfigMetadata) for type-safe configuration metadata
returned by load_config_with_metadata().

Example:
    >>> from intent_classifier.utils.config_loader import load_config, load_config_with_metadata
    >>> config = load_config("config/dataset/clinc150/tiny.yaml")
    >>> print(config["general"]["run_name"])
    >>>
    >>> # With metadata (type-safe)
    >>> metadata = load_config_with_metadata("config/dataset/clinc150/tiny.yaml")
    >>> print(metadata["dataset_name"])  # Type checker knows this is a str
    >>> print(metadata["label_type"])    # Type checker knows this is "singlelabel" | "multilabel"
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml

from intent_classifier.utils.paths import get_config_path, get_repo_root

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)


# Type definitions for configuration structures
class ConfigMetadata(TypedDict):
    """Metadata returned with loaded configuration."""

    config: dict[str, Any]
    dataset_name: str
    config_name: str
    label_type: Literal["singlelabel", "multilabel"]
    config_path: Path
    config_file: str


# Cache for loaded configs to avoid re-reading files
_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def substitute_vars(value: Any, config: dict[str, Any]) -> Any:
    """Replace variable references in string values with their actual values from config.

    Supports ${section.var} syntax for variable substitution.
    If a variable is not found, it remains as-is (caller should handle fallbacks).

    Args:
        value: Value that may contain variable references
        config: Configuration dictionary to look up variables

    Returns:
        Value with variables substituted (or original if substitution failed)

    Example:
        >>> config = {"general": {"run_name": "experiment1"}}
        >>> substitute_vars("output/${general.run_name}/models", config)
        'output/experiment1/models'
    """
    if isinstance(value, str) and "${" in value:
        var_pattern = r"\${([^}]+)}"
        original_value = value
        for var_path in re.findall(var_pattern, value):
            if "." in var_path:
                section, var = var_path.split(".", 1)
                if section in config and var in config[section]:
                    replacement = str(config[section][var])
                    value = value.replace(f"${{{var_path}}}", replacement)
                else:
                    # Variable not found - log warning but keep original
                    logger.debug(
                        f"Variable ${{{var_path}}} not found in config "
                        f"(section={section}, var={var}). Keeping original value."
                    )
        # If no substitutions were made and value still contains ${, return original
        if value == original_value and "${" in value:
            logger.debug(f"Could not substitute variables in: {value}")
        return value
    return value


def process_config_vars(config_value: Any, main_config: dict[str, Any]) -> Any:
    """Recursively process a configuration value to substitute variables.

    Args:
        config_value: Configuration value (can be dict, list, str, or other)
        main_config: Main configuration dictionary for variable lookup

    Returns:
        Configuration value with all variables substituted
    """
    if isinstance(config_value, dict):
        return {k: process_config_vars(v, main_config) for k, v in config_value.items()}
    elif isinstance(config_value, list):
        return [process_config_vars(v, main_config) for v in config_value]
    elif isinstance(config_value, str):
        return substitute_vars(config_value, main_config)
    return config_value


def path_to_config_file_str(config_path: Path) -> str:
    """Convert a Path object to a config file string format expected by load_config.

    Converts absolute paths to relative paths starting with "config/".
    This is the canonical format expected by load_config() and other config functions.

    Args:
        config_path: Path object (absolute or relative) to a config file

    Returns:
        Config file path string starting with "config/" (e.g., "config/dataset/clinc150/tiny.yaml")
        If config_path is absolute and can't be made relative, returns absolute path as string

    Raises:
        ValueError: If config_path is relative but doesn't start with "config/"
    """
    repo_root = get_repo_root()

    if config_path.is_absolute():
        try:
            # Convert absolute path to relative path string starting with "config/"
            rel_path = config_path.relative_to(repo_root)
            config_file_str = str(rel_path)
            # Verify it starts with "config/" (it should, since get_config_path enforces this)
            if not config_file_str.startswith("config/"):
                raise ValueError(
                    f"Config path relative to repo root should start with 'config/', "
                    f"got: {config_file_str}. Absolute path was: {config_path}"
                )
            return config_file_str
        except ValueError as e:
            # If can't make relative (shouldn't happen for valid config paths),
            # fall back to absolute
            logger.warning(
                f"Could not convert absolute config path to relative: {e}. Using absolute path."
            )
            return str(config_path)
    else:
        # Path is already relative, convert to string
        config_file_str = str(config_path)
        # Ensure it starts with "config/" (it should already from get_config_path, but verify)
        if not config_file_str.startswith("config/"):
            raise ValueError(
                f"Config path should start with 'config/', got: {config_file_str}. "
                f"Use get_config_path() first to normalize the path."
            )
        return config_file_str


def discover_config_file() -> str:
    """Discover the default config file to use.

    Tries multiple strategies:
    1. CONFIG_FILE environment variable
    2. First available dataset config (tiny.yaml or default.yaml)
    3. Fallback to config/dataset/clinc150/tiny.yaml

    Returns:
        Config file path starting with "config/" (e.g., "config/dataset/clinc150/tiny.yaml")
    """
    # Check environment variable first
    config_file = os.environ.get("CONFIG_FILE")
    if config_file:
        return config_file

    # Try to find first available dataset config
    repo_root = get_repo_root()
    dataset_dir = repo_root / "config" / "dataset"
    if dataset_dir.exists():
        dataset_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if dataset_dirs:
            first_dataset = sorted(dataset_dirs)[0].name
            # Try tiny.yaml first, fallback to default.yaml
            if (dataset_dir / first_dataset / "tiny.yaml").exists():
                return f"config/dataset/{first_dataset}/tiny.yaml"
            elif (dataset_dir / first_dataset / "default.yaml").exists():
                return f"config/dataset/{first_dataset}/default.yaml"

    # Final fallback
    return "config/dataset/clinc150/tiny.yaml"


def parse_config_path(config_file: str) -> tuple[str, str]:
    """Parse config file path to extract dataset name and config name.

    Args:
        config_file: Config file path. Must be:
                    - "config/dataset/clinc150/tiny.yaml" (relative to repo root)
                    - Absolute path ending with config/dataset/.../file.yaml

    Returns:
        Tuple of (dataset_name, config_name)

    Raises:
        ValueError: If config file path doesn't match expected structure
    """
    repo_root = get_repo_root()
    config_path = get_config_path(config_file)

    try:
        # Get path relative to repo root
        rel_path = config_path.relative_to(repo_root)
        parts = rel_path.parts

        # Format: config/dataset/name/file.yaml
        if len(parts) >= 4 and parts[0] == "config" and parts[1] == "dataset":
            dataset_name = parts[2]
            config_name = config_path.stem
            return dataset_name, config_name

        # If we get here, the path doesn't match expected structure
        raise ValueError(
            f"Config file must be in structure: "
            f"config/dataset/{{dataset_name}}/{{config_name}}.yaml\n"
            f"Got: {config_file}"
        )
    except ValueError as e:
        if "Config file must be" in str(e):
            raise
        rel_path = (
            config_path.relative_to(repo_root / "config")
            if config_path.is_relative_to(repo_root / "config")
            else config_path
        )
        raise ValueError(
            f"Config file must be in structure: "
            f"config/dataset/{{dataset_name}}/{{config_name}}.yaml\n"
            f"Got: {rel_path}"
        ) from e


def _load_llm_config() -> dict[str, Any] | None:
    """Load the centralized LLM configuration file.

    Returns:
        LLM configuration dictionary, or None if file doesn't exist
    """
    try:
        repo_root = get_repo_root()
        llm_config_path = repo_root / "config" / "llm_config.yaml"
        if llm_config_path.exists():
            with open(llm_config_path, encoding="utf-8") as f:
                llm_config = yaml.safe_load(f)
            if isinstance(llm_config, dict):
                logger.debug(f"Loaded LLM config from: {llm_config_path}")
                return llm_config
    except Exception as e:
        logger.debug(f"Could not load LLM config: {e}")
    return None


def detect_label_type(
    config: dict[str, Any], dataset_name: str | None = None
) -> Literal["singlelabel", "multilabel"]:
    """Detect label type (singlelabel or multilabel) from config.

    Args:
        config: Configuration dictionary
        dataset_name: Optional dataset name for fallback detection

    Returns:
        "singlelabel" or "multilabel"
    """
    if "dataset" in config and "multilabel" in config["dataset"]:
        return "multilabel" if config["dataset"]["multilabel"] else "singlelabel"

    # Default to singlelabel if multilabel is not specified
    # Individual dataset loaders will enforce their own requirements
    return "singlelabel"


def load_config(
    config_file: str | None = None,
    apply_variable_substitution: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load and process a YAML configuration file.

    Args:
        config_file: Config file path starting with "config/" (e.g.,
                    "config/dataset/clinc150/tiny.yaml"). If None, uses
                    discover_config_file() to find default.
        apply_variable_substitution: If True, applies variable substitution to config values
        use_cache: If True, caches loaded configs to avoid re-reading files

    Returns:
        Configuration dictionary with variables substituted

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    # Discover config file if not provided
    if config_file is None:
        config_file = discover_config_file()

    # Check cache
    cache_key = f"{config_file}:{apply_variable_substitution}"
    if use_cache and cache_key in _CONFIG_CACHE:
        logger.debug(f"Using cached config: {config_file}")
        return _CONFIG_CACHE[cache_key]

    # Resolve config path
    config_path = get_config_path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Config files must be in structure: "
            f"config/dataset/{{dataset_name}}/{{config_name}}.yaml"
        )

    # Load YAML file
    logger.debug(f"Loading config from: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML dictionary, got {type(config)}")

    # Load and merge LLM config (centralized LLM provider settings)
    llm_config = _load_llm_config()
    if llm_config:
        # Merge LLM config into model section for backward compatibility
        if "model" not in config:
            config["model"] = {}

        # Merge Ollama settings
        if "ollama" in llm_config:
            ollama_cfg = llm_config["ollama"]
            # Set llm_model from ollama.default_model if not already set
            if "llm_model" not in config["model"] and "default_model" in ollama_cfg:
                config["model"]["llm_model"] = ollama_cfg["default_model"]
            # Set ollama_endpoint from ollama.endpoint if not already set
            if "ollama_endpoint" not in config["model"] and "endpoint" in ollama_cfg:
                config["model"]["ollama_endpoint"] = ollama_cfg["endpoint"]

        # Merge OpenAI settings
        if "openai" in llm_config:
            openai_cfg = llm_config["openai"]
            # Set openai_model_name if not already set
            if "openai_model_name" not in config["model"] and "embedding_model" in openai_cfg:
                config["model"]["openai_model_name"] = openai_cfg["embedding_model"]

    # Apply variable substitution if requested
    if apply_variable_substitution:
        # Process all values recursively
        for section_key, section_value in config.items():
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    if isinstance(value, str) and "${" in value:
                        config[section_key][key] = substitute_vars(value, config)
        # Also process nested structures
        config = process_config_vars(config, config)

    # Cache the result
    if use_cache:
        _CONFIG_CACHE[cache_key] = config

    return config  # type: ignore[no-any-return]


def load_config_with_metadata(
    config_file: str | None = None,
    apply_variable_substitution: bool = True,
) -> ConfigMetadata:
    """Load config and return it with metadata (dataset_name, config_name, label_type).

    Args:
        config_file: Config file path starting with "config/" (e.g.,
                    "config/dataset/clinc150/tiny.yaml"). If None, uses
                    discover_config_file() to find default.
        apply_variable_substitution: If True, applies variable substitution to config values

    Returns:
        ConfigMetadata dictionary containing:
        - "config": The loaded configuration dictionary
        - "dataset_name": Name of the dataset
        - "config_name": Name of the config (without .yaml)
        - "label_type": "singlelabel" or "multilabel"
        - "config_path": Path to the config file
        - "config_file": Original config file path
    """
    # Discover config file if not provided
    if config_file is None:
        config_file = discover_config_file()

    # Parse metadata first (before loading, so we can set run_name early)
    dataset_name, config_name = parse_config_path(config_file)
    config_path = get_config_path(config_file)

    # Load config without substitution first
    config = load_config(config_file, apply_variable_substitution=False)

    # Override run_name with config file name BEFORE variable substitution
    # This ensures paths use the config name instead of the original run_name
    if "general" in config and "run_name" in config["general"]:
        config["general"]["run_name"] = config_name

    # Now apply variable substitution with the updated run_name
    if apply_variable_substitution:
        config = process_config_vars(config, config)

    # Detect label type
    label_type = detect_label_type(config, dataset_name)

    return {
        "config": config,
        "dataset_name": dataset_name,
        "config_name": config_name,
        "label_type": label_type,
        "config_path": config_path,
        "config_file": config_file,
    }


def get_first_available_dataset() -> str | None:
    """Get the name of the first available dataset from config/dataset directory.

    Returns:
        Name of the first dataset directory found, or None if no datasets exist
    """
    try:
        repo_root = get_repo_root()
        dataset_dir = repo_root / "config" / "dataset"
        if dataset_dir.exists():
            dataset_dirs = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
            if dataset_dirs:
                return sorted(dataset_dirs)[0]
    except Exception:
        pass
    return None


def get_run_name_from_config(config_file: str | None = None) -> str | None:
    """Extract run_name from a config file.

    Args:
        config_file: Config file path. If None, uses discover_config_file()

    Returns:
        run_name from config, or None if not found or config can't be loaded
    """
    try:
        if config_file is None:
            config_file = discover_config_file()
        config = load_config(config_file, apply_variable_substitution=False)
        return config.get("general", {}).get("run_name")  # type: ignore[no-any-return]
    except Exception:
        return None
