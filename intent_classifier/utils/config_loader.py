"""
Centralized Configuration Loader

This module provides a single source of truth for loading and processing
layered YAML configuration files with variable substitution, validation,
and caching.

The module uses TypedDict (ConfigMetadata) for type-safe configuration metadata
returned by load_config_with_metadata().

Example:
    >>> from intent_classifier.utils.config_loader import load_config, load_config_with_metadata
    >>> config = load_config("config/experiments/clinc150/tiny.yaml")
    >>> print(config["general"]["run_name"])
    >>>
    >>> # With metadata (type-safe)
    >>> metadata = load_config_with_metadata("config/experiments/clinc150/tiny.yaml")
    >>> print(metadata["dataset_name"])  # Type checker knows this is a str
    >>> print(metadata["label_type"])    # Type checker knows this is "singlelabel" | "multilabel"
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml

from intent_classifier.utils.paths import compute_paths, get_config_path, get_repo_root

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


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries (override wins).

    This is used for layered configuration loading:
    base defaults <- dataset config <- experiment overrides.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


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
        >>> config = {"resolved": {"run_id": "singlelabel/clinc150/tiny"}}
        >>> substitute_vars("output/runs/${resolved.run_id}/models", config)
        'output/runs/singlelabel/clinc150/tiny/models'
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

    Converts absolute paths to relative paths starting with ``config/``.
    This is the canonical format expected by load_config() and other config functions.

    Args:
        config_path: Path object (absolute or relative) to a config file

    Returns:
        Config file path string starting with ``config/`` (e.g.,
        ``config/experiments/clinc150/tiny.yaml``). If ``config_path`` is
        absolute and can't be made relative, returns absolute path as string.

    Raises:
        ValueError: If ``config_path`` is relative but doesn't start with ``config/``.
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
    1. ``CONFIG_FILE`` environment variable (highest priority)
    2. ``DATASET`` / ``VARIANT`` environment variables
       - ``DATASET``: dataset name (e.g., \"clinc150\")
       - ``VARIANT``: config name without extension (e.g., \"tiny\" or \"default\")
    3. Fallback to ``config/experiments/clinc150/tiny.yaml``

    Returns:
        Config file path starting with ``config/`` (e.g.,
        ``config/experiments/clinc150/tiny.yaml``)
    """
    # Check explicit config file first
    config_file = os.environ.get("CONFIG_FILE")
    if config_file:
        return config_file

    # Next: DATASET/VARIANT pair (thin discovery API).
    # New implementation: **only** layered experiment configs are supported here.
    dataset_env = os.environ.get("DATASET")
    variant_env = os.environ.get("VARIANT")
    if dataset_env:
        variant = variant_env or "tiny"

        repo_root = get_repo_root()
        exp_path = repo_root / "config" / "experiments" / dataset_env / f"{variant}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(
                "Requested experiment config not found.\n"
                f"Expected: config/experiments/{dataset_env}/{variant}.yaml\n"
            )
        return f"config/experiments/{dataset_env}/{variant}.yaml"

    # Final fallback
    return "config/experiments/clinc150/tiny.yaml"


def parse_config_path(config_file: str) -> tuple[str, str]:
    """Parse config file path to extract dataset name and config name.

    Args:
        config_file: Config file path. Must be:
                    - ``config/experiments/{dataset_name}/{config_name}.yaml``
                      (relative to repo root), or an absolute path ending with the same
                      structure.

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

        # Required format: config/experiments/{dataset_name}/{config_name}.yaml
        if len(parts) >= 4 and parts[0] == "config" and parts[1] == "experiments":
            dataset_name = parts[2]
            config_name = config_path.stem
            return dataset_name, config_name

        # If we get here, the path doesn't match expected structure
        raise ValueError(
            "Config file must be in structure: "
            "config/experiments/{dataset_name}/{config_name}.yaml\n"
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
            "Config file must be in structure: "
            "config/experiments/{dataset_name}/{config_name}.yaml\n"
            f"Got: {rel_path}"
        ) from e


def _load_llm_config() -> dict[str, Any] | None:
    """Load centralized provider configuration.

    The implementation treats ``config/base/providers.yaml`` as the
    single source of truth. The historical ``config/llm_config.yaml`` file
    is no longer consulted.

    Returns:
        LLM/provider configuration dictionary, or None if nothing is configured.
    """
    repo_root = get_repo_root()

    # Provider configuration (single source of truth)
    providers_path = repo_root / "config" / "base" / "providers.yaml"
    if providers_path.exists():
        try:
            with open(providers_path, encoding="utf-8") as f:
                providers_cfg = yaml.safe_load(f) or {}
            if not isinstance(providers_cfg, dict):
                raise ValueError("providers.yaml must contain a YAML dictionary")

            providers_block = providers_cfg.get("providers", {})
            if not isinstance(providers_block, dict):
                raise ValueError("providers.yaml must contain a 'providers' mapping")

            llm_config: dict[str, Any] = {}

            ollama_cfg = providers_block.get("ollama", {}) or {}
            if isinstance(ollama_cfg, dict):
                llm_config["ollama"] = {
                    "endpoint": ollama_cfg.get("endpoint"),
                    "default_model": ollama_cfg.get("llm_default"),
                    "embedding_model": ollama_cfg.get("embed_default"),
                }

            openai_cfg = providers_block.get("openai", {}) or {}
            if isinstance(openai_cfg, dict):
                llm_config["openai"] = {
                    "default_model": openai_cfg.get("llm_default"),
                    "embedding_model": openai_cfg.get("embed_default"),
                }

            logger.debug("Loaded provider config from: %s", providers_path)
            return llm_config
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug("Could not load providers.yaml: %s", exc)

    return None


def _ensure_general_and_paths(
    config: dict[str, Any],
    dataset_name: str,
    config_name: str,
    label_type: Literal["singlelabel", "multilabel"],
) -> None:
    """Ensure general metadata and paths are present in the config.

    This centralises run_name/run_id and path computation so that:
    - YAML configs can omit the ``paths`` section entirely
    - ``general.run_name`` is always derived from the config file name
    - ``general.run_id`` is a stable identifier including label type + dataset
    """
    general = config.setdefault("general", {})

    # Always derive run_name from the config file name.
    # This was previously only overridden if present; making it unconditional
    # keeps behaviour consistent and avoids "half-templated" configs.
    general["run_name"] = config_name

    # Stable identifier that can be used for directory layout or logging.
    # Example: "singlelabel/clinc150/tiny"
    run_id = f"{label_type}/{dataset_name}/{config_name}"
    general["run_id"] = run_id

    # Compute canonical paths for this run_id using the output schema.
    # Paths are always derived in code from the run_id.
    computed = compute_paths(run_id, root="output/runs")
    config["paths"] = {
        "run_dir": computed["run_dir"],
        "meta_dir": computed["meta_dir"],
        "dataset_dir": computed["dataset_dir"],
        "features_dir": computed["features_dir"],
        "models_dir": computed["models_dir"],
        "eval_dir": computed["eval_dir"],
        "compare_dir": computed["compare_dir"],
        "llm_logs_dir": computed["llm_logs_dir"],
        "figures_dir": computed["figures_dir"],
    }


def _attach_providers_and_resolved(config: dict[str, Any]) -> None:
    """Attach provider metadata and resolved model choices to the config.

    This gives downstream components a single, normalized place to read:
    - ``providers.*``: raw provider settings (endpoints, default models)
    - ``resolved.*``: effective embedding / LLM models based on backends
    """
    llm_config = _load_llm_config() or {}

    providers: dict[str, Any] = {}

    # Ollama provider
    ollama_cfg = llm_config.get("ollama", {}) if isinstance(llm_config, dict) else {}
    if ollama_cfg:
        # Environment variables override static config, matching the rules documented
        # in config/llm_config.yaml.
        endpoint_env = (
            os.getenv("OLLAMA_API_BASE")
            or os.getenv("OLLAMA_HOST")
            or os.getenv("MODEL_OLLAMA_ENDPOINT")
        )
        endpoint = endpoint_env or ollama_cfg.get("endpoint") or "http://localhost:11434"
        providers["ollama"] = {
            "endpoint": endpoint,
            "llm_default": ollama_cfg.get("default_model"),
            "embed_default": ollama_cfg.get("embedding_model"),
        }

    # OpenAI provider
    openai_cfg = llm_config.get("openai", {}) if isinstance(llm_config, dict) else {}
    if openai_cfg:
        providers["openai"] = {
            "llm_default": openai_cfg.get("default_model"),
            "embed_default": openai_cfg.get("embedding_model"),
        }

    if providers:
        # Do not overwrite if user already provided a providers block.
        config.setdefault("providers", providers)

    # ------------------------------------------------------------------
    # Build resolved.* section
    # ------------------------------------------------------------------
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}

    # Embedding backend and effective model
    embedding_backend = model_cfg.get("embedding_backend", "sbert")
    embedding_model: str | None

    if embedding_backend == "openai":
        embedding_model = providers.get("openai", {}).get("embed_default")
    elif embedding_backend == "ollama":
        embedding_model = providers.get("ollama", {}).get("embed_default")
    else:  # sbert / local
        embedding_model = model_cfg.get("sbert_model_name")

    # LLM backend and effective model
    #
    # Provider model IDs live in config/base/providers.yaml.
    # Downstream code should rely on resolved.llm_model and resolved.embedding_model.
    llm_backend = model_cfg.get("llm_backend")
    llm_model: str | None = None

    if llm_backend == "openai":
        llm_model = providers.get("openai", {}).get("llm_default")
    elif llm_backend == "ollama":
        llm_model = providers.get("ollama", {}).get("llm_default")

    # Effective Ollama endpoint
    ollama_endpoint = providers.get("ollama", {}).get("endpoint")

    resolved: dict[str, Any] = {
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "ollama_endpoint": ollama_endpoint,
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
    }

    # Do not overwrite an explicit resolved block; just fill in missing keys.
    existing_resolved = config.setdefault("resolved", {})
    if isinstance(existing_resolved, dict):
        for key, value in resolved.items():
            existing_resolved.setdefault(key, value)


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
        config_file: Config file path starting with ``config/`` (e.g.,
                    ``config/experiments/clinc150/tiny.yaml``). If None, uses
                    discover_config_file() to find a default.
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
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Layered loading for experiment configs:
    #   config/base/defaults.yaml
    #   + config/base/providers.yaml
    #   + config/datasets/{dataset}.yaml
    #   + config/experiments/{dataset}/{variant}.yaml   (the requested config_file)
    repo_root = get_repo_root()
    rel_parts = config_path.relative_to(repo_root).parts
    config: dict[str, Any]

    if len(rel_parts) >= 3 and rel_parts[0] == "config" and rel_parts[1] == "experiments":
        dataset_name = rel_parts[2]
        base_config: dict[str, Any] = {}

        # 1) Base defaults (optional)
        base_defaults_path = repo_root / "config" / "base" / "defaults.yaml"
        if base_defaults_path.exists():
            logger.debug("Loading base defaults from: %s", base_defaults_path)
            with open(base_defaults_path, encoding="utf-8") as f:
                defaults_cfg = yaml.safe_load(f) or {}
            if not isinstance(defaults_cfg, dict):
                raise ValueError(
                    f"Base defaults config must contain a YAML dictionary, got {type(defaults_cfg)}"
                )
            base_config = _deep_merge_dicts(base_config, defaults_cfg)

        # 2) Provider defaults (optional)
        providers_path = repo_root / "config" / "base" / "providers.yaml"
        if providers_path.exists():
            logger.debug("Loading provider defaults from: %s", providers_path)
            with open(providers_path, encoding="utf-8") as f:
                providers_cfg = yaml.safe_load(f) or {}
            if not isinstance(providers_cfg, dict):
                raise ValueError(
                    f"Provider config must contain a YAML dictionary, got {type(providers_cfg)}"
                )
            base_config = _deep_merge_dicts(base_config, providers_cfg)

        # 3) Dataset-level config (optional)
        dataset_cfg_path = repo_root / "config" / "datasets" / f"{dataset_name}.yaml"
        if dataset_cfg_path.exists():
            logger.debug("Loading dataset config from: %s", dataset_cfg_path)
            with open(dataset_cfg_path, encoding="utf-8") as f:
                dataset_cfg = yaml.safe_load(f) or {}
            if not isinstance(dataset_cfg, dict):
                raise ValueError(
                    f"Dataset config must contain a YAML dictionary, got {type(dataset_cfg)}"
                )
            base_config = _deep_merge_dicts(base_config, dataset_cfg)

        # 4) Experiment-level overrides (required for this branch)
        logger.debug("Loading experiment config from: %s", config_path)
        with open(config_path, encoding="utf-8") as f:
            experiment_cfg = yaml.safe_load(f) or {}
        if not isinstance(experiment_cfg, dict):
            raise ValueError(
                f"Experiment config must contain a YAML dictionary, got {type(experiment_cfg)}"
            )

        config = _deep_merge_dicts(base_config, experiment_cfg)
    else:
        raise ValueError(
            "Config file must be in structure: "
            "config/experiments/{dataset_name}/{config_name}.yaml\n"
            f"Got: {config_path}"
        )

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

    return config


def load_config_with_metadata(
    config_file: str | None = None,
    apply_variable_substitution: bool = True,
) -> ConfigMetadata:
    """Load config and return it with metadata (dataset_name, config_name, label_type).

    Args:
        config_file: Config file path starting with ``config/`` (e.g.,
                    ``config/experiments/clinc150/tiny.yaml``). If None, uses
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

    # Detect label type early; this is cheap and allows us to build a stable run_id.
    label_type = detect_label_type(config, dataset_name)

    # Ensure general metadata (run_name / run_id) and default paths are present
    _ensure_general_and_paths(
        config=config,
        dataset_name=dataset_name,
        config_name=config_name,
        label_type=label_type,
    )

    # Attach provider metadata and resolved model selections for downstream use
    _attach_providers_and_resolved(config)

    # Enrich resolved.* with run_id / label_type / paths snapshot
    general = config.get("general", {})
    paths = config.get("paths", {})
    resolved = config.setdefault("resolved", {})
    if isinstance(resolved, dict):
        resolved.setdefault("run_id", general.get("run_id"))
        resolved.setdefault("label_type", label_type)
        if "paths" not in resolved and isinstance(paths, dict):
            # Store a shallow snapshot to keep effective paths discoverable
            resolved["paths"] = dict(paths)

    # Now apply variable substitution with the updated structure
    if apply_variable_substitution:
        config = process_config_vars(config, config)

    return {
        "config": config,
        "dataset_name": dataset_name,
        "config_name": config_name,
        "label_type": label_type,
        "config_path": config_path,
        "config_file": config_file,
    }


def get_first_available_dataset() -> str | None:
    """Get the name of the first available dataset from config/experiments directory.

    Returns:
        Name of the first dataset directory found, or None if no datasets exist.
    """
    try:
        repo_root = get_repo_root()
        experiments_dir = repo_root / "config" / "experiments"
        if experiments_dir.exists():
            dataset_dirs = [d.name for d in experiments_dir.iterdir() if d.is_dir()]
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
