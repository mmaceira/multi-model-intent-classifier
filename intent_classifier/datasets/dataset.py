"""
Dataset Module.

This module provides functions for loading intent classification datasets.

The system automatically discovers datasets from `config/datasets/{name}.yaml`
that define a `loader` section.
To add a new dataset, simply create `config/datasets/{name}.yaml` with a `loader`
configuration - no Python code needed!

Functions:
    get_dataset: Common entry point for dataset loading
    register_dataset_loader: Register a custom dataset loader (for advanced use cases)
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)

# These imports come after logger setup for consistency with module structure
from intent_classifier.utils.paths import get_repo_root  # noqa: E402

# Import generic loader
from .generic_loader import load_dataset_from_config  # noqa: E402

# Map dataset names to loader functions (for custom loaders)
_CUSTOM_LOADERS: dict[str, Callable] = {}


def _discover_datasets() -> dict[str, Callable]:
    """Discover datasets from `config/datasets/*.yaml` files.

    Any dataset config file that defines a top-level `loader` section
    will be exposed via a generic loader.

    Returns:
        Dictionary mapping dataset names to loader functions
    """
    repo_root = get_repo_root()
    config_dir = repo_root / "config" / "datasets"

    if not config_dir.exists():
        return {}

    discovered: dict[str, Callable] = {}

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        dataset_name = yaml_path.stem

        # Always register a loader; generic_loader will validate the presence
        # of a usable configuration at runtime (and raise a clear error if missing).
        def make_loader(name: str):
            def loader(
                use_oos: bool = False,
                max_train_samples: int | None = None,
                max_test_samples: int | None = None,
                max_val_samples: int | None = None,
                max_classes: int | None = None,
                seed: int = 42,
                csv_path: str | Path | None = None,
                multilabel: bool | None = None,
                **kwargs,
            ):
                return load_dataset_from_config(
                    name,
                    use_oos=use_oos,
                    max_train_samples=max_train_samples,
                    max_test_samples=max_test_samples,
                    max_val_samples=max_val_samples,
                    max_classes=max_classes,
                    seed=seed,
                    csv_path=csv_path,
                    multilabel=multilabel,
                    **kwargs,
                )

            return loader

        discovered[dataset_name] = make_loader(dataset_name)

    return discovered


# Cache discovered datasets
_DISCOVERED_DATASETS: dict[str, Callable] | None = None


def _get_all_loaders() -> dict[str, Callable]:
    """Get all available dataset loaders (discovered + custom)."""
    global _DISCOVERED_DATASETS
    if _DISCOVERED_DATASETS is None:
        _DISCOVERED_DATASETS = _discover_datasets()

    # Merge discovered and custom loaders (custom take precedence)
    return {**_DISCOVERED_DATASETS, **_CUSTOM_LOADERS}


def register_dataset_loader(
    dataset_name: str,
    loader_function: Callable,
):
    """Register a custom dataset loader function.

    This is for advanced use cases where you need a custom Python loader.
    For most cases, just create `config/datasets/{name}.yaml` with a `loader`
    section instead.

    Args:
        dataset_name: Name of the dataset
        loader_function: Function that loads the dataset
    """
    _CUSTOM_LOADERS[dataset_name] = loader_function


def get_dataset(
    dataset_name: str,
    use_oos: bool = False,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    max_val_samples: int | None = None,
    max_classes: int | None = None,
    seed: int = 42,
    csv_path: str | Path | None = None,
    multilabel: bool = False,
    **kwargs,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    Main entry point for loading intent classification datasets.

    Args:
        dataset_name: Name of the dataset to load (must be registered)
        use_oos: If True, include OOS (out-of-scope) examples as an extra class label
        max_train_samples: Maximum number of training samples to load (None = all)
        max_test_samples: Maximum number of test samples to load (None = all)
        max_val_samples: Maximum number of validation samples to load (None = all)
        max_classes: Maximum number of classes to include (None = all classes)
        seed: Random seed for reproducibility
        csv_path: Path to CSV file (if required by dataset)
        multilabel: Whether dataset supports multi-label format
        **kwargs: Additional arguments passed to dataset-specific loaders

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
            X_train, y_train, X_val, y_val, X_test, y_test, classes
            Train, validation, and test sets are kept separate.
    """
    # Get all available loaders (discovered from config + custom)
    all_loaders = _get_all_loaders()

    if dataset_name not in all_loaders:
        available = ", ".join(sorted(all_loaders.keys()))
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. "
            f"Available datasets: {available}. "
            f"To add a new dataset, create config/datasets/{{dataset_name}}.yaml "
            f"with a `loader` section."
        )

    loader_function = all_loaders[dataset_name]

    # Get function signature to know which parameters it accepts
    sig = inspect.signature(loader_function)
    accepted_params = set(sig.parameters.keys())

    # Start with provided kwargs
    loader_kwargs = {**kwargs}

    # Add common parameters if accepted by the loader
    if "max_train_samples" in accepted_params:
        loader_kwargs["max_train_samples"] = max_train_samples
    if "max_test_samples" in accepted_params:
        loader_kwargs["max_test_samples"] = max_test_samples
    if "max_val_samples" in accepted_params:
        loader_kwargs["max_val_samples"] = max_val_samples
    if "max_classes" in accepted_params:
        loader_kwargs["max_classes"] = max_classes
    if "seed" in accepted_params:
        loader_kwargs["seed"] = seed

    # Add dataset-specific parameters if accepted by the loader
    if "use_oos" in accepted_params:
        loader_kwargs["use_oos"] = use_oos
    if "csv_path" in accepted_params:
        if csv_path is not None:
            loader_kwargs["csv_path"] = csv_path
    if "multilabel" in accepted_params:
        loader_kwargs["multilabel"] = multilabel

    logger.info(f"Loading {dataset_name} dataset with configuration:")
    if use_oos and "use_oos" in loader_kwargs:
        logger.info(f"  - Use OOS: {use_oos}")
    if csv_path and "csv_path" in loader_kwargs:
        logger.info(f"  - CSV path: {csv_path}")
    if multilabel and "multilabel" in loader_kwargs:
        logger.info(f"  - Multi-label: {multilabel}")
    if max_classes is not None:
        logger.info(f"  - Max classes: {max_classes}")
    if max_train_samples is not None:
        logger.info(f"  - Max training samples: {max_train_samples}")
    if max_val_samples is not None:
        logger.info(f"  - Max validation samples: {max_val_samples}")
    if max_test_samples is not None:
        logger.info(f"  - Max test samples: {max_test_samples}")
    for key, value in kwargs.items():
        if value is not None:
            logger.info(f"  - {key}: {value}")

    X_train, y_train, X_val, y_val, X_test, y_test, classes = loader_function(**loader_kwargs)

    logger.info("Dataset loaded:")
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Validation samples: {len(X_val)}")
    logger.info(f"  - Test samples: {len(X_test)}")
    logger.info(f"  - Classes: {len(classes)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, classes
