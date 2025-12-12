"""
Model Registry

This module provides a decorator-based registration system for model classes.
Models can register themselves using the @register_model decorator, making
the system extensible without requiring changes to model_loader.py.

Example:
    >>> from intent_classifier.utils.model_registry import register_model
    >>>
    >>> @register_model("MyModel")
    >>> class MyModel:
    ...     pass
    >>>
    >>> # Later, retrieve the model class
    >>> from intent_classifier.utils.model_registry import get_model_class
    >>> ModelClass = get_model_class("MyModel")
"""

from typing import Any, TypeVar

from intent_classifier.model import TextClassifier

# Type variable for model classes
T = TypeVar("T", bound=TextClassifier)

# Registry mapping model class names to their actual classes
_MODEL_REGISTRY: dict[str, type[TextClassifier]] = {}


def register_model(class_name: str) -> Any:
    """Decorator to register a model class in the registry.

    Args:
        class_name: The name to register the model under (typically the class name)

    Returns:
        The decorated class (unchanged)

    Example:
        >>> @register_model("NaiveBayesClassifier")
        >>> class NaiveBayesClassifier(TextClassifier):
        ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        """Register the class in the model registry.

        Args:
            cls: The model class to register (must be a subclass of TextClassifier)

        Returns:
            The same class (unchanged)
        """
        if class_name in _MODEL_REGISTRY:
            import warnings

            warnings.warn(
                f"Model class '{class_name}' is already registered. "
                f"Overwriting previous registration.",
                UserWarning,
                stacklevel=2,
            )
        _MODEL_REGISTRY[class_name] = cls
        return cls

    return decorator


def get_model_class(class_name: str) -> type[TextClassifier] | None:
    """Get a model class from the registry by name.

    Args:
        class_name: Name of the model class to retrieve

    Returns:
        The model class if found, None otherwise. The returned class is guaranteed
        to be a subclass of TextClassifier.
    """
    return _MODEL_REGISTRY.get(class_name)


def get_all_registered_models() -> dict[str, type[TextClassifier]]:
    """Get all registered model classes.

    Returns:
        Dictionary mapping class names to their classes. All classes are
        guaranteed to be subclasses of TextClassifier.
    """
    return _MODEL_REGISTRY.copy()


def is_registered(class_name: str) -> bool:
    """Check if a model class is registered.

    Args:
        class_name: Name of the model class to check

    Returns:
        True if the class is registered, False otherwise
    """
    return class_name in _MODEL_REGISTRY


def discover_and_register_models() -> None:
    """Auto-discover and register models from the algorithms package.

    This function imports all modules in the algorithms package to trigger
    their registration decorators. This allows models to be registered
    automatically without explicit imports.
    """
    import importlib
    from pathlib import Path

    # Get the algorithms package path
    algorithms_package = Path(__file__).parent.parent / "algorithms"

    if not algorithms_package.exists():
        return

    # Import all Python modules in the algorithms package
    for module_file in algorithms_package.glob("*.py"):
        if module_file.name == "__init__.py":
            continue

        module_name = module_file.stem
        try:
            importlib.import_module(f"intent_classifier.algorithms.{module_name}")
        except ImportError as e:
            import warnings

            warnings.warn(
                f"Failed to import algorithm module '{module_name}': {e}",
                ImportWarning,
                stacklevel=2,
            )
