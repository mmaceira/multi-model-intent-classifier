"""
Dataset Module.

This module provides functions for loading the CLINC150 intent classification dataset.

Functions:
    get_dataset: Common entry point for dataset loading
"""

from __future__ import annotations

import logging
from typing import List, Tuple

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)


def get_dataset(
    dataset_name: str = "clinc150",
    use_oos: bool = False,
    max_train_samples: int = None,
    max_test_samples: int = None,
    max_val_samples: int = None,
    max_classes: int = None,
    seed: int = 42,
    **kwargs,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Main entry point for loading CLINC150 dataset.

    Args:
        dataset_name: Name of the dataset to load (default: "clinc150", only supported value)
        use_oos: If True, include OOS (out-of-scope) examples as an extra class label
        max_train_samples: Maximum number of training samples to load (None = all)
        max_test_samples: Maximum number of test samples to load (None = all)
        max_val_samples: Maximum number of validation samples to load (None = all)
        max_classes: Maximum number of classes to include (None = all classes)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
            X_train, y_train, X_val, y_val, X_test, y_test, classes
            Train, validation, and test sets are kept separate.
    """
    if dataset_name != "clinc150":
        logger.warning(f"Unknown dataset_name: {dataset_name}. Using 'clinc150' as default.")
        dataset_name = "clinc150"

    from .clinc150 import load_clinc150

    logger.info("Loading CLINC150 dataset with configuration:")
    logger.info(f"  - Use OOS: {use_oos}")
    if max_classes is not None:
        logger.info(f"  - Max classes: {max_classes}")
    if max_train_samples is not None:
        logger.info(f"  - Max training samples: {max_train_samples}")
    if max_val_samples is not None:
        logger.info(f"  - Max validation samples: {max_val_samples}")
    if max_test_samples is not None:
        logger.info(f"  - Max test samples: {max_test_samples}")

    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_clinc150(
        use_oos=use_oos,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        max_val_samples=max_val_samples,
        max_classes=max_classes,
        seed=seed,
    )

    logger.info("Dataset loaded:")
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Validation samples: {len(X_val)}")
    logger.info(f"  - Test samples: {len(X_test)}")
    logger.info(f"  - Classes: {len(classes)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, classes
