"""
Dataset Module.

This module provides functions for loading the CLINC150 intent classification dataset.

Functions:
    get_dataset: Common entry point for dataset loading

Created: 2025-05-03
"""

from __future__ import annotations

import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataset(
    dataset_name: str = "clinc150", use_oos: bool = False, **kwargs
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Main entry point for loading CLINC150 dataset.

    Args:
        dataset_name: Name of the dataset to load (default: "clinc150", only supported value)
        use_oos: If True, include OOS (out-of-scope) examples as an extra class label
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Tuple containing:
        - X_train: List of training document texts
        - y_train: List of training document labels
        - X_test: List of test document texts
        - y_test: List of test document labels
        - classes: List of class labels
    """
    if dataset_name != "clinc150":
        logger.warning(f"Unknown dataset_name: {dataset_name}. Using 'clinc150' as default.")
        dataset_name = "clinc150"

    from .clinc150 import load_clinc150

    logger.info("Loading CLINC150 dataset with configuration:")
    logger.info(f"  - Use OOS: {use_oos}")

    X_train, y_train, X_test, y_test, classes = load_clinc150(use_oos=use_oos)

    logger.info("Dataset loaded:")
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Test samples: {len(X_test)}")
    logger.info(f"  - Classes: {len(classes)}")

    return X_train, y_train, X_test, y_test, classes
