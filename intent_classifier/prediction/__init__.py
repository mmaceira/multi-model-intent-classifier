"""Prediction module for single-label and multi-label classification.

This module provides separate implementations for single-label and multi-label
prediction workflows, with a common base class for shared functionality.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from intent_classifier.prediction.base import BasePredictionRunner
from intent_classifier.prediction.multilabel import MultiLabelPredictionRunner
from intent_classifier.prediction.singlelabel import SingleLabelPredictionRunner
from intent_classifier.utils.label_utils import is_multilabel


def run_prediction(
    models_or_paths: dict[str, str | Path | Any],
    *,
    X_train: Sequence[str] | None = None,
    y_train: Sequence[Any] | None = None,
    X_test: Sequence[str] | None = None,
    y_test: Sequence[Any] | None = None,
    output_dir: str | Path = "prediction_artefacts",
    save_train_predictions: bool = True,
    save_test_predictions: bool = True,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Load models and generate predictions for training and test data.

    This is the main prediction pipeline that handles model loading, prediction
    generation, and artifact persistence. It automatically detects whether the
    task is single-label or multi-label and uses the appropriate implementation.

    Args:
        models_or_paths: Dictionary mapping model names to either:
                        - paths of serialized models (str or Path)
                        - already loaded model objects (sklearn estimators)
        X_train: Training text data for prediction
        y_train: Training labels for evaluation
        X_test: Test text data for prediction
        y_test: Test labels for evaluation
        output_dir: Root directory for saving all prediction artifacts
        save_train_predictions: Whether to save training set predictions
        save_test_predictions: Whether to save test set predictions
        verbose: Whether to print progress and warning messages

    Returns:
        dict[str, dict[str, np.ndarray]]: Dictionary mapping model names to their
            predictions for both train and test sets

    Raises:
        ValueError: If train or test data is inconsistently provided
        FileNotFoundError: If model file doesn't exist
        Exception: Any exceptions raised during prediction are propagated

    Note:
        - Prediction times are tracked and saved for each model
        - Directory structure is automatically created if it doesn't exist
        - All operations are performed sequentially for each model
        - Label type (single-label vs multi-label) is automatically detected
    """
    # Validate data consistency
    if (X_train is None) != (y_train is None):
        raise ValueError("X_train and y_train must both be provided or both be None")
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must both be provided or both be None")

    if X_train is None and X_test is None:
        raise ValueError("At least one of X_train or X_test must be provided")

    # Detect label type from ground truth data (prefer test set if available)
    is_multilabel_task = False
    if y_test is not None and len(y_test) > 0:
        is_multilabel_task = is_multilabel(y_test)
    elif y_train is not None and len(y_train) > 0:
        is_multilabel_task = is_multilabel(y_train)
    else:
        # If no ground truth available, try to detect from first model
        # This is a fallback - ideally we should have ground truth
        if models_or_paths:
            first_model_path = next(iter(models_or_paths.values()))
            # Use SingleLabelPredictionRunner as a concrete instance for detection
            temp_runner = SingleLabelPredictionRunner(verbose=False)
            try:
                if isinstance(first_model_path, (str, Path)):
                    estimator = temp_runner.load_model(first_model_path)
                else:
                    estimator = first_model_path

                # Try to detect from model attribute or sample prediction
                if X_test is not None and len(X_test) > 0:
                    is_multilabel_task = temp_runner.detect_model_type(estimator, X_test[:1])
                elif X_train is not None and len(X_train) > 0:
                    is_multilabel_task = temp_runner.detect_model_type(estimator, X_train[:1])
            except Exception:
                # If detection fails, default to single-label
                is_multilabel_task = False

    # Select appropriate runner based on label type
    runner: BasePredictionRunner
    if is_multilabel_task:
        runner = MultiLabelPredictionRunner(verbose=verbose)
    else:
        runner = SingleLabelPredictionRunner(verbose=verbose)

    # Delegate to the appropriate runner
    return runner.run_prediction(
        models_or_paths=models_or_paths,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        save_train_predictions=save_train_predictions,
        save_test_predictions=save_test_predictions,
    )


__all__ = [
    "run_prediction",
    "BasePredictionRunner",
    "SingleLabelPredictionRunner",
    "MultiLabelPredictionRunner",
]
