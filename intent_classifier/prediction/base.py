"""Base prediction runner with shared functionality for single-label and multi-label."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cloudpickle
import numpy as np
from sklearn.base import BaseEstimator

from intent_classifier.model import TextClassifier
from intent_classifier.utils.file_ops import ensure_dir


# Protocol for anything with predict() method
@runtime_checkable
class _Predictor(Protocol):
    """Minimal protocol for anything with predict() and optional _is_multilabel flag."""

    def predict(self, X: Any) -> Any: ...

    # Optional attribute; guarded at runtime where used
    # mypy will accept attribute checks via hasattr(...)
    # _is_multilabel: bool


# Type alias for any classifier/estimator
type EstimatorType = TextClassifier | BaseEstimator | _Predictor

# Suppress Pydantic serialization warnings - these are not serious, just verbose
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Expected `Usage`.*",
    category=UserWarning,
)


class BasePredictionRunner(ABC):
    """Base class for prediction runners with shared functionality.

    This class provides common functionality for both single-label and multi-label
    prediction workflows, including model loading, timing, and directory management.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the prediction runner.

        Args:
            verbose: Whether to print progress and warning messages
        """
        self.verbose = verbose

    @abstractmethod
    def ensure_valid_predictions(
        self,
        predictions: Sequence[Any],
        estimator: EstimatorType,
        X: Sequence[str],
    ) -> Sequence[Any]:
        """Ensure predictions are in valid format for this label type.

        Ensures all predictions have at least one label (never empty/None/NaN).
        If a prediction is empty, uses probabilities to select the top-1 class.

        Args:
            predictions: Raw predictions from model
            estimator: The model estimator (for getting probabilities if needed)
            X: Input texts (for getting probabilities if needed)

        Returns:
            Validated predictions in appropriate format (always at least one label)
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def normalize_label_for_saving(self, label: Any) -> str:
        """Normalize a label to a string for saving to CSV.

        Args:
            label: Label value (format depends on label type)

        Returns:
            String representation suitable for CSV
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def persist_predictions(
        self,
        estimator: EstimatorType,
        X: Sequence[str],
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        output_dir: Path,
        prefix: str,
        model_name: str,
    ) -> None:
        """Persist predictions and probabilities for a dataset.

        Args:
            estimator: Fitted scikit-learn estimator
            X: Input text data
            y_true: Ground truth labels
            y_pred: Predicted labels
            output_dir: Directory path to save predictions
            prefix: Prefix for output files (e.g., 'train' or 'test')
            model_name: Name of the model being used
        """
        pass  # pylint: disable=unnecessary-pass

    def load_model(self, model_or_path: str | Path | EstimatorType) -> EstimatorType:
        """Load a model from a path or return the model object if already loaded.

        Args:
            model_or_path: Either a path to a serialized model or an already loaded model

        Returns:
            Loaded model object (TextClassifier, BaseEstimator, or compatible estimator)
        """
        if isinstance(model_or_path, (str, Path)):
            if self.verbose:
                print(f"Loading model from: {model_or_path}", flush=True)
            with open(model_or_path, "rb") as f:
                estimator = cloudpickle.load(f)
            if self.verbose:
                print("Model loaded successfully", flush=True)
            return estimator
        else:
            # Assume it's already a model object
            if self.verbose:
                print("Using provided model object", flush=True)
            return model_or_path

    def detect_model_type(
        self, estimator: EstimatorType, X_sample: Sequence[str] | None = None
    ) -> bool:
        """Detect if model is multi-label.

        Args:
            estimator: Model object (TextClassifier, BaseEstimator, or compatible)
            X_sample: Optional sample data for detection

        Returns:
            True if multi-label, False if single-label
        """
        from intent_classifier.utils.label_utils import is_multilabel as check_is_multilabel

        # Check attribute first (fastest)
        # Guard optional attribute; keeps mypy happy and avoids AttributeError.
        if hasattr(estimator, "_is_multilabel"):
            return bool(getattr(estimator, "_is_multilabel", False))

        # Try to detect from a sample prediction
        if hasattr(estimator, "predict") and X_sample is not None and len(X_sample) > 0:
            try:
                sample_pred = estimator.predict(X_sample[:1])
                return check_is_multilabel(sample_pred)
            except Exception:
                # If detection fails, assume single-label (safer default)
                return False

        # Default to single-label if we can't determine
        return False

    def run_prediction(
        self,
        models_or_paths: dict[str, str | Path | EstimatorType],
        *,
        X_train: Sequence[str] | None = None,
        y_train: Sequence[Any] | None = None,
        X_test: Sequence[str] | None = None,
        y_test: Sequence[Any] | None = None,
        output_dir: str | Path = "prediction_artefacts",
        save_train_predictions: bool = True,
        save_test_predictions: bool = True,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Load models and generate predictions for training and test data.

        This is the main prediction pipeline that handles model loading, prediction
        generation, and artifact persistence. It provides a unified interface for
        generating predictions from multiple models and saving their outputs.

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

        Returns:
            dict[str, dict[str, np.ndarray]]: Dictionary mapping model names to their
                predictions for both train and test sets

        Raises:
            ValueError: If train or test data is inconsistently provided
            FileNotFoundError: If model file doesn't exist
            Exception: Any exceptions raised during prediction are propagated
        """
        # Validate data consistency
        if (X_train is None) != (y_train is None):
            raise ValueError("X_train and y_train must both be provided or both be None")
        if (X_test is None) != (y_test is None):
            raise ValueError("X_test and y_test must both be provided or both be None")

        if X_train is None and X_test is None:
            raise ValueError("At least one of X_train or X_test must be provided")

        output_dir = ensure_dir(output_dir)
        predictions: dict[str, dict[str, np.ndarray]] = {}
        prediction_times: dict[str, float] = {}
        total_models = len(models_or_paths)

        for idx, (name, model_or_path) in enumerate(models_or_paths.items(), 1):
            if self.verbose:
                print("\n" + "=" * 60)
                print(f"[{idx}/{total_models}] Running predictions with algorithm: {name}")
                print("=" * 60)

            try:
                # Load model
                estimator = self.load_model(model_or_path)

                # Generate predictions and time the process
                start = time.perf_counter()

                # Initialize predictions dictionary for this model
                predictions[name] = {}
                y_pred_train = None
                y_pred_test = None

                # Generate and store train and test predictions
                if X_train is not None:
                    if self.verbose:
                        print(f"Predicting on {len(X_train)} training samples...", flush=True)
                    y_pred_train = estimator.predict(X_train)
                    y_pred_train = self.ensure_valid_predictions(
                        y_pred_train, estimator=estimator, X=X_train
                    )
                    predictions[name]["train"] = y_pred_train
                    if self.verbose:
                        print("✓ Training predictions completed", flush=True)

                if X_test is not None:
                    if self.verbose:
                        print(f"Predicting on {len(X_test)} test samples...", flush=True)
                    y_pred_test = estimator.predict(X_test)
                    y_pred_test = self.ensure_valid_predictions(
                        y_pred_test, estimator=estimator, X=X_test
                    )
                    predictions[name]["test"] = y_pred_test
                    if self.verbose:
                        print("✓ Test predictions completed", flush=True)

                end = time.perf_counter()
                execution_time = end - start
                prediction_times[name] = execution_time

                if self.verbose:
                    print(f"\n✅ Algorithm '{name}' predictions completed successfully")
                    print(
                        f"   Time taken: {execution_time:.2f} seconds "
                        f"({execution_time / 60:.2f} minutes)"
                    )
                    if idx < total_models:
                        print(f"   Progress: {idx}/{total_models} algorithms completed\n")

                model_dir = ensure_dir(output_dir / name)

                # Save individual execution time
                with open(model_dir / f"{name}_prediction_time.txt", "w", encoding="utf-8") as f:
                    f.write(f"Prediction time: {execution_time:.2f} seconds")

                # Persist predictions
                datasets = [
                    (
                        "train",
                        X_train,
                        y_train,
                        y_pred_train,
                        save_train_predictions and X_train is not None,
                    ),
                    (
                        "test",
                        X_test,
                        y_test,
                        y_pred_test,
                        save_test_predictions and X_test is not None,
                    ),
                ]

                for prefix, X, y, y_pred, should_save in datasets:
                    if should_save and X is not None and y is not None and y_pred is not None:
                        self.persist_predictions(
                            estimator=estimator,
                            X=X,
                            y_true=y,
                            y_pred=y_pred,
                            output_dir=model_dir,
                            prefix=prefix,
                            model_name=name,
                        )

            except Exception as e:
                if self.verbose:
                    print(f"\n❌ Error running predictions for algorithm '{name}': {e}", flush=True)
                    print(
                        "   Skipping this model and continuing with remaining models...", flush=True
                    )
                    print(
                        f"   Progress: {idx - 1}/{total_models} algorithms completed before error\n"
                    )
                # Log the error but continue with other models
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to generate predictions for model '{name}': {e}", exc_info=True
                )
                # Continue to next model instead of raising
                continue

        # Save all prediction times to a single file
        with open(output_dir / "prediction_times.txt", "w", encoding="utf-8") as f:
            for model_name, time_taken in prediction_times.items():
                f.write(f"{model_name}: {time_taken:.2f} seconds\n")

        if self.verbose:
            print("\n" + "=" * 60)
            print("Prediction Summary")
            print("=" * 60)
            print(f"Total algorithms processed: {len(predictions)}/{total_models}")
            total_time = sum(prediction_times.values())
            print(
                f"Total prediction time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)"
            )
            print("\nPer-algorithm prediction times:")
            for name, time_taken in sorted(
                prediction_times.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {name}: {time_taken:.2f}s ({time_taken / 60:.2f}min)")
            print(f"\nResults saved to: {output_dir}")
            print("=" * 60)

        return predictions
