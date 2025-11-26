"""Prediction utilities for running trained text classifiers and saving outputs."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import cloudpickle
import numpy as np
import pandas as pd

from intent_classifier.utils.file_ops import ensure_dir

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


def _persist_predictions(
    estimator: Any,
    X: Sequence[str],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    output_dir: Path,
    prefix: str,
    model_name: str,
    verbose: bool = True,
) -> None:
    """Persist predictions and probabilities for a dataset.

    This internal function handles the persistence of model predictions and
    probability scores to disk. It creates CSV files for predictions and
    NumPy arrays for probability scores when available.

    Args:
        estimator: Fitted scikit-learn estimator
        X: Input text data to generate predictions for
        y_true: Ground truth labels for the input data
        y_pred: Predicted labels for the input data (cached)
        output_dir: Directory path to save the predictions
        prefix: Prefix for output files (e.g., 'train' or 'test')
        model_name: Name of the model being used
        verbose: Whether to print warning messages

    Note:
        - Predictions are saved as CSV with 'y_true' and 'y_pred' columns
        - Probabilities are saved as NumPy arrays if the model supports predict_proba()
        - Warnings are printed if probability saving fails
    """
    # Save predictions
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)

    # Save probabilities if available
    if hasattr(estimator, "predict_proba"):
        try:
            y_prob = estimator.predict_proba(X)
            np.save(output_dir / f"{prefix}_prob.npy", y_prob)
        except Exception as e:
            if verbose:
                print(
                    f"[{model_name}] Warning: Could not save {prefix} probabilities: {e}",
                    flush=True,
                )


def run_prediction(
    models_or_paths: Dict[str, Union[str, Path, Any]],
    *,
    X_train: Optional[Sequence[str]] = None,
    y_train: Optional[Sequence[Any]] = None,
    X_test: Optional[Sequence[str]] = None,
    y_test: Optional[Sequence[Any]] = None,
    output_dir: Union[str, Path] = "prediction_artefacts",
    save_train_predictions: bool = True,
    save_test_predictions: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load models and generate predictions for training and test data.

    This is the main prediction pipeline that handles model loading, prediction
    generation, and artifact persistence. It provides a unified interface for
    generating predictions from multiple models and saving their outputs in a structured format.

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
        Dict[str, Dict[str, np.ndarray]]: Dictionary mapping model names to their
            predictions for both train and test sets

    Raises:
        ValueError: If train or test data is inconsistently provided
        FileNotFoundError: If model file doesn't exist
        Exception: Any exceptions raised during prediction are propagated

    Note:
        - Prediction times are tracked and saved for each model
        - Directory structure is automatically created if it doesn't exist
        - All operations are performed sequentially for each model
    """
    # Validate data consistency
    if (X_train is None) != (y_train is None):
        raise ValueError("X_train and y_train must both be provided or both be None")
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must both be provided or both be None")

    if X_train is None and X_test is None:
        raise ValueError("At least one of X_train or X_test must be provided")

    output_dir = ensure_dir(output_dir)
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    prediction_times: Dict[str, float] = {}
    total_models = len(models_or_paths)

    for idx, (name, model_or_path) in enumerate(models_or_paths.items(), 1):
        if verbose:
            print("\n" + "=" * 60)
            print(f"[{idx}/{total_models}] Running predictions with algorithm: {name}")
            print("=" * 60)

        try:
            # Determine if the input is a path to a model or an actual model object
            if isinstance(model_or_path, (str, Path)):
                if verbose:
                    print(f"Loading model from: {model_or_path}", flush=True)
                with open(model_or_path, "rb") as f:
                    estimator = cloudpickle.load(f)
                if verbose:
                    print("Model loaded successfully", flush=True)

            else:
                # Assume it's already a model object
                estimator = model_or_path
                if verbose:
                    print("Using provided model object", flush=True)

            # Generate predictions and time the process
            start = time.perf_counter()

            # Initialize predictions dictionary for this model
            predictions[name] = {}
            y_pred_train = None
            y_pred_test = None

            # Generate and store train and test predictions with progress bars
            if X_train is not None:
                if verbose:
                    print(f"Predicting on {len(X_train)} training samples...", flush=True)
                # Check if model has a predict method that supports progress tracking
                # For RAG models, the progress bar is handled internally
                y_pred_train = estimator.predict(X_train)
                predictions[name]["train"] = y_pred_train
                if verbose:
                    print("✓ Training predictions completed", flush=True)

            if X_test is not None:
                if verbose:
                    print(f"Predicting on {len(X_test)} test samples...", flush=True)
                y_pred_test = estimator.predict(X_test)
                predictions[name]["test"] = y_pred_test
                if verbose:
                    print("✓ Test predictions completed", flush=True)

            end = time.perf_counter()
            execution_time = end - start
            prediction_times[name] = execution_time

            if verbose:
                print(f"\n✅ Algorithm '{name}' predictions completed successfully")
                print(
                    f"   Time taken: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
                )
                if idx < total_models:
                    print(f"   Progress: {idx}/{total_models} algorithms completed\n")

            model_dir = ensure_dir(output_dir / name)

            # Save individual execution time
            with open(model_dir / f"{name}_prediction_time.txt", "w") as f:
                f.write(f"Prediction time: {execution_time:.2f} seconds")

            # --- persist predictions --------------------------------------------
            datasets = [
                (
                    "train",
                    X_train,
                    y_train,
                    y_pred_train,
                    save_train_predictions and X_train is not None,
                ),
                ("test", X_test, y_test, y_pred_test, save_test_predictions and X_test is not None),
            ]

            for prefix, X, y, y_pred, should_save in datasets:
                if should_save:
                    _persist_predictions(
                        estimator=estimator,
                        X=X,
                        y_true=y,
                        y_pred=y_pred,
                        output_dir=model_dir,
                        prefix=prefix,
                        model_name=name,
                        verbose=verbose,
                    )

        except Exception as e:
            if verbose:
                print(f"\n❌ Error running predictions for algorithm '{name}': {e}", flush=True)
                print(f"   Progress: {idx-1}/{total_models} algorithms completed before error\n")
            raise

    # Save all prediction times to a single file
    with open(output_dir / "prediction_times.txt", "w") as f:
        for model_name, time_taken in prediction_times.items():
            f.write(f"{model_name}: {time_taken:.2f} seconds\n")

    if verbose:
        print("\n" + "=" * 60)
        print("Prediction Summary")
        print("=" * 60)
        print(f"Total algorithms processed: {len(predictions)}/{total_models}")
        total_time = sum(prediction_times.values())
        print(f"Total prediction time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("\nPer-algorithm prediction times:")
        for name, time_taken in sorted(prediction_times.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {time_taken:.2f}s ({time_taken/60:.2f}min)")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 60)

    return predictions
