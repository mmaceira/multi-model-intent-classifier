"""Training Module for Text Classification

This module provides a robust training workflow for text classification tasks.
It handles model training, model persistence, and training performance tracking.

Key Features:
    - Multi-model training pipeline
    - Model persistence and training artifact storage
    - Training time tracking and logging
    - Flexible configuration options
    - Error handling and validation
    - Progress monitoring and verbose logging

Dependencies:
    - pathlib: Path manipulation
    - typing: Type hints
    - cloudpickle: Model serialization
    - numpy: Numerical operations
    - pandas: Data handling
    - sklearn.base: Base estimator functionality

Example:
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Define models to train
    >>> models = {
    ...     'svm': SVC(probability=True),
    ...     'rf': RandomForestClassifier()
    ... }
    >>>
    >>> # Run training pipeline
    >>> fitted_models = run_training(
    ...     models=models,
    ...     X_train=train_texts,
    ...     y_train=train_labels,
    ...     output_dir='experiments/run_001'
    ... )

Output Structure:
    output_dir/
    ├── model_name/
    │   ├── model.pkl                 # Serialized model
    │   ├── train_predictions.csv     # Training predictions
    │   ├── train_prob.npy           # Training probabilities
    │   └── model_name_execution_time.txt  # Training time
    └── training_times.txt           # Summary of all training times

Notes:
    - All models must implement scikit-learn's estimator interface
    - Models with predict_proba() support will have probabilities saved
    - Training times are tracked and persisted for each model
    - Directory structure is automatically created if not exists

Version: 1.0.0
Author: CLINC150 RAG Classifier Team
License: MIT
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import cloudpickle
from sklearn.base import clone as safe_clone

from src.utils.file_ops import ensure_dir


def run_training(
    models: Dict[str, Any],
    *,
    X_train: Sequence[str],
    y_train: Sequence[Any],
    X_val: Optional[Sequence[str]] = None,
    y_val: Optional[Sequence[Any]] = None,
    output_dir: Union[str, Path] = "artefacts",
    save_models: bool = True,
    save_train_predictions: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit models and persist training artifacts.

    This is the main training pipeline that handles model fitting and artifact persistence.
    It provides a unified interface for training multiple models and saving their outputs
    in a structured format.

    Args:
        models: Dictionary mapping model names to unfitted scikit-learn estimators
        X_train: Training text data
        y_train: Training labels
        X_val: Optional validation text data. Used by models that support early stopping
               or validation-based model selection. Models using cross-validation internally
               (e.g., GridSearchCV) will ignore this parameter.
        y_val: Optional validation labels. Must be provided if X_val is provided.
        output_dir: Root directory for saving all artifacts
        save_models: Whether to persist fitted models to disk
        save_train_predictions: Whether to save training set predictions
        verbose: Whether to print progress and warning messages

    Returns:
        Dict[str, Any]: Dictionary mapping model names to their fitted estimators

    Raises:
        ValueError: If X_val is provided but y_val is not (or vice versa)
        Exception: Any exceptions raised during model training are propagated

    Note:
        - Training times are tracked and saved for each model
        - Models are cloned before fitting to prevent modification of input objects
        - Directory structure is automatically created if it doesn't exist
        - All operations are performed sequentially for each model
        - Most sklearn models use cross-validation internally and don't require a separate
          validation set. The validation set is provided for models that support early stopping
          or for future extensibility.
    """
    # Validate validation set consistency
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must both be provided or both be None")
    output_dir = ensure_dir(output_dir)
    fitted: Dict[str, Any] = {}
    training_times: Dict[str, float] = {}

    # Optional MLflow integration
    use_mlflow = bool(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow

            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "intent-classification")
            run_name = os.getenv("MLFLOW_RUN_NAME", "training-run")

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=run_name)

            # Log basic parameters
            mlflow.log_params(
                {
                    "n_train_samples": len(X_train),
                    "n_val_samples": len(X_val) if X_val is not None else 0,
                    "n_models": len(models),
                    "output_dir": str(output_dir),
                }
            )
            if verbose:
                print(f"[MLflow] Logging to {mlflow_tracking_uri}, experiment: {mlflow_experiment}")
        except ImportError:
            if verbose:
                print("[MLflow] MLflow not installed. Skipping MLflow logging.")
            use_mlflow = False
        except Exception as e:
            if verbose:
                print(f"[MLflow] Failed to initialize MLflow: {e}. Skipping MLflow logging.")
            use_mlflow = False

    for name, model in models.items():
        if verbose:
            print(f"[run_training] Fitting {name}...", flush=True)
            if X_val is not None:
                print(
                    f"  Using validation set ({len(X_val)} samples) "
                    f"for model selection/early stopping",
                    flush=True,
                )

        try:
            start = time.perf_counter()

            estimator = safe_clone(model)

            # Check if model supports validation data in fit() method
            # Most sklearn models don't, but some custom models might
            import inspect

            fit_signature = inspect.signature(estimator.fit)
            fit_params = list(fit_signature.parameters.keys())

            # Try to use validation set if model supports it
            if X_val is not None and ("X_val" in fit_params or "validation_data" in fit_params):
                if "X_val" in fit_params:
                    estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                elif "validation_data" in fit_params:
                    estimator.fit(X_train, y_train, validation_data=(X_val, y_val))
            else:
                # Standard sklearn fit - validation set not used (model may use CV internally)
                estimator.fit(X_train, y_train)

            fitted[name] = estimator

            end = time.perf_counter()
            execution_time = end - start
            training_times[name] = execution_time

            print(f"\nTraining completed for {name}")
            print(f"Time taken: {execution_time:.2f} seconds\n")

            if verbose:
                print(f"Training time for {name}: {execution_time:.2f} seconds", flush=True)

            model_dir = ensure_dir(output_dir / name)

            # Save individual execution time
            with open(model_dir / f"{name}_execution_time.txt", "w") as f:
                f.write(f"Training time: {execution_time:.2f} seconds")

            # Log to MLflow if enabled
            if use_mlflow and mlflow_run is not None:
                try:
                    import mlflow

                    mlflow.log_metric(f"{name}_training_time", execution_time)
                    # Log model parameters if available
                    if hasattr(estimator, "get_params"):
                        params = estimator.get_params()
                        # Filter out non-serializable params
                        serializable_params = {
                            k: str(v)
                            for k, v in params.items()
                            if isinstance(v, (str, int, float, bool))
                        }
                        for param_name, param_value in serializable_params.items():
                            mlflow.log_param(f"{name}_{param_name}", param_value)
                except Exception as e:
                    if verbose:
                        print(f"[MLflow] Failed to log metrics for {name}: {e}")

            # --- persist model ---------------------------------------------------
            if save_models:
                with open(model_dir / "model.pkl", "wb") as f:
                    cloudpickle.dump(estimator, f)

        except Exception as e:
            if verbose:
                print(f"Error training model {name}: {e}", flush=True)
            raise

    # Save all training times to a single file
    with open(output_dir / "training_times.txt", "w") as f:
        for name, time_taken in training_times.items():
            f.write(f"{name}: {time_taken:.2f} seconds\n")

    # End MLflow run if active
    if use_mlflow and mlflow_run is not None:
        try:
            import mlflow

            mlflow.end_run()
            if verbose:
                print("[MLflow] Training run completed and logged to MLflow")
        except Exception as e:
            if verbose:
                print(f"[MLflow] Failed to end MLflow run: {e}")

    return fitted
