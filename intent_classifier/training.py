"""Training utilities for fitting text classifiers and persisting models."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cloudpickle
from sklearn.base import clone as safe_clone

from intent_classifier.utils.embeddings import EmbeddingServiceError
from intent_classifier.utils.file_ops import ensure_dir
from intent_classifier.utils.method_logger import get_logger
from intent_classifier.utils.slugify import slugify_model_id


def run_training(
    models: dict[str, Any],
    *,
    X_train: Sequence[str],
    y_train: Sequence[Any],
    X_val: Sequence[str] | None = None,
    y_val: Sequence[Any] | None = None,
    output_dir: str | Path = "artefacts",
    save_models: bool = True,
    save_train_predictions: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
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
    # Disable method logging during training
    get_logger().disable()

    # Validate validation set consistency
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must both be provided or both be None")
    output_dir = ensure_dir(output_dir)
    fitted: dict[str, Any] = {}
    training_times: dict[str, float] = {}

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

    total_models = len(models)
    for idx, (name, model) in enumerate(models.items(), 1):
        if verbose:
            print("\n" + "=" * 60)
            print(f"[{idx}/{total_models}] Training algorithm: {name}")
            print("=" * 60)
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(
                    f"Validation samples: {len(X_val)} (used for model selection/early stopping)",
                    flush=True,
                )
            print("Starting training...", flush=True)

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

            if verbose:
                print(f"\n✅ Algorithm '{name}' training completed successfully")
                print(
                    f"   Time taken: {execution_time:.2f} seconds "
                    f"({execution_time / 60:.2f} minutes)"
                )
                if idx < total_models:
                    print(f"   Progress: {idx}/{total_models} algorithms completed\n")

            # Slugify model name for filesystem use (consistent with finalize)
            safe_dir_name = slugify_model_id(name)
            model_dir = ensure_dir(output_dir / safe_dir_name)

            # Save individual execution time
            # Use a filesystem‑safe file name (model names may contain "/" etc.)
            safe_name = slugify_model_id(name)
            with open(model_dir / f"{safe_name}_execution_time.txt", "w", encoding="utf-8") as f:
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

        except EmbeddingServiceError as e:
            # Recoverable path: external embedding backend (e.g. Ollama) is unavailable.
            # We log and **skip** this model while continuing with the rest.
            if verbose:
                print(
                    f"\n⚠️  Skipping algorithm '{name}' due to embedding backend error: {e}",
                    flush=True,
                )
                print(
                    f"   Progress: {idx - 1}/{total_models} algorithms completed before skip\n",
                    flush=True,
                )
            continue
        except Exception as e:
            if verbose:
                print(f"\n❌ Error training algorithm '{name}': {e}", flush=True)
                print(
                    f"   Progress: {idx - 1}/{total_models} algorithms completed before error\n",
                    flush=True,
                )
            raise

    # Save all training times to a single file
    with open(output_dir / "training_times.txt", "w", encoding="utf-8") as f:
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

    if verbose:
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total algorithms trained: {len(fitted)}/{total_models}")
        total_time = sum(training_times.values())
        print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print("\nPer-algorithm training times:")
        for name, time_taken in sorted(training_times.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {time_taken:.2f}s ({time_taken / 60:.2f}min)")
        print("=" * 60)

    return fitted
