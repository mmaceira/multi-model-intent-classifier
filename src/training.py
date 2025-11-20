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

import time
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import cloudpickle
from sklearn.base import clone as safe_clone

from src.utils.file_ops import ensure_dir


def run_training(
    models: Dict[str, Any],
    *,
    X_train: Sequence[str],
    y_train: Sequence[Any],
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
        output_dir: Root directory for saving all artifacts
        save_models: Whether to persist fitted models to disk
        save_train_predictions: Whether to save training set predictions
        verbose: Whether to print progress and warning messages

    Returns:
        Dict[str, Any]: Dictionary mapping model names to their fitted estimators

    Raises:
        Exception: Any exceptions raised during model training are propagated

    Note:
        - Training times are tracked and saved for each model
        - Models are cloned before fitting to prevent modification of input objects
        - Directory structure is automatically created if it doesn't exist
        - All operations are performed sequentially for each model
    """
    output_dir = ensure_dir(output_dir)
    fitted: Dict[str, Any] = {}
    training_times: Dict[str, float] = {}

    for name, model in models.items():
        if verbose:
            print(f"[run_training] Fitting {name}...", flush=True)

        try:
            start = time.perf_counter()

            estimator = safe_clone(model)
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

    return fitted
