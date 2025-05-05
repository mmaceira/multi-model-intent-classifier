"""Training Module for Text Classification

This module provides a unified training workflow for text classification models, ensuring
consistent model training, artifact persistence, and prediction generation. It is designed
to work seamlessly with the evaluation module, enabling reproducible model assessment.

Key Features:
- Unified training interface for multiple models
- Consistent artifact persistence
- Safe model cloning
- Training and test prediction generation
- Probability score handling
- Comprehensive error handling
- Progress logging

Main Function:
run_training:
    - Trains multiple estimators in sequence
    - Persists trained models (optional)
    - Saves predictions for both training and test sets
    - Handles probability scores when available
    - Returns dictionary of fitted estimators

Artifact Structure:
output_dir/
├── model_name1/
│   ├── model.joblib          # Trained model
│   ├── train_predictions.csv # Training set predictions
│   ├── train_prob.npy       # Training probability scores
│   ├── test_predictions.csv # Test set predictions
│   └── test_prob.npy       # Test probability scores
└── model_name2/
    └── ...

Helper Functions:
- _ensure_dir: Create directory if it doesn't exist

Requirements for Models:
- Must implement fit() and predict()
- Should implement predict_proba() for ROC curves
- Must be compatible with scikit-learn's clone utility

Dependencies:
- pathlib
- typing
- joblib
- numpy
- pandas
- sklearn.base

Example Usage:
    >>> # Define models to train
    >>> models = {
    ...     'svm': SVC(probability=True),
    ...     'rf': RandomForestClassifier()
    ... }
    >>> 
    >>> # Run training workflow
    >>> fitted_models = run_training(
    ...     models=models,
    ...     X_train=train_texts,
    ...     y_train=train_labels,
    ...     X_test=test_texts,
    ...     y_test=test_labels,
    ...     output_dir='experiments/run_001',
    ...     save_models=True,
    ...     save_train_predictions=True,
    ...     verbose=True
    ... )
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone as safe_clone
from src.utils.file_ops import ensure_dir

def run_training(
    models: Dict[str, Any],
    *,
    X_train: Sequence[str],
    y_train: Sequence[Any],
    X_test: Sequence[str] | None = None,
    y_test: Sequence[Any] | None = None,
    output_dir: str | Path = "artefacts",
    save_models: bool = True,
    save_train_predictions: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit *models* and persist artefacts.

    Parameters
    ----------
    models
        Mapping ``name → unfitted estimator``. Estimators **must** implement
        `fit`, `predict`, and, if ROC curves are required, `predict_proba`.
    X_train, y_train
        Training data.
    X_test, y_test
        Test data – pass these **once** here so that downstream evaluation
        notebooks can operate purely on saved artefacts.
    output_dir
        Root directory that will contain one sub‑folder per model.
    save_models
        Persist fitted estimators to disk.
    save_train_predictions
        Save training predictions and probabilities (for evaluating train metrics).
    verbose
        Emit progress to stdout.

    Returns
    -------
    Dict[str, Any]
        Mapping ``name → fitted estimator``.
    """
    output_dir = ensure_dir(output_dir)
    fitted: Dict[str, Any] = {}

    for name, model in models.items():
        if verbose:
            print(f"[run_training] Fitting {name}…", flush=True)

        estimator = safe_clone(model)
        estimator.fit(X_train, y_train)
        fitted[name] = estimator

        model_dir = ensure_dir(output_dir / name)

        # --- persist model ---------------------------------------------------
        if save_models:
            joblib.dump(estimator, model_dir / "model.joblib")

        # --- persist training predictions -----------------------------------
        if save_train_predictions:
            y_train_pred = estimator.predict(X_train)
            df_train = pd.DataFrame({"y_true": y_train, "y_pred": y_train_pred})
            df_train.to_csv(model_dir / "train_predictions.csv", index=False)

            if hasattr(estimator, "predict_proba"):
                try:
                    y_train_prob = estimator.predict_proba(X_train)
                    np.save(model_dir / "train_prob.npy", y_train_prob)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not save train probabilities for {name}: {e}", flush=True)
        
        # --- persist test predictions ---------------------------------------
        if X_test is not None and y_test is not None:
            y_pred = estimator.predict(X_test)
            df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
            df.to_csv(model_dir / "test_predictions.csv", index=False)

            if hasattr(estimator, "predict_proba"):
                try:
                    y_prob = estimator.predict_proba(X_test)
                    np.save(model_dir / "test_prob.npy", y_prob)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not save test probabilities for {name}: {e}", flush=True)

    return fitted
