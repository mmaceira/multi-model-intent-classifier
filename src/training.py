"""Training Module for Text Classification

Version: 1.0.0
Author: Reuters RAG Classifier Team
License: MIT

Provides unified training workflow for text classification:
- Multiple model training
- Artifact persistence
- Prediction generation
- Probability score handling
- Progress logging

Dependencies:
- pathlib
- typing
- joblib
- numpy
- pandas
- sklearn.base

Usage:
    >>> models = {
    ...     'svm': SVC(probability=True),
    ...     'rf': RandomForestClassifier()
    ... }
    >>> fitted_models = run_training(
    ...     models=models,
    ...     X_train=train_texts,
    ...     y_train=train_labels,
    ...     X_test=test_texts,
    ...     y_test=test_labels,
    ...     output_dir='experiments/run_001'
    ... )
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Any, Optional, Union

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
    X_test: Optional[Sequence[str]] = None,
    y_test: Optional[Sequence[Any]] = None,
    output_dir: Union[str, Path] = "artefacts",
    save_models: bool = True,
    save_train_predictions: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit models and persist training artifacts.
    
    Args:
        models: Mapping of model names to unfitted estimators
        X_train: Training text data
        y_train: Training labels
        X_test: Optional test text data
        y_test: Optional test labels
        output_dir: Root directory for artifacts
        save_models: Whether to save fitted models
        save_train_predictions: Whether to save training predictions
        verbose: Whether to print progress
        
    Returns:
        Dict mapping model names to fitted estimators
        
    Raises:
        ValueError: Invalid test data or model requirements
    """
    # Validate test data consistency
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must both be provided or both be None")
        
    output_dir = ensure_dir(output_dir)
    fitted: Dict[str, Any] = {}

    for name, model in models.items():
        if verbose:
            print(f"[run_training] Fitting {name}...", flush=True)

        try:
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
                            
        except Exception as e:
            if verbose:
                print(f"Error training model {name}: {e}", flush=True)
            raise

    return fitted
