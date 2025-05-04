"""Unified training workflow for text classification models.

This module defines a *single* `run_training` function that:
1. trains one or more estimators,
2. optionally persists them,
3. persists their predictions on the *test* split so that the evaluation
   stage can run **without touching the models again**.

The API is intentionally symmetric with `src.evaluation.run_evaluations`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone as safe_clone

def _ensure_dir(path: str | Path) -> Path:
    """Create *path* (including parents) if it does not exist and return a `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

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
    output_dir = _ensure_dir(output_dir)
    fitted: Dict[str, Any] = {}

    for name, model in models.items():
        if verbose:
            print(f"[run_training] Fitting {name}…", flush=True)

        estimator = safe_clone(model)
        estimator.fit(X_train, y_train)
        fitted[name] = estimator

        model_dir = _ensure_dir(output_dir / name)

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
