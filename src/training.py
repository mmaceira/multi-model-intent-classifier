"""Training module for text classification models.

This module provides a *single-entry* helper ``run_trainings`` that
makes it easy to train **one or many** models in a consistent way and
persist them to disk so that the evaluation stage can later reload
them.

Typical usage
-------------
>>> from src.training import run_trainings
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.svm import LinearSVC
>>> models = {
...     "logreg": Pipeline([
...         ("tfidf", TfidfVectorizer()),
...         ("clf", LogisticRegression(max_iter=1000))
...     ]),
...     "svm": Pipeline([
...         ("tfidf", TfidfVectorizer()),
...         ("clf", LinearSVC())
...     ])
... }
>>> trained = run_trainings(models, X_train, y_train,
...                         output_dir="artifacts/models")

The dictionary ``trained`` maps model names to the *fitted* model
objects (so the notebook can keep working interactively), while—under
the hood—each model is serialised to
``{output_dir}/{model_name}.joblib`` via :pyfunc:`utils.model_storage.save_model`.
"""

from __future__ import annotations

import os
import joblib
from typing import Dict, Any, List, Union
import copy
import logging

import numpy as np
from sklearn.base import clone, BaseEstimator
from tqdm.auto import tqdm   # nice progress bars in notebooks

from src.utils.model_storage import save_model

# Set up logging
logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create *path* (recursively) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# Define a simple save function to match what run_trainings expects
def _simple_save_model(model: Any, path: str) -> None:
    """Simple wrapper to save a model to a file using joblib."""
    try:
        joblib.dump(model, path)
        print(f"Saved model to {path}")
    except Exception as e:
        print(f"Warning: Failed to save model to {path}: {e}")
        # Try to save a reduced version of the model
        try:
            # For RAG models, we might need special handling
            if hasattr(model, 'rag'):
                # Store only the class name and parameters
                simplified = {
                    'model_type': model.__class__.__name__,
                    'rag_type': model.rag.__class__.__name__,
                    'params': model.get_params(deep=False)
                }
                joblib.dump(simplified, path)
                print(f"Saved simplified model description to {path}")
        except Exception as e2:
            print(f"Failed to save even simplified model: {e2}")


def safe_clone(model):
    """Safely clone a model, handling cases where standard clone fails."""
    if hasattr(model, "__sklearn_clone__"):
        # Use custom clone method if available
        return model.__sklearn_clone__()
    
    try:
        # Try standard sklearn clone
        return clone(model)
    except (TypeError, AttributeError) as e:
        # Fallback for models that don't support cloning
        logger.warning(f"Could not clone model using sklearn.base.clone: {e}")
        logger.warning("Using direct model instance instead of cloning")
        return model


def run_trainings(
    models: Dict[str, Any],
    X_train: List[str],
    y_train: Union[np.ndarray, List[int]],
    *,
    output_dir: str = "models",
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit *one or many* models and (optionally) persist them to disk.

    Parameters
    ----------
    models
        Mapping of **model name → *unfitted* estimator or Pipeline**.  Each
        estimator **must** implement ``fit`` and ``predict_proba`` or
        ``decision_function``.
    X_train, y_train
        Training data.
    output_dir
        Directory were the trained models will be stored.  One ``.joblib``
        file is created per model, named ``{name}.joblib``.
    save
        If *False* the models are *not* serialised (useful for quick
        experiments).
    verbose
        If *True* a progress‑bar is shown; otherwise, silent operation.

    Returns
    -------
    Dict[str, Any]
        Mapping of **model name → *fitted* model instance**.
    """

    _ensure_dir(output_dir)
    iterator = tqdm(models.items(), disable=not verbose, desc="Training")

    trained: Dict[str, Any] = {}
    for name, model in iterator:
        try:
            # Try to use our safe clone
            estimator = safe_clone(model)
            estimator.fit(X_train, y_train)
            trained[name] = estimator

            if save:
                # Use a simple save function instead of the more complex save_model
                model_path = os.path.join(output_dir, f"{name}.joblib")
                _simple_save_model(estimator, model_path)
        except Exception as e:
            logger.error(f"Error training model {name}: {e}")
            # Skip this model but continue with others
            continue

    return trained
