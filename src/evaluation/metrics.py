"""
Metrics computation module.

This module provides functions for computing various evaluation metrics
for text classification models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def compute_metrics(
    model_dir: Path,
    split: Dict[str, Any],
    classes: np.ndarray,
    is_multiclass: bool,
    logger: logging.Logger,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]]:
    """Compute metrics for a given model and data split.
    
    Parameters
    ----------
    model_dir : Path
        Directory containing model artifacts.
    split : Dict[str, Any]
        Dictionary containing split information and data.
    classes : np.ndarray
        Unique class labels.
    is_multiclass : bool
        Whether the task is multiclass classification.
    logger : logging.Logger
        Logger instance for logging progress.
        
    Returns
    -------
    Optional[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]]
        Tuple containing:
        - y_pred: Predicted labels
        - y_prob: Predicted probabilities (if available)
        - metrics: Dictionary of computed metrics
    """
    # Load predictions
    pred_path = model_dir / split["pred_file"]
    if not pred_path.exists():
        if split["name"] == "test":
            logger.error(f"Predictions not found at {pred_path}")
            return None
        return None

    df_pred = pd.read_csv(pred_path)
    y_pred = df_pred["y_pred"].values

    # Compute basic metrics
    metrics = {
        f"{split['prefix']}accuracy": accuracy_score(split["y_true"], y_pred),
        f"{split['prefix']}macro_f1": f1_score(split["y_true"], y_pred, average="macro"),
        f"{split['prefix']}weighted_f1": f1_score(split["y_true"], y_pred, average="weighted"),
    }
    
    # Save classification report
    report_dict = classification_report(split["y_true"], y_pred, output_dict=True)
    pd.DataFrame(report_dict).T.to_csv(split["out_dir"] / f"{split['name']}_report.csv")
    
    # Load probabilities if available
    y_prob = None
    prob_path = model_dir / split["prob_file"]
    if prob_path.exists():
        logger.info(f"Loading probabilities from {prob_path}")
        y_prob = np.load(prob_path)
        
        # Ensure probabilities are properly normalized
        if not np.allclose(y_prob.sum(axis=1), 1.0):
            logger.warning("Probabilities do not sum to 1, normalizing...")
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Compute probability-based metrics
        if is_multiclass:
            metrics[f"{split['prefix']}roc_auc"] = roc_auc_score(
                split["y_true_bin"], y_prob, multi_class="ovr"
            )
            metrics[f"{split['prefix']}avg_precision"] = average_precision_score(
                split["y_true_bin"], y_prob
            )
        else:
            metrics[f"{split['prefix']}roc_auc"] = roc_auc_score(split["y_true"], y_prob)
            metrics[f"{split['prefix']}avg_precision"] = average_precision_score(
                split["y_true"], y_prob
            )
    
    return y_pred, y_prob, metrics 