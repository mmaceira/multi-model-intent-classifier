"""Single-label classification metrics module.

This module provides metrics computation for single-label classification tasks.
It implements standard single-label evaluation metrics including accuracy, precision,
recall, and F1 scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

# Track which splits have already had descriptions printed to avoid repetition
_printed_descriptions: set[str] = set()


def compute_singlelabel_metrics(
    y_true: Union[np.ndarray, List[str]],
    y_pred: Union[np.ndarray, List[str]],
    split_name: str,
    output_dir: Path,
    logger: Any,
) -> Dict[str, float]:
    """Compute single-label classification metrics.

    Parameters
    ----------
    y_true : Union[np.ndarray, List[str]]
        True labels (1D array or list of strings).
    y_pred : Union[np.ndarray, List[str]]
        Predicted labels (1D array or list of strings).
    split_name : str
        Name of the split (e.g., "train", "test", "val").
    output_dir : Path
        Directory to save classification report CSV.
    logger : Any
        Logger instance for logging.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics with keys like "{split_name}_accuracy",
        "{split_name}_macro_f1", etc.
    """
    # Convert to numpy arrays if they're lists or other types
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clean data: remove NaN values and ensure consistent types
    mask = np.array(pd.notna(y_true) & pd.notna(y_pred), dtype=bool)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    # Convert to strings to ensure consistent type
    y_true_clean = np.array([str(v) for v in y_true_clean])
    y_pred_clean = np.array([str(v) for v in y_pred_clean])

    if len(y_true_clean) == 0:
        logger.warning(f"No valid data for {split_name}, skipping metrics")
        return {
            f"{split_name}_accuracy": 0.0,
            f"{split_name}_macro_f1": 0.0,
            f"{split_name}_weighted_f1": 0.0,
        }

    # Compute metrics
    metrics = {
        f"{split_name}_accuracy": accuracy_score(y_true_clean, y_pred_clean),
        f"{split_name}_macro_f1": f1_score(
            y_true_clean, y_pred_clean, average="macro", zero_division=0
        ),
        f"{split_name}_weighted_f1": f1_score(
            y_true_clean, y_pred_clean, average="weighted", zero_division=0
        ),
    }

    # Save classification report
    report_dict = classification_report(
        y_true_clean, y_pred_clean, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).T.to_csv(output_dir / f"{split_name}_report.csv")

    # Save metrics as JSON for programmatic access
    import json

    with open(output_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save and print metric descriptions
    metric_descriptions = _get_singlelabel_metric_descriptions()
    _save_and_print_metric_descriptions(metric_descriptions, split_name, output_dir, logger)

    return metrics


def get_singlelabel_overfitting_metrics() -> list[str]:
    """Get list of metrics to use for overfitting analysis in single-label tasks.

    Returns
    -------
    list[str]
        List of metric names (without split prefix) to track for overfitting.
    """
    return ["accuracy", "macro_f1", "weighted_f1"]


def get_singlelabel_summary_metrics() -> list[str]:
    """Get list of metrics to include in summary tables for single-label tasks.

    Returns
    -------
    list[str]
        List of metric names (with "test_" prefix) to include in summaries.
    """
    return [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]


def _get_singlelabel_metric_descriptions() -> Dict[str, str]:
    """Get descriptions for single-label classification metrics.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping metric names to their descriptions.
    """
    return {
        "accuracy": "Accuracy: Proportion of correctly classified samples. Number of correct predictions divided by total samples. Simple metric but can be misleading with imbalanced classes.",
        "macro_f1": "F1-Score (macro): Unweighted harmonic mean of precision and recall per class. Calculates F1 for each class and takes the arithmetic mean. Treats all classes equally, regardless of frequency. Better for evaluating performance on minority classes.",
        "weighted_f1": "F1-Score (weighted): Weighted harmonic mean of precision and recall per class. Each class has a weight proportional to the number of samples. More frequent classes have more influence on the final result. Useful when classes are imbalanced and you want to give more importance to major classes.",
    }


def _save_and_print_metric_descriptions(
    descriptions: Dict[str, str],
    split_name: str,
    output_dir: Path,
    logger: Any,
) -> None:
    """Save and print metric descriptions.

    Parameters
    ----------
    descriptions : Dict[str, str]
        Dictionary mapping metric names to descriptions.
    split_name : str
        Name of the split (e.g., "train", "test", "val").
    output_dir : Path
        Directory to save descriptions file.
    logger : Any
        Logger instance for logging.
    """
    # Save to file (always save per model directory)
    desc_file = output_dir / f"{split_name}_metric_descriptions.txt"
    with open(desc_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Single-Label Metric Descriptions - {split_name.upper()}\n")
        f.write("=" * 80 + "\n\n")
        for metric_name, description in descriptions.items():
            f.write(f"{metric_name}:\n")
            f.write(f"  {description}\n\n")

    # Print to console only once per split (across all models)
    split_key = f"singlelabel_{split_name}"
    if split_key not in _printed_descriptions:
        logger.info("\n" + "=" * 80)
        logger.info(f"Single-Label Metric Descriptions - {split_name.upper()}")
        logger.info("=" * 80)
        for metric_name, description in descriptions.items():
            logger.info(f"\n{metric_name}:")
            logger.info(f"  {description}")
        logger.info("\n" + "=" * 80)
        logger.info(f"Descriptions saved to: {desc_file}")
        _printed_descriptions.add(split_key)
