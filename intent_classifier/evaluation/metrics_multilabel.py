"""Multi-label classification metrics module.

This module provides comprehensive metrics computation for multi-label classification
tasks. It implements all standard multi-label evaluation metrics including subset
accuracy, Hamming loss, Jaccard similarity, precision, recall, and F1 scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

from intent_classifier.utils.label_utils import binarize_labels

# Track which splits have already had descriptions printed to avoid repetition
_printed_descriptions: set[str] = set()


def compute_multilabel_metrics(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    split_name: str,
    output_dir: Path,
    logger: Any,
) -> dict[str, float] | None:
    """Compute comprehensive multi-label classification metrics.

    This function computes all main multi-label metrics:
    - Subset accuracy (exact match ratio)
    - Hamming loss
    - Jaccard similarity (samples, macro, weighted)
    - Precision (macro, micro)
    - Recall (macro, micro)
    - F1-score (macro, micro)
    - Per-label metrics (precision, recall, F1 per class)

    Parameters
    ----------
    y_true : Sequence[Sequence[str]]
        True labels in multi-label format (list of lists).
    y_pred : Sequence[Sequence[str]]
        Predicted labels in multi-label format (list of lists).
    split_name : str
        Name of the split (e.g., "train", "test", "val").
    output_dir : Path
        Directory to save per-label metrics CSV.
    logger : Any
        Logger instance for logging.

    Returns
    -------
    Dict[str, float] | None
        Dictionary of computed metrics with keys like "{split_name}_accuracy",
        "{split_name}_hamming_loss", etc. Returns None if no classes found.
    """
    # Get all unique classes from both true and predicted labels
    all_classes = sorted(
        set(tag for labels in y_true if labels for tag in labels)
        | set(tag for labels in y_pred if labels for tag in labels)
    )

    if not all_classes:
        logger.warning(f"No classes found for {split_name}, skipping metrics")
        return None

    # Convert to binary format for sklearn metrics
    y_true_binary, _ = binarize_labels(y_true, classes=all_classes)
    y_pred_binary, _ = binarize_labels(y_pred, classes=all_classes)

    if y_true_binary.size == 0 or y_pred_binary.size == 0:
        logger.warning(f"Empty binary arrays for {split_name}, skipping metrics")
        return None

    # Compute aggregated metrics
    metrics = _compute_aggregated_metrics(y_true_binary, y_pred_binary, split_name)

    # Compute and save per-label metrics
    per_label_metrics = _compute_per_label_metrics(y_true_binary, y_pred_binary, all_classes)

    if per_label_metrics:
        per_label_df = pd.DataFrame(per_label_metrics)
        per_label_df = per_label_df.sort_values("f1_score", ascending=False)
        per_label_df.to_csv(
            output_dir / f"{split_name}_per_label_metrics.csv",
            index=False,
        )
        logger.info(f"Saved per-label metrics for {split_name} ({len(per_label_metrics)} labels)")

    # Save metrics as JSON for programmatic access
    import json

    with open(output_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save and print metric descriptions
    metric_descriptions = _get_multilabel_metric_descriptions()
    _save_and_print_metric_descriptions(metric_descriptions, split_name, output_dir, logger)

    return metrics


def _compute_aggregated_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    """Compute aggregated multi-label metrics.

    Parameters
    ----------
    y_true_binary : np.ndarray
        Binary matrix of true labels (n_samples, n_classes).
    y_pred_binary : np.ndarray
        Binary matrix of predicted labels (n_samples, n_classes).
    split_name : str
        Name of the split for metric keys.

    Returns
    -------
    Dict[str, float]
        Dictionary of aggregated metrics.
    """
    # 1. Subset accuracy (exact match ratio) - MAIN METRIC
    subset_acc = accuracy_score(y_true_binary, y_pred_binary)

    # 2. Hamming loss (lower is better) - MAIN METRIC
    hamming = hamming_loss(y_true_binary, y_pred_binary)

    # 3. Jaccard similarity (intersection over union) - MAIN METRIC
    jaccard_samples = jaccard_score(
        y_true_binary, y_pred_binary, average="samples", zero_division=0
    )
    jaccard_macro = jaccard_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    jaccard_weighted = jaccard_score(
        y_true_binary, y_pred_binary, average="weighted", zero_division=0
    )

    # 4. Precision (macro and micro) - MAIN METRIC
    precision_macro = precision_score(
        y_true_binary, y_pred_binary, average="macro", zero_division=0
    )
    precision_micro = precision_score(
        y_true_binary, y_pred_binary, average="micro", zero_division=0
    )

    # 5. Recall (macro and micro) - MAIN METRIC
    recall_macro = recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    recall_micro = recall_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)

    # 6. F1 scores (macro and micro) - MAIN METRIC
    f1_macro = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    f1_micro = f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)

    return {
        f"{split_name}_accuracy": subset_acc,
        f"{split_name}_hamming_loss": hamming,
        f"{split_name}_jaccard_samples": jaccard_samples,
        f"{split_name}_jaccard_macro": jaccard_macro,
        f"{split_name}_jaccard_weighted": jaccard_weighted,
        f"{split_name}_precision_macro": precision_macro,
        f"{split_name}_precision_micro": precision_micro,
        f"{split_name}_recall_macro": recall_macro,
        f"{split_name}_recall_micro": recall_micro,
        f"{split_name}_macro_f1": f1_macro,
        f"{split_name}_micro_f1": f1_micro,
        f"{split_name}_weighted_f1": f1_micro,  # Micro F1 as weighted equivalent
    }


def _compute_per_label_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    all_classes: list[str],
) -> list[dict[str, Any]]:
    """Compute per-label metrics (precision, recall, F1 per class).

    Parameters
    ----------
    y_true_binary : np.ndarray
        Binary matrix of true labels (n_samples, n_classes).
    y_pred_binary : np.ndarray
        Binary matrix of predicted labels (n_samples, n_classes).
    all_classes : List[str]
        List of all class labels.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with per-label metrics.
    """
    per_label_metrics = []

    for i, class_name in enumerate(all_classes):
        y_true_class = y_true_binary[:, i]
        y_pred_class = y_pred_binary[:, i]

        # Skip if class has no true positives (avoid division by zero)
        if y_true_class.sum() == 0 and y_pred_class.sum() == 0:
            continue

        prec = precision_score(y_true_class, y_pred_class, zero_division=0)
        rec = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        support = int(y_true_class.sum())

        per_label_metrics.append(
            {
                "label": class_name,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "support": support,
            }
        )

    return per_label_metrics


def get_multilabel_overfitting_metrics() -> list[str]:
    """Get list of metrics to use for overfitting analysis in multi-label tasks.

    Returns
    -------
    List[str]
        List of metric names (without split prefix) to track for overfitting.
    """
    return [
        "accuracy",
        "macro_f1",
        "micro_f1",
        "hamming_loss",
        "jaccard_samples",
        "jaccard_macro",
        "precision_macro",
        "recall_macro",
    ]


def get_multilabel_summary_metrics() -> list[str]:
    """Get list of metrics to include in summary tables for multi-label tasks.

    Returns
    -------
    List[str]
        List of metric names (with "test_" prefix) to include in summaries.
    """
    return [
        "test_accuracy",
        "test_macro_f1",
        "test_micro_f1",
        "test_hamming_loss",
        "test_jaccard_samples",
        "test_jaccard_macro",
        "test_precision_macro",
        "test_recall_macro",
    ]


def _get_multilabel_metric_descriptions() -> dict[str, str]:
    """Get descriptions for multi-label classification metrics.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping metric names to their descriptions.
    """
    return {
        "accuracy": (
            "Subset Accuracy (Exact Match Ratio): Proportion of samples where "
            "all predicted labels match exactly with the true labels. Strict "
            "metric: all labels must be correct."
        ),
        "hamming_loss": (
            "Hamming Loss: Average proportion of incorrect labels per sample. "
            "Counts errors per label. Lower values are better (0 = perfect, "
            "1 = worst)."
        ),
        "jaccard_samples": (
            "Jaccard Similarity (per sample): Average of intersection over "
            "union of labels for each sample. Measures similarity between "
            "predicted and true label sets."
        ),
        "jaccard_macro": (
            "Jaccard Similarity (macro): Unweighted mean of Jaccard similarity "
            "per class. Treats all classes equally, regardless of frequency."
        ),
        "jaccard_weighted": (
            "Jaccard Similarity (weighted): Weighted mean by the frequency of "
            "each class. More frequent classes have more weight."
        ),
        "precision_macro": (
            "Precision (macro): Unweighted mean of precision per class. "
            "Proportion of correct positive predictions per class, arithmetic "
            "mean."
        ),
        "precision_micro": (
            "Precision (micro): Precisions aggregated at global level. Counts "
            "all true positives, false positives and calculates global "
            "precision."
        ),
        "recall_macro": (
            "Recall (macro): Unweighted mean of recall per class. Proportion "
            "of actual cases detected per class, arithmetic mean."
        ),
        "recall_micro": (
            "Recall (micro): Recalls aggregated at global level. Counts all "
            "true positives, false negatives and calculates global recall."
        ),
        "macro_f1": (
            "F1-Score (macro): Unweighted harmonic mean of precision and "
            "recall per class. Balances precision and recall, treating all "
            "classes equally."
        ),
        "micro_f1": (
            "F1-Score (micro): F1 aggregated at global level. Equivalent to "
            "micro precision/recall when they are equal. Better for imbalanced "
            "datasets."
        ),
    }


def _save_and_print_metric_descriptions(
    descriptions: dict[str, str],
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
        f.write(f"Multi-Label Metric Descriptions - {split_name.upper()}\n")
        f.write("=" * 80 + "\n\n")
        for metric_name, description in descriptions.items():
            f.write(f"{metric_name}:\n")
            f.write(f"  {description}\n\n")

    # Print to console only once per split (across all models)
    split_key = f"multilabel_{split_name}"
    if split_key not in _printed_descriptions:
        logger.info("\n" + "=" * 80)
        logger.info(f"Multi-Label Metric Descriptions - {split_name.upper()}")
        logger.info("=" * 80)
        for metric_name, description in descriptions.items():
            logger.info(f"\n{metric_name}:")
            logger.info(f"  {description}")
        logger.info("\n" + "=" * 80)
        logger.info(f"Descriptions saved to: {desc_file}")
        _printed_descriptions.add(split_key)
