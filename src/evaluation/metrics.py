"""
Metrics computation module.

This module provides functions for computing various evaluation metrics
for text classification models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
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
        f"{split['prefix']}micro_f1": f1_score(split["y_true"], y_pred, average="micro"),
        f"{split['prefix']}weighted_f1": f1_score(split["y_true"], y_pred, average="weighted"),
    }

    # Compute confusion matrix (normalized)
    cm = confusion_matrix(split["y_true"], y_pred, labels=classes)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Save normalized confusion matrix
    cm_df = pd.DataFrame(cm_normalized, index=classes, columns=classes)
    cm_df.to_csv(split["out_dir"] / f"{split['name']}_confusion_matrix_normalized.csv")

    # Also save raw confusion matrix
    cm_df_raw = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df_raw.to_csv(split["out_dir"] / f"{split['name']}_confusion_matrix.csv")

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

        # Compute Expected Calibration Error (ECE)
        ece = compute_ece(split["y_true"], y_pred, y_prob, classes)
        metrics[f"{split['prefix']}ece"] = ece

        # Generate per-class PR curves
        generate_per_class_pr_curves(
            split["y_true"],
            y_prob,
            classes,
            split["out_dir"] / f"{split['name']}_pr_curves_per_class.png",
        )

    return y_pred, y_prob, metrics


def analyze_text_features(predictions_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Analyze text features that might contribute to classification errors.

    Returns a dataframe with text statistics for correct and incorrect predictions.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.

    Returns
    -------
    pd.DataFrame
        DataFrame containing text feature statistics for each model and prediction type.
    """
    all_rows = []

    for model_name, splits in predictions_dict.items():
        # We'll analyze both train and test sets
        for split_name, df in splits.items():
            # Add features
            df["is_correct"] = df["y_true"] == df["y_pred"]
            df["text_length"] = df["text"].apply(lambda x: len(str(x)))
            df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

            # Group by correct/incorrect
            for is_correct in [True, False]:
                subset = df[df["is_correct"] == is_correct]
                if len(subset) > 0:
                    all_rows.append(
                        {
                            "model": model_name,
                            "split": split_name,
                            "is_correct": is_correct,
                            "count": len(subset),
                            "avg_text_length": subset["text_length"].mean(),
                            "avg_word_count": subset["word_count"].mean(),
                        }
                    )

    return pd.DataFrame(all_rows)


def analyze_text_characteristics(misclassified_df: pd.DataFrame) -> None:
    """Analyze text characteristics of misclassified examples.

    Parameters
    ----------
    misclassified_df : pd.DataFrame
        DataFrame containing columns: 'text', 'true_label', 'pred_label'
    """
    # Calculate text length and word count
    misclassified_df["text_length"] = misclassified_df["text"].apply(lambda x: len(str(x)))
    misclassified_df["word_count"] = misclassified_df["text"].apply(lambda x: len(str(x).split()))

    # Group by true label and compute statistics
    (
        misclassified_df.groupby("true_label")
        .agg({"text_length": ["count", "mean", "std"], "word_count": ["mean", "std"]})
        .round(2)
    )

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Text length distribution
    sns.histplot(data=misclassified_df, x="text_length", hue="true_label", ax=ax1)
    ax1.set_title("Distribution of Text Length by True Label")
    ax1.set_xlabel("Text Length")

    # Word count distribution
    sns.histplot(data=misclassified_df, x="word_count", hue="true_label", ax=ax2)
    ax2.set_title("Distribution of Word Count by True Label")
    ax2.set_xlabel("Word Count")

    plt.tight_layout()
    plt.close()  # Close figure instead of showing to prevent pop-ups

    # Calculate correlation between text length and word count
    corr = misclassified_df[["text_length", "word_count"]].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Correlation between Text Length and Word Count")
    plt.close()  # Close figure instead of showing to prevent pop-ups


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well-calibrated the predicted probabilities are.
    Lower values indicate better calibration.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities (shape: [n_samples, n_classes])
    classes : np.ndarray
        Class labels
    n_bins : int, default=10
        Number of bins for calibration error computation

    Returns
    -------
    float
        Expected Calibration Error
    """
    from sklearn.preprocessing import label_binarize

    # Get probabilities for predicted class
    if len(classes) == 2:
        # Binary classification: use probability of positive class
        prob_pred = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.flatten()
        y_true_bin = (y_true == classes[1]).astype(int)
    else:
        # Multiclass: use max probability (confidence)
        prob_pred = np.max(y_prob, axis=1)
        # Binarize for comparison
        y_true_bin = label_binarize(y_true, classes=classes)
        y_true_bin = np.argmax(y_true_bin, axis=1) == np.argmax(y_prob, axis=1)

    # Bin the probabilities
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        # Find samples in this bin
        in_bin = (prob_pred > bin_lower) & (prob_pred <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = (
                y_true_bin[in_bin].mean()
                if isinstance(y_true_bin, np.ndarray)
                else (y_true[in_bin] == y_pred[in_bin]).mean()
            )
            # Average confidence in this bin
            avg_confidence_in_bin = prob_pred[in_bin].mean()
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def generate_per_class_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    output_path: Path,
    n_classes_to_plot: int = 10,
) -> None:
    """Generate per-class Precision-Recall curves.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities (shape: [n_samples, n_classes])
    classes : np.ndarray
        Class labels
    output_path : Path
        Path to save the plot
    n_classes_to_plot : int, default=10
        Maximum number of classes to plot (to avoid overcrowding)
    """
    from sklearn.preprocessing import label_binarize

    # Binarize labels for multiclass
    if len(classes) > 2:
        y_true_bin = label_binarize(y_true, classes=classes)
    else:
        y_true_bin = y_true.reshape(-1, 1)
        if y_prob.shape[1] == 1:
            y_prob = np.hstack([1 - y_prob, y_prob])

    # Limit number of classes to plot
    n_classes = min(len(classes), n_classes_to_plot)
    classes_to_plot = classes[:n_classes]

    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(classes_to_plot):
        if i >= y_prob.shape[1]:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"{class_label}", alpha=0.7, linewidth=2)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Per-Class Precision-Recall Curves", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
