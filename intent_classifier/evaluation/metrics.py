"""
Metrics computation module.

This module provides functions for computing various evaluation metrics
for text classification models.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    split: dict[str, Any],
    classes: np.ndarray,
    is_multiclass: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]] | None:
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


def analyze_text_features(predictions_dict: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
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
            # Create a copy to avoid modifying original
            df = df.copy()

            # Parse multilabel strings (comma-separated) into sets for comparison
            def parse_labels(label_str: str) -> set[str]:
                """Parse comma-separated label string into a set."""
                if pd.isna(label_str) or label_str == "":
                    return set()
                return {tag.strip() for tag in str(label_str).split(",") if tag.strip()}

            # Compute correctness: for multilabel, compare sets
            df["y_true_set"] = df["y_true"].apply(parse_labels)
            df["y_pred_set"] = df["y_pred"].apply(parse_labels)
            df["is_correct"] = df["y_true_set"] == df["y_pred_set"]

            # Add text features
            df["text_length"] = df["text"].apply(lambda x: len(str(x)))
            df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

            # For multilabel, also compute number of labels
            df["n_true"] = df["y_true_set"].apply(len)
            df["n_pred"] = df["y_pred_set"].apply(len)

            # Group by correct/incorrect and compute statistics
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
                            "avg_n_true": subset["n_true"].mean(),
                            "avg_n_pred": subset["n_pred"].mean(),
                        }
                    )

    return pd.DataFrame(all_rows)


def analyze_per_bucket_metrics(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Compute metrics by number of labels (k=1, k=2, k>=3).

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics per label count bucket.
    """
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

    from intent_classifier.utils.label_utils import binarize_labels

    all_rows = []

    for model_name, splits in predictions_dict.items():
        for split_name, df in splits.items():
            df = df.copy()

            # Parse multilabel strings
            def parse_labels(label_str: str) -> set[str]:
                if pd.isna(label_str) or label_str == "":
                    return set()
                return {tag.strip() for tag in str(label_str).split(",") if tag.strip()}

            y_true_list = [parse_labels(str(row)) for row in df["y_true"]]
            y_pred_list = [parse_labels(str(row)) for row in df["y_pred"]]

            # Get all classes
            all_classes = sorted(
                set(tag for labels in y_true_list for tag in labels)
                | set(tag for labels in y_pred_list for tag in labels)
            )

            if not all_classes:
                continue

            # Convert to binary (convert sets to lists)
            y_true_binary, _ = binarize_labels(
                [list(labels) for labels in y_true_list], classes=all_classes
            )
            y_pred_binary, _ = binarize_labels(
                [list(labels) for labels in y_pred_list], classes=all_classes
            )

            # Compute n_true and n_pred
            n_true = np.array([len(labels) for labels in y_true_list])
            n_pred = np.array([len(labels) for labels in y_pred_list])

            # Define buckets
            buckets = [
                (1, 1, "k=1"),
                (2, 2, "k=2"),
                (3, None, "k>=3"),
            ]

            for min_k, max_k, bucket_name in buckets:
                if max_k is None:
                    mask = n_true >= min_k
                else:
                    mask = (n_true >= min_k) & (n_true <= max_k)

                if mask.sum() == 0:
                    continue

                y_true_bucket = y_true_binary[mask]
                y_pred_bucket = y_pred_binary[mask]

                if y_true_bucket.size == 0:
                    continue

                # Compute metrics for this bucket
                accuracy = accuracy_score(y_true_bucket, y_pred_bucket)
                f1_micro = f1_score(y_true_bucket, y_pred_bucket, average="micro", zero_division=0)
                f1_macro = f1_score(y_true_bucket, y_pred_bucket, average="macro", zero_division=0)
                hamming = hamming_loss(y_true_bucket, y_pred_bucket)
                jaccard = jaccard_score(
                    y_true_bucket, y_pred_bucket, average="samples", zero_division=0
                )

                all_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "bucket": bucket_name,
                        "n_samples": int(mask.sum()),
                        "accuracy": accuracy,
                        "f1_micro": f1_micro,
                        "f1_macro": f1_macro,
                        "hamming_loss": hamming,
                        "jaccard": jaccard,
                        "mean_n_true": float(n_true[mask].mean()),
                        "mean_n_pred": float(n_pred[mask].mean()),
                    }
                )

    return pd.DataFrame(all_rows)


def analyze_coverage_per_label(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Analyze prediction coverage per label (true_support vs predicted_support).

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.

    Returns
    -------
    pd.DataFrame
        DataFrame with coverage metrics per label.
    """
    all_rows = []

    for model_name, splits in predictions_dict.items():
        for split_name, df in splits.items():
            df = df.copy()

            # Parse multilabel strings
            def parse_labels(label_str: str) -> set[str]:
                if pd.isna(label_str) or label_str == "":
                    return set()
                return {tag.strip() for tag in str(label_str).split(",") if tag.strip()}

            y_true_list = [parse_labels(str(row)) for row in df["y_true"]]
            y_pred_list = [parse_labels(str(row)) for row in df["y_pred"]]

            # Get all classes
            all_classes = sorted(
                set(tag for labels in y_true_list for tag in labels)
                | set(tag for labels in y_pred_list for tag in labels)
            )

            # Count occurrences per label
            for label in all_classes:
                true_support = sum(1 for labels in y_true_list if label in labels)
                predicted_support = sum(1 for labels in y_pred_list if label in labels)
                coverage_ratio = predicted_support / true_support if true_support > 0 else 0.0

                all_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "label": label,
                        "true_support": true_support,
                        "predicted_support": predicted_support,
                        "coverage_ratio": coverage_ratio,
                        "under_predicted": predicted_support < true_support,
                        "over_predicted": predicted_support > true_support,
                    }
                )

    return pd.DataFrame(all_rows)


def analyze_optimal_thresholds(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    artefacts_root: Path,
) -> pd.DataFrame:
    """Analyze optimal per-label thresholds using validation probabilities.

    This function loads probability files and computes optimal thresholds per label
    to maximize F1 score. Useful for understanding calibration issues.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
    artefacts_root : Path
        Root directory containing probability files (*_prob.npy).

    Returns
    -------
    pd.DataFrame
        DataFrame with optimal thresholds per label per model.
    """
    from intent_classifier.utils.label_utils import (
        binarize_labels,
        calibrate_per_label_thresholds,
    )

    all_rows = []

    for model_name, splits in predictions_dict.items():
        # Use validation set if available, otherwise test set
        split_to_use = "val" if "val" in splits else "test"
        if split_to_use not in splits:
            continue

        df = splits[split_to_use].copy()

        # Parse multilabel strings
        def parse_labels(label_str: str) -> set[str]:
            if pd.isna(label_str) or label_str == "":
                return set()
            return {tag.strip() for tag in str(label_str).split(",") if tag.strip()}

        y_true_list = [parse_labels(str(row)) for row in df["y_true"]]

        # Get all classes
        all_classes = sorted(set(tag for labels in y_true_list for tag in labels))

        if not all_classes:
            continue

        # Convert to binary (convert sets to lists)
        y_true_binary, _ = binarize_labels(
            [list(labels) for labels in y_true_list], classes=all_classes
        )

        # Try to load probabilities
        prob_file = artefacts_root / model_name / f"{split_to_use}_prob.npy"
        if not prob_file.exists():
            continue

        try:
            y_proba = np.load(prob_file)

            # Handle MultiOutputClassifier format (list of arrays)
            if isinstance(y_proba, list):
                # Convert list of (n_samples, 2) arrays to (n_samples, n_classes)
                y_proba = np.array(
                    [proba[:, 1] if proba.shape[1] > 1 else proba[:, 0] for proba in y_proba]
                ).T

            if y_proba.shape[1] != len(all_classes):
                continue

            # Calibrate thresholds
            thresholds = calibrate_per_label_thresholds(
                y_true_binary, y_proba, all_classes, metric="f1"
            )

            for label, threshold in thresholds.items():
                all_rows.append(
                    {
                        "model": model_name,
                        "split": split_to_use,
                        "label": label,
                        "optimal_threshold": threshold,
                    }
                )
        except Exception:
            # Skip if probabilities can't be loaded or processed
            continue

    return pd.DataFrame(all_rows)


def analyze_label_count_distribution(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Analyze distribution of predicted vs true label counts.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.

    Returns
    -------
    pd.DataFrame
        DataFrame with label count distribution statistics.
    """
    all_rows = []

    for model_name, splits in predictions_dict.items():
        for split_name, df in splits.items():
            df = df.copy()

            # Parse multilabel strings
            def parse_labels(label_str: str) -> set[str]:
                if pd.isna(label_str) or label_str == "":
                    return set()
                return {tag.strip() for tag in str(label_str).split(",") if tag.strip()}

            y_true_list = [parse_labels(str(row)) for row in df["y_true"]]
            y_pred_list = [parse_labels(str(row)) for row in df["y_pred"]]

            n_true = np.array([len(labels) for labels in y_true_list])
            n_pred = np.array([len(labels) for labels in y_pred_list])

            # Compute statistics
            all_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "mean_n_true": float(n_true.mean()),
                    "mean_n_pred": float(n_pred.mean()),
                    "median_n_true": float(np.median(n_true)),
                    "median_n_pred": float(np.median(n_pred)),
                    "std_n_true": float(n_true.std()),
                    "std_n_pred": float(n_pred.std()),
                    "min_n_true": int(n_true.min()),
                    "min_n_pred": int(n_pred.min()),
                    "max_n_true": int(n_true.max()),
                    "max_n_pred": int(n_pred.max()),
                    "pct_zero_pred": float((n_pred == 0).mean() * 100),
                    "pct_one_pred": float((n_pred == 1).mean() * 100),
                    "pct_two_pred": float((n_pred == 2).mean() * 100),
                    "pct_three_plus_pred": float((n_pred >= 3).mean() * 100),
                }
            )

    return pd.DataFrame(all_rows)


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
