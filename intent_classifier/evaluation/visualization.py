"""
Visualization module.

This module provides functions for generating various evaluation visualizations
for text classification models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc as _sk_auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .utils import load_all_prediction_files

# Configure matplotlib style
plt.style.use("default")
plt.rcParams.update(
    {
        "figure.figsize": (10, 8),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "black",
        "legend.fancybox": True,
        "legend.shadow": True,
    }
)

# Configure seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


def plot_label_distribution(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Plot and compare the distribution of true vs predicted labels for each model.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    output_dir : Path
        Directory to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique labels from both train and test sets
    all_labels_set = set()
    for splits in predictions_dict.values():
        for split_name in ["train", "test"]:
            if split_name in splits:
                df = splits[split_name]
                # Filter out NaN and convert to strings
                true_labels = [str(label) for label in df["y_true"].unique() if pd.notna(label)]
                pred_labels = [str(label) for label in df["y_pred"].unique() if pd.notna(label)]
                all_labels_set.update(true_labels)
                all_labels_set.update(pred_labels)

    # Filter out NaN and ensure all labels are strings
    all_labels: list[str] = sorted([str(label) for label in all_labels_set if pd.notna(label)])

    for model_name, splits in predictions_dict.items():
        for split_name in ["train", "test"]:
            if split_name in splits:
                df = splits[split_name]

                # Compute label distributions
                true_counts = (
                    pd.Series(df["y_true"]).value_counts().reindex(all_labels, fill_value=0)
                )
                pred_counts = (
                    pd.Series(df["y_pred"]).value_counts().reindex(all_labels, fill_value=0)
                )

                df_dist = pd.DataFrame({"True": true_counts, "Predicted": pred_counts})

                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                df_dist.plot(kind="bar", ax=ax)

                # Configure plot
                ax.set_title(f"Label Distribution - {model_name} ({split_name.capitalize()} Set)")
                ax.set_ylabel("Count")
                ax.set_xlabel("Class")
                ax.legend(["True", "Predicted"])
                ax.grid(alpha=0.3)
                plt.xticks(rotation=45, ha="right")

                # Save plot
                plt.tight_layout()
                plt.savefig(output_dir / f"{model_name}_{split_name}_label_distribution.png")
                plt.close(fig)


def plot_roc_curves(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
    is_multiclass: bool = True,
) -> float:
    """Plot ROC curves for binary or multiclass classification."""
    fig, ax = plt.subplots()

    if is_multiclass:
        # For multiclass, plot one curve per class
        auc_scores = []
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_i = _sk_auc(fpr, tpr)
            auc_scores.append(auc_i)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_i:.3f})")

        # Plot micro-average and macro-average ROC curves
        all_fpr = np.unique(
            np.concatenate(
                [roc_curve(y_true_bin[:, i], y_prob[:, i])[0] for i in range(len(classes))]
            )
        )
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= len(classes)
        macro_auc = _sk_auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr, mean_tpr, "b--", label=f"Macro-average (AUC={macro_auc:.3f})", lw=2, alpha=0.8
        )

        # Compute micro-average ROC curve
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        micro_auc = _sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, "g--", label=f"Micro-average (AUC={micro_auc:.3f})", lw=2, alpha=0.8)
    else:
        # For binary classification
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        auc_score = _sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")

    # Configure plot
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(
        title=f"ROC Curves - {model_name} ({split_name})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # Save plot
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

    result = macro_auc if is_multiclass else auc_score
    return float(result)  # type: ignore[return-value]


def plot_precision_recall_curves(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Plot precision-recall curves for all models.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    output_dir : Path
        Directory to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique labels from both train and test sets
    all_labels_set = set()
    for splits in predictions_dict.values():
        for split_name in ["train", "test"]:
            if split_name in splits:
                df = splits[split_name]
                all_labels_set.update(df["y_true"].unique())

    # Filter out NaN and ensure all labels are strings
    all_labels: list[str] = sorted([str(label) for label in all_labels_set if pd.notna(label)])

    for model_name, splits in predictions_dict.items():
        for split_name in ["train", "test"]:
            if split_name in splits and "probabilities" in splits[split_name].columns:
                df = splits[split_name]

                # Get true labels and probabilities
                y_true = df["y_true"]
                y_prob = df["probabilities"]

                # Binarize the labels
                y_true_bin = label_binarize(y_true, classes=all_labels)

                # Plot precision-recall curves for each class
                fig, ax = plt.subplots(figsize=(10, 8))

                for i, label in enumerate(all_labels):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    ap_score = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    ax.plot(recall, precision, label=f"{label} (AP={ap_score:.3f})")

                # Add micro-average curve
                precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
                ap_score = average_precision_score(y_true_bin.ravel(), y_prob.ravel())
                ax.plot(
                    recall,
                    precision,
                    "k--",
                    label=f"Micro-average (AP={ap_score:.3f})",
                    lw=2,
                    alpha=0.8,
                )

                # Configure plot
                ax.set(
                    title=f"Precision-Recall Curves - {model_name} ({split_name.capitalize()} Set)",
                    xlabel="Recall",
                    ylabel="Precision",
                    xlim=[0.0, 1.0],
                    ylim=[0.0, 1.05],
                )
                ax.grid(alpha=0.3)
                ax.legend(loc="lower left")

                # Save plot
                plt.tight_layout()
                plt.savefig(output_dir / f"{model_name}_{split_name}_precision_recall.png")
                plt.close(fig)


def plot_confusion_matrix(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> None:
    """Plot confusion matrices for each model's train and test set predictions.

    Generates separate files for normalized and non-normalized versions of the confusion matrix.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary mapping model names to another dictionary containing
        'train' and 'test' DataFrames with predictions.
    output_dir : str | Path
        Directory to save the plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique labels from all train and test sets
    classes_set = set()
    for model_predictions in predictions_dict.values():
        for split_name in ["train", "test"]:
            if split_name in model_predictions:
                # Filter out NaN and convert to strings
                true_labels = [
                    str(label)
                    for label in model_predictions[split_name]["y_true"].unique()
                    if pd.notna(label)
                ]
                classes_set.update(true_labels)
    classes: list[str] = sorted(classes_set)

    for model_name, model_predictions in predictions_dict.items():
        for split_name in ["train", "test"]:
            if split_name in model_predictions:
                df = model_predictions[split_name]
                y_true = df["y_true"]
                y_pred = df["y_pred"]

                # Clean data: filter NaN and convert to strings
                mask = pd.notna(y_true) & pd.notna(y_pred)
                y_true_clean = np.array([str(v) for v in y_true[mask]])
                y_pred_clean = np.array([str(v) for v in y_pred[mask]])

                if len(y_true_clean) == 0:
                    continue  # Skip if no valid data

                # Compute confusion matrix
                cm = confusion_matrix(y_true_clean, y_pred_clean, labels=classes)

                # Plot non-normalized confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
                )
                plt.title(f"Confusion Matrix - {model_name} ({split_name.capitalize()} Set)")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(output_dir / f"{model_name}_{split_name}_confusion_matrix.png")
                plt.close()

                # Plot normalized confusion matrix
                cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=classes,
                    yticklabels=classes,
                )
                plt.title(
                    f"Normalized Confusion Matrix - {model_name} ({split_name.capitalize()} Set)"
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(
                    output_dir / f"{model_name}_{split_name}_confusion_matrix_normalized.png"
                )
                plt.close()


def plot_model_comparisons(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> None:
    """Generate comparison plots for multiple models across train and test sets.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary mapping model names to another dictionary containing
        'train' and 'test' DataFrames with predictions.
    output_dir : str | Path
        Directory to save the plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics for each model and split
    metrics = []
    for model_name, model_predictions in predictions_dict.items():
        for split_name in ["train", "test"]:
            if split_name in model_predictions:
                df = model_predictions[split_name]
                y_true = df["y_true"].values
                y_pred = df["y_pred"].values

                # Detect and handle multi-label format
                import pandas as pd

                from intent_classifier.utils.label_utils import binarize_labels

                is_multi = False
                if len(y_true) > 0:
                    sample = str(y_true[0]) if not pd.isna(y_true[0]) else ""
                    if "," in sample and not sample.startswith("["):
                        is_multi = True
                        # Convert comma-separated strings to lists (preserve
                        # all entries, use empty list for NaN/empty)
                        y_true_list: list[list[str]] = []
                        for label in y_true:
                            if pd.isna(label) or label == "":
                                y_true_list.append([])
                            else:
                                tags = [tag.strip() for tag in str(label).split(",") if tag.strip()]
                                y_true_list.append(tags if tags else [])
                        y_pred_list: list[list[str]] = []
                        for label in y_pred:
                            if pd.isna(label) or label == "":
                                y_pred_list.append([])
                            else:
                                tags = [tag.strip() for tag in str(label).split(",") if tag.strip()]
                                y_pred_list.append(tags if tags else [])
                        y_true = y_true_list
                        y_pred = y_pred_list

                if is_multi:
                    # Multi-label metrics
                    all_classes = sorted(
                        set(tag for labels in y_true for tag in labels)
                        | set(tag for labels in y_pred for tag in labels)
                    )
                    if all_classes:
                        y_true_binary, _ = binarize_labels(y_true, classes=all_classes)
                        y_pred_binary, _ = binarize_labels(y_pred, classes=all_classes)
                        from sklearn.metrics import f1_score as sk_f1_score

                        metrics.append(
                            {
                                "Model": model_name,
                                "Split": split_name.capitalize(),
                                "Accuracy": accuracy_score(
                                    y_true_binary, y_pred_binary
                                ),  # Subset accuracy
                                "Macro F1": sk_f1_score(
                                    y_true_binary, y_pred_binary, average="macro", zero_division=0
                                ),
                                "Weighted F1": sk_f1_score(
                                    y_true_binary, y_pred_binary, average="micro", zero_division=0
                                ),
                            }
                        )
                else:
                    # Single-label metrics
                    # Clean data: remove NaN values and ensure consistent types
                    mask = pd.notna(y_true) & pd.notna(y_pred)
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]

                    # Convert to strings to ensure consistent type
                    y_true_clean = np.array([str(v) for v in y_true_clean])
                    y_pred_clean = np.array([str(v) for v in y_pred_clean])

                    if len(y_true_clean) > 0:
                        metrics.append(
                            {
                                "Model": model_name,
                                "Split": split_name.capitalize(),
                                "Accuracy": accuracy_score(y_true_clean, y_pred_clean),
                                "Macro F1": f1_score(
                                    y_true_clean, y_pred_clean, average="macro", zero_division=0
                                ),
                                "Weighted F1": f1_score(
                                    y_true_clean, y_pred_clean, average="weighted", zero_division=0
                                ),
                            }
                        )
                    # Skip if no valid data

    metrics_df = pd.DataFrame(metrics)

    # Create separate plots for each metric
    for metric in ["Accuracy", "Macro F1", "Weighted F1"]:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=metrics_df, x="Model", y=metric, hue="Split")
        plt.title(f"Model Comparison - {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_{metric.lower().replace(' ', '_')}.png")
        plt.close()

    # Create combined subplot figure with train and test side by side
    metrics_list = ["Accuracy", "Macro F1", "Weighted F1"]
    n_metrics = len(metrics_list)

    fig, axes = plt.subplots(n_metrics, 2, figsize=(12, 4 * n_metrics))

    for i, metric in enumerate(metrics_list):
        # Train set
        train_df = metrics_df[metrics_df["Split"] == "Train"]
        sns.barplot(data=train_df, y="Model", x=metric, ax=axes[i, 0])
        axes[i, 0].set_title(f"{metric} (Train)")
        axes[i, 0].tick_params(axis="y", rotation=0)

        # Test set
        test_df = metrics_df[metrics_df["Split"] == "Test"]
        sns.barplot(data=test_df, y="Model", x=metric, ax=axes[i, 1])
        axes[i, 1].set_title(f"{metric} (Test)")
        axes[i, 1].tick_params(axis="y", rotation=0)

        # Set x-axis limits to be the same for train and test
        x_min = min(axes[i, 0].get_xlim()[0], axes[i, 1].get_xlim()[0])
        x_max = max(axes[i, 0].get_xlim()[1], axes[i, 1].get_xlim()[1])
        axes[i, 0].set_xlim(x_min, x_max)
        axes[i, 1].set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_metrics_combined.png")
    plt.close()


def plot_top_misclassifications(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Plot and save top misclassifications with their text content for both train and test sets.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    output_dir : Path
        Directory to save the misclassifications or direct path to output file
    top_n : int, optional
        Number of top misclassifications to save, by default 10
    """
    output_dir = Path(output_dir)

    # Handle both directory and file paths
    if output_dir.suffix == ".csv":
        # If output_dir is a file path, use it directly
        output_path = output_dir
        # Get the model name and split from the parent directory
        parent_dir = output_dir.parent
        model_name = parent_dir.parent.name
        split_name = parent_dir.name

        # Find the corresponding DataFrame
        if model_name in predictions_dict and split_name in predictions_dict[model_name]:
            df = predictions_dict[model_name][split_name]
            # Filter misclassifications
            misclass_df = df[df["y_true"] != df["y_pred"]].copy()
            # Save to CSV
            misclass_df.head(top_n).to_csv(output_path, index=False)
    else:
        # If output_dir is a directory, create it and save files for each model and split
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name, splits in predictions_dict.items():
            for split_name in ["train", "test"]:
                if split_name in splits:
                    df = splits[split_name]
                    # Filter misclassifications
                    misclass_df = df[df["y_true"] != df["y_pred"]].copy()
                    # Save to CSV
                    misclass_df.head(top_n).to_csv(
                        output_dir
                        / f"{model_name}_{split_name}_top_{top_n}_misclassifications.csv",
                        index=False,
                    )


def visualize_error_distribution(
    predictions_dict: dict[str, dict[str, pd.DataFrame]], output_dir: Path
) -> None:
    """Create visualizations of error distributions across models and classes
    for both train and test sets.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    output_dir : Path
        Directory where visualizations will be saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "test"]:
        # Combine all predictions for comparison
        model_results = []
        class_error_rates = []

        for model_name, splits in predictions_dict.items():
            if split_name in splits:
                df = splits[split_name]
                # Calculate overall accuracy
                accuracy = (df["y_true"] == df["y_pred"]).mean()
                model_results.append({"model": model_name, "accuracy": accuracy})

                # Calculate per-class error rates
                for class_name in df["y_true"].unique():
                    class_df = df[df["y_true"] == class_name]
                    error_rate = (class_df["y_true"] != class_df["y_pred"]).mean()
                    class_error_rates.append(
                        {
                            "model": model_name,
                            "class": class_name,
                            "error_rate": error_rate,
                            "count": len(class_df),
                        }
                    )

        if not model_results:
            continue

        # Plot per-class error rates
        class_df = pd.DataFrame(class_error_rates)
        plt.figure(figsize=(12, 8))
        sns.barplot(x="class", y="error_rate", hue="model", data=class_df)
        plt.title(f"Error Rate by Class and Model ({split_name.capitalize()} Set)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"error_rate_by_class_{split_name}.png", dpi=300)
        plt.close()


def generate_detailed_error_report(
    predictions_dict: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    only_split: str | None = None,
) -> None:
    """Generate an HTML report with detailed analysis of classification errors
    for both train and test sets.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    output_dir : Path
        Directory where the report will be saved
    only_split : str, optional
        Which split to analyze ('train' or 'test'), by default None (both splits)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_to_generate = [only_split] if only_split else ["train", "test"]
    for split_name in splits_to_generate:
        # Get consistently misclassified examples
        misclass_df = consistently_misclassified(predictions_dict, split_name=split_name)

        # Get common error patterns
        error_patterns_df = analyse_error_patterns(predictions_dict, split_name=split_name)

        # Create HTML report
        html = []
        html.append("<html><head><title>Detailed Error Analysis</title>")
        html.append("<style>body{font-family:Arial;max-width:1200px;margin:0 auto;padding:20px}")
        html.append("table{border-collapse:collapse;width:100%;margin-bottom:20px}")
        html.append("th,td{border:1px solid #ddd;padding:8px}")
        html.append("th{background-color:#f2f2f2;text-align:left}")
        html.append("tr:nth-child(even){background-color:#f9f9f9}")
        html.append("h1,h2,h3{color:#333}</style></head><body>")

        html.append(
            f"<h1>Detailed Classification Error Analysis ({split_name.capitalize()} Set)</h1>"
        )

        # Common error patterns
        html.append("<h2>Common Error Patterns</h2>")
        if not error_patterns_df.empty:
            html.append("<table><tr><th>Error Type</th><th>Count</th></tr>")
            for _, row in error_patterns_df.head(10).iterrows():
                html.append(f"<tr><td>{row['error_type']}</td><td>{row['total_count']}</td></tr>")
            html.append("</table>")
        else:
            html.append("<p>No error patterns found.</p>")

        # Consistently misclassified examples
        html.append("<h2>Consistently Misclassified Examples</h2>")
        if not misclass_df.empty:
            html.append(
                "<table><tr><th>Text</th><th>True Label</th>"
                "<th>Predicted Label</th><th>Models</th></tr>"
            )
            for _, row in misclass_df.head(20).iterrows():
                models = [name for name in predictions_dict if row.get(name, False)]
                html.append(f"<tr><td>{row['text']}</td><td>{row['y_true']}</td>")
                html.append(f"<td>{row['y_pred']}</td><td>{', '.join(models)}</td></tr>")
            html.append("</table>")
        else:
            html.append("<p>No consistently misclassified examples found.</p>")

        # Model-specific analyses
        html.append("<h2>Model-Specific Error Analysis</h2>")
        for model_name, splits in predictions_dict.items():
            if split_name in splits:
                df = splits[split_name]
                errors = df[df["y_true"] != df["y_pred"]]
                html.append(f"<h3>{model_name}</h3>")

                # Error count by class
                error_by_class = errors.groupby("y_true").size().reset_index(name="count")
                html.append("<h4>Error Count by True Class</h4>")
                html.append("<table><tr><th>Class</th><th>Error Count</th></tr>")
                for _, row in error_by_class.sort_values("count", ascending=False).iterrows():
                    html.append(f"<tr><td>{row['y_true']}</td><td>{row['count']}</td></tr>")
                html.append("</table>")

                # Sample errors
                html.append("<h4>Sample Errors</h4>")
                html.append(
                    "<table><tr><th>Text</th><th>True Label</th><th>Predicted Label</th></tr>"
                )
                for _, row in errors.head(5).iterrows():
                    html.append(
                        f"<tr><td>{row['text']}</td><td>{row['y_true']}</td><td>{row['y_pred']}</td></tr>"
                    )
                html.append("</table>")

        html.append("</body></html>")

        # Save the report
        with open(
            output_dir / f"detailed_error_report_{split_name}.html", "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(html))


def plot_confusion_matrices(
    experiment_dir: str | Path, figsize: tuple[int, int] = (15, 15)
) -> None:
    """Plot confusion matrices for all models in the experiment.

    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing the experiment results
    figsize : tuple
        Figure size for the confusion matrices
    """
    predictions = load_all_prediction_files(experiment_dir)

    model_names = list(predictions.keys())
    n_models = len(model_names)

    # Calculate dimensions for subplots
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Get all unique labels from all models' test sets
    all_labels_set = set()
    for splits in predictions.values():
        if "test" in splits:
            df = splits["test"]
            # Filter out NaN and convert to strings
            true_labels = [str(label) for label in df["y_true"].unique() if pd.notna(label)]
            pred_labels = [str(label) for label in df["y_pred"].unique() if pd.notna(label)]
            all_labels_set.update(true_labels)
            all_labels_set.update(pred_labels)

    all_labels: list[str] = sorted(all_labels_set)

    for i, (model_name, splits) in enumerate(predictions.items()):
        if i < len(axes):
            ax = axes[i]

            # Only analyze test set
            if "test" in splits:
                df = splits["test"]

                # Calculate confusion matrix
                cm = confusion_matrix(df["y_true"], df["y_pred"], labels=all_labels)

                # Normalize by row (true labels)
                cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)

                # Plot heatmap
                sns.heatmap(
                    cm_normalized,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=all_labels,
                    yticklabels=all_labels,
                    ax=ax,
                )

                ax.set_title(f"Confusion Matrix - {model_name} (Test Set)")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    # Remove empty subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.close()  # Close figure instead of showing to prevent pop-ups


def plot_top_error_types(df: pd.DataFrame, output_path, n: int = 10):
    """
    Plot and save the top n error types (true_label -> pred_label) as a bar chart.
    Args:
        df (pd.DataFrame): DataFrame with columns 'true_label' and 'pred_label'
            (or 'y_true'/'y_pred').
        output_path (str or Path): Path to save the PNG plot.
        n (int): Number of top error types to plot (default 10).
    """
    # Accept both naming conventions
    if (
        "y_true" in df.columns
        and "y_pred" in df.columns
        or "y_true" in df.columns
        and "y_pred" in df.columns
    ):
        y_true = df["y_true"]
        y_pred = df["y_pred"]
    else:
        raise ValueError(
            "DataFrame must contain either ('y_true', 'y_pred') or "
            "('true_label', 'pred_label') columns."
        )

    # Create error type column
    error_df = df[y_true != y_pred].copy()
    error_df["error_type"] = (
        y_true[y_true != y_pred].astype(str) + " -> " + y_pred[y_true != y_pred].astype(str)
    )

    # Get top n error types
    error_counts = error_df["error_type"].value_counts().head(n)

    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.bar(error_counts.index, error_counts.values)
    plt.title(f"Top {n} Error Types")
    plt.xlabel("Error Type (True → Predicted)")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def consistently_misclassified(
    predictions_dict: dict[str, dict[str, pd.DataFrame]], split_name: str = "test"
) -> pd.DataFrame:
    """Find examples that are consistently misclassified across models for a specific split.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    split_name : str, optional
        Which split to analyze ('train' or 'test'), by default "test"

    Returns
    -------
    pd.DataFrame
        DataFrame containing consistently misclassified examples
    """
    # Initialize DataFrame to store results
    misclass_df = None

    for model_name, splits in predictions_dict.items():
        if split_name in splits:
            df = splits[split_name]
            # Get misclassifications for this model
            model_misclass = df[df["y_true"] != df["y_pred"]].copy()

            if misclass_df is None:
                # Initialize with first model's misclassifications
                misclass_df = model_misclass.copy()
                misclass_df[model_name] = True
            else:
                # Merge with existing misclassifications
                misclass_df = misclass_df.merge(
                    model_misclass[["text", "y_true", "y_pred"]],
                    on=["text", "y_true", "y_pred"],
                    how="outer",
                )
                misclass_df[model_name] = (
                    misclass_df[model_name].fillna(False).infer_objects(copy=False)
                )

    if misclass_df is None:
        return pd.DataFrame()

    # Sort by number of models that misclassified
    misclass_df["n_models"] = misclass_df[list(predictions_dict.keys())].sum(axis=1)
    misclass_df = misclass_df.sort_values("n_models", ascending=False)

    return misclass_df


def analyse_error_patterns(
    predictions_dict: dict[str, dict[str, pd.DataFrame]], split_name: str = "test"
) -> pd.DataFrame:
    """Analyze common error patterns across models for a specific split.

    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames
    split_name : str, optional
        Which split to analyze ('train' or 'test'), by default "test"

    Returns
    -------
    pd.DataFrame
        DataFrame containing error patterns and their frequencies
    """
    error_patterns = []

    for model_name, splits in predictions_dict.items():
        if split_name in splits:
            df = splits[split_name]
            # Get misclassifications
            misclass = df[df["y_true"] != df["y_pred"]]

            # Count error patterns
            error_counts = misclass.groupby(["y_true", "y_pred"]).size().reset_index(name="count")
            error_counts["model"] = model_name
            error_patterns.append(error_counts)

    if not error_patterns:
        return pd.DataFrame()

    # Combine error patterns from all models
    error_patterns_df = pd.concat(error_patterns, ignore_index=True)

    # Create error type column
    error_patterns_df["error_type"] = (
        error_patterns_df["y_true"].astype(str) + " -> " + error_patterns_df["y_pred"].astype(str)
    )

    # Aggregate across models
    error_patterns_df = (
        error_patterns_df.groupby(["y_true", "y_pred", "error_type"])["count"].sum().reset_index()
    )
    error_patterns_df = error_patterns_df.rename(columns={"count": "total_count"})

    return error_patterns_df.sort_values("total_count", ascending=False)
