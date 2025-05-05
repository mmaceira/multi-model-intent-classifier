"""
Evaluation Module

This module provides a unified evaluation workflow for text classification models.
It reads model artifacts (predictions, probabilities) and computes comprehensive
metrics and visualizations without touching the models directly, ensuring
deterministic and reproducible evaluation.

Key Features:
- Deterministic evaluation from saved predictions
- Comprehensive metric computation
- Rich visualization generation
- Support for both binary and multiclass classification
- Training vs test performance comparison
- Overfitting analysis

Functions:
- run_evaluations: Main function for running the evaluation workflow
- _plot_label_distribution: Generate label distribution plots
- _plot_roc_curves: Generate ROC curve plots
- _plot_precision_recall_curves: Generate precision-recall curve plots
- _plot_confusion_matrix: Generate confusion matrix plots
- _ensure_dir: Utility for directory creation

Metrics Generated:
- Accuracy
- Macro and weighted F1 scores
- ROC AUC (per class and average)
- Average precision
- Confusion matrices
- Label distributions
- Training vs test performance gaps

Visualizations:
- Label distribution plots
- ROC curves (per class and average)
- Precision-recall curves
- Enhanced confusion matrices
- Model comparison plots

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pathlib
- logging

Example Usage:
    >>> # Run evaluation for multiple models
    >>> results = run_evaluations(
    ...     model_names=['svm', 'naive_bayes'],
    ...     y_true=y_test,
    ...     y_train_true=y_train,
    ...     artefacts_root='artifacts',
    ...     output_dir='results'
    ... )
    
    >>> # Access evaluation metrics
    >>> print(results['svm']['test_accuracy'])
    >>> print(results['naive_bayes']['test_macro_f1'])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, List, Dict, Any, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc as _sk_auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s',
    force=True  # Force reconfiguration of the root logger
)
_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)  # Set logger to INFO level

# Configure matplotlib style
plt.style.use('default')  # Use default style as base
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'black',
    'legend.fancybox': True,
    'legend.shadow': True,
})

# Configure seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

def _ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : str | Path
        Path to the directory to ensure exists.
        
    Returns
    -------
    Path
        The path to the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _plot_label_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
) -> pd.DataFrame:
    """Plot and compare the distribution of true vs predicted labels.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    classes : np.ndarray
        Unique class labels.
    split_name : str
        Name of the data split (e.g., 'Train', 'Test').
    model_name : str
        Name of the model being evaluated.
    out_path : Path
        Path to save the plot.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the label distribution counts.
    """
    # Compute label distributions
    true_counts = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)
    
    df_dist = pd.DataFrame({
        'True': true_counts,
        'Predicted': pred_counts
    })
    
    # Create plot
    fig, ax = plt.subplots()
    df_dist.plot(kind='bar', ax=ax)
    
    # Configure plot
    ax.set_title(f'Label Distribution - {model_name} ({split_name})')
    ax.set_ylabel('Count')
    ax.set_xlabel('Class')
    ax.legend(['True', 'Predicted'])
    ax.grid(alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    
    return df_dist

def _plot_roc_curves(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
    is_multiclass: bool = True,
) -> float:
    """Plot ROC curves for binary or multiclass classification.
    
    Parameters
    ----------
    y_true_bin : np.ndarray
        Binarized true labels.
    y_prob : np.ndarray
        Predicted probabilities.
    classes : np.ndarray
        Unique class labels.
    split_name : str
        Name of the data split (e.g., 'Train', 'Test').
    model_name : str
        Name of the model being evaluated.
    out_path : Path
        Path to save the plot.
    is_multiclass : bool, optional
        Whether the task is multiclass classification, by default True.
        
    Returns
    -------
    float
        The AUC score (macro-average for multiclass, single score for binary).
    """
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
        all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_prob[:, i])[0] 
                                          for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= len(classes)
        macro_auc = _sk_auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, 'b--', 
                label=f'Macro-average (AUC={macro_auc:.3f})', 
                lw=2, alpha=0.8)
        
        # Compute micro-average ROC curve
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        micro_auc = _sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, 'g--', 
                label=f'Micro-average (AUC={micro_auc:.3f})', 
                lw=2, alpha=0.8)
    else:
        # For binary classification
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        auc_score = _sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    
    # Configure plot
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
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
    
    return macro_auc if is_multiclass else auc_score

def _plot_precision_recall_curves(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
    is_multiclass: bool = True,
) -> float:
    """Plot precision-recall curves for binary or multiclass classification.
    
    Parameters
    ----------
    y_true_bin : np.ndarray
        Binarized true labels.
    y_prob : np.ndarray
        Predicted probabilities.
    classes : np.ndarray
        Unique class labels.
    split_name : str
        Name of the data split (e.g., 'Train', 'Test').
    model_name : str
        Name of the model being evaluated.
    out_path : Path
        Path to save the plot.
    is_multiclass : bool, optional
        Whether the task is multiclass classification, by default True.
        
    Returns
    -------
    float
        The average precision score (mean for multiclass, single score for binary).
    """
    fig, ax = plt.subplots()
    
    if is_multiclass:
        # For multiclass, plot one curve per class
        ap_scores = []
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            ap_scores.append(ap)
            ax.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
        
        # Plot micro-average precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        ap_micro = average_precision_score(y_true_bin.ravel(), y_prob.ravel())
        ax.plot(recall, precision, 'b--', 
                label=f'Micro-average (AP={ap_micro:.3f})', 
                lw=2, alpha=0.8)
    else:
        # For binary classification
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
        ap = average_precision_score(y_true_bin, y_prob)
        ax.plot(recall, precision, label=f"AP={ap:.3f}")
    
    # Configure plot
    ax.set(
        title=f"Precision-Recall Curves - {model_name} ({split_name})",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    
    return np.mean(ap_scores) if is_multiclass else ap

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
) -> np.ndarray:
    """Plot and save an enhanced confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    classes : np.ndarray
        Unique class labels.
    split_name : str
        Name of the data split (e.g., 'Train', 'Test').
    model_name : str
        Name of the model being evaluated.
    out_path : Path
        Path to save the plot.
        
    Returns
    -------
    np.ndarray
        The confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with zero
    
    # Plot absolute confusion matrix
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=classes, yticklabels=classes)
    ax1.set_title(f'Confusion Matrix - {model_name} ({split_name})')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig1)
    
    # Plot normalized confusion matrix
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=classes, yticklabels=classes)
    ax2.set_title(f'Normalized Confusion Matrix - {model_name} ({split_name})')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    plt.tight_layout()
    
    # Save normalized matrix
    normalized_path = out_path.parent / f"{out_path.stem}_normalized{out_path.suffix}"
    plt.savefig(normalized_path)
    plt.close(fig2)
    
    return cm

def _plot_model_comparisons(
    summary_df: pd.DataFrame,
    model_names: List[str],
    output_dir: Path,
) -> None:
    """Generate model comparison visualizations.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame containing model metrics.
    model_names : List[str]
        List of model names to compare.
    output_dir : Path
        Directory to save the visualizations.
    """
    # Bar chart comparing key metrics across models
    plt.figure(figsize=(12, 8))
    key_metrics = [col for col in summary_df.columns 
                  if not col.endswith('_best') 
                  and not col.endswith('_diff') 
                  and col.startswith('test_')]
    
    summary_df[key_metrics].plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison (Test)')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(output_dir / "models_comparison.png")
    plt.close()
    
    # Generate train vs test comparison if training metrics available
    train_metrics = [col for col in summary_df.columns 
                    if col.startswith('train_') 
                    and not col.endswith('_best')]
    if train_metrics:
        # Create a figure for each metric comparing train vs test
        common_metrics = [m.replace('test_', '') for m in key_metrics 
                        if m.replace('test_', '') in [t.replace('train_', '') 
                                                     for t in train_metrics]]
        
        for metric in common_metrics:
            plt.figure(figsize=(10, 6))
            
            # Create comparison dataframe
            compare_df = pd.DataFrame(index=model_names)
            train_col = [col for col in train_metrics 
                       if col.replace('train_', '') == metric][0]
            test_col = [col for col in key_metrics 
                      if col.replace('test_', '') == metric][0]
            
            compare_df['Train'] = summary_df.loc[model_names, train_col]
            compare_df['Test'] = summary_df.loc[model_names, test_col]
            
            # Plot comparison
            compare_df.plot(kind='bar', figsize=(10, 6))
            plt.title(f'{metric.title()} Comparison: Train vs Test')
            plt.xlabel('Model')
            plt.ylabel(metric.title())
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}_train_test_comparison.png")
            plt.close()

def _plot_top_misclassifications(
    texts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Plot and save top misclassifications with their text content.
    
    Parameters
    ----------
    texts : np.ndarray
        Array of text samples.
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    output_path : Path
        Path to save the results.
    top_n : int, optional
        Number of top misclassifications to show, by default 10.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the top misclassifications.
    """
    # Create DataFrame with misclassifications
    df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    # Filter misclassifications
    df = df[df['true_label'] != df['predicted_label']]
    
    # Save to CSV
    df.head(top_n).to_csv(output_path, index=False)
    
    return df

def _plot_tsne(
    embeddings: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """Generate t-SNE visualization of embeddings colored by predictions.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Array of embeddings.
    y_pred : np.ndarray
        Predicted labels.
    output_path : Path
        Path to save the plot.
    perplexity : int, optional
        t-SNE perplexity parameter, by default 30.
    n_iter : int, optional
        Number of iterations for t-SNE, by default 1000.
    """
    from sklearn.manifold import TSNE
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_evaluations(
    model_names: List[str],
    *,
    y_true: Sequence[Any],
    y_train_true: Optional[Sequence[Any]] = None,
    artefacts_root: str | Path = "artefacts",
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics from persisted predictions.
    
    This function orchestrates the evaluation workflow by:
    1. Reading model predictions and probabilities
    2. Computing various metrics (accuracy, F1, ROC AUC, etc.)
    3. Generating visualizations (confusion matrices, ROC curves, etc.)
    4. Comparing training vs test performance
    5. Computing overfitting metrics
    6. Additional analyses (misclassifications, t-SNE, calibration) when data is available
    
    Parameters
    ----------
    model_names : List[str]
        List of model identifiers (must correspond to sub-folders inside
        *artefacts_root*).
    y_true : Sequence[Any]
        Ground-truth labels for the test set – must be in the same order
        that was used when generating the predictions.
    y_train_true : Optional[Sequence[Any]], optional
        Ground-truth labels for the training set – if provided,
        training metrics will also be computed, by default None.
    artefacts_root : str | Path, optional
        Directory containing model artifacts, by default "artefacts".
    output_dir : str | Path, optional
        Where to write evaluation artifacts (reports, matrices, ROC curves),
        by default "results".
    verbose : bool, optional
        Whether to log progress, by default True.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their evaluation metrics.
        Each model's metrics include:
        - test_accuracy: Test set accuracy
        - test_macro_f1: Test set macro-averaged F1 score
        - test_weighted_f1: Test set weighted F1 score
        - test_roc_auc: Test set ROC AUC score (if probabilities available)
        - test_avg_precision: Test set average precision (if probabilities available)
        - train_*: Corresponding training set metrics (if available)
        - accuracy_diff: Training accuracy - test accuracy
        - macro_f1_diff: Training macro F1 - test macro F1
        - roc_auc_diff: Training ROC AUC - test ROC AUC (if probabilities available)
        - Additional metrics from extra analyses when data is available
        
    Raises
    ------
    FileNotFoundError
        If required prediction files are not found.
    """
    _LOG.info("Starting evaluation process")
    _LOG.info(f"Output directory for plots and results: {output_dir}")
    
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    is_multiclass = len(classes) > 2
    
    # Prepare binarized versions of labels for ROC and PR curves
    y_true_bin = label_binarize(y_true, classes=classes) if is_multiclass else y_true
    
    if y_train_true is not None:
        y_train_true = np.asarray(y_train_true)
        y_train_true_bin = label_binarize(y_train_true, classes=classes) if is_multiclass else y_train_true

    # Initialize results dictionary and ensure output directory exists
    results: Dict[str, Dict[str, Any]] = {}
    artefacts_root = Path(artefacts_root)
    output_dir = _ensure_dir(output_dir)

    # Process each model
    for name in model_names:
        model_dir = artefacts_root / name
        model_out_dir = _ensure_dir(output_dir / name)
        train_out_dir = _ensure_dir(model_out_dir / "train")
        test_out_dir = _ensure_dir(model_out_dir / "test")
        _LOG.info(f"Output directories for model {name}:")
        _LOG.info(f"  - Training results: {train_out_dir}")
        _LOG.info(f"  - Test results: {test_out_dir}")

        # Dictionary to store all results for this model
        model_results = {}

        # Define splits to process
        splits = [
            {
                "name": "test",
                "y_true": y_true,
                "y_true_bin": y_true_bin,
                "pred_file": "test_predictions.csv",
                "prob_file": "test_prob.npy",
                "prefix": "test_",
                "out_dir": test_out_dir
            }
        ]
        
        if y_train_true is not None:
            splits.append({
                "name": "train",
                "y_true": y_train_true,
                "y_true_bin": y_train_true_bin,
                "pred_file": "train_predictions.csv",
                "prob_file": "train_prob.npy",
                "prefix": "train_",
                "out_dir": train_out_dir
            })

        # Process each split
        for split in splits:
            # Load predictions
            pred_path = model_dir / split["pred_file"]
            if not pred_path.exists():
                if split["name"] == "test":
                    raise FileNotFoundError(f"Predictions not found at {pred_path}")
                continue

            df_pred = pd.read_csv(pred_path)
            y_pred = df_pred["y_pred"].values

            # Compute metrics
            acc = accuracy_score(split["y_true"], y_pred)
            macro_f1 = f1_score(split["y_true"], y_pred, average="macro")
            weighted_f1 = f1_score(split["y_true"], y_pred, average="weighted")
            
            # Save classification report
            report_dict = classification_report(split["y_true"], y_pred, output_dict=True)
            pd.DataFrame(report_dict).T.to_csv(split["out_dir"] / f"{split['name']}_report.csv")
            
            # Generate visualizations
            _LOG.info(f"Processing distribution analysis for {name} on {split['name']} set")
            dist_df = _plot_label_distribution(
                split["y_true"], 
                y_pred, 
                classes,
                split["name"].title(),
                name,
                split["out_dir"] / f"{split['name']}_label_distribution.png"
            )
            dist_df.to_csv(split["out_dir"] / f"{split['name']}_label_distribution.csv")
            
            _LOG.info(f"Processing confusion matrix for {name} on {split['name']} set")
            cm = _plot_confusion_matrix(
                split["y_true"],
                y_pred,
                classes,
                split["name"].title(),
                name,
                split["out_dir"] / f"{split['name']}_confusion.png"
            )
            
            # Store metrics
            model_results.update({
                f"{split['prefix']}accuracy": acc,
                f"{split['prefix']}macro_f1": macro_f1,
                f"{split['prefix']}weighted_f1": weighted_f1,
            })
            
            # Process probabilities if available
            prob_path = model_dir / split["prob_file"]
            if prob_path.exists():
                _LOG.info(f"Loading probabilities from {prob_path}")
                y_prob = np.load(prob_path)
                
                # Ensure probabilities are properly normalized
                if not np.allclose(y_prob.sum(axis=1), 1.0):
                    _LOG.warning("Probabilities do not sum to 1, normalizing...")
                    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
                
                # Generate ROC curves
                _LOG.info(f"Processing ROC curves for {name} on {split['name']} set")
                roc_auc = _plot_roc_curves(
                    split["y_true_bin"],
                    y_prob,
                    classes,
                    split["name"].title(),
                    name,
                    split["out_dir"] / f"{split['name']}_roc_curves.png",
                    is_multiclass
                )
                model_results[f"{split['prefix']}roc_auc"] = roc_auc
                
                # Generate precision-recall curves
                _LOG.info(f"Processing precision-recall curves for {name} on {split['name']} set")
                avg_precision = _plot_precision_recall_curves(
                    split["y_true_bin"],
                    y_prob,
                    classes,
                    split["name"].title(),
                    name,
                    split["out_dir"] / f"{split['name']}_pr_curves.png",
                    is_multiclass
                )
                model_results[f"{split['prefix']}avg_precision"] = avg_precision
            
            # Additional analyses for test set
            if split["name"] == "test":
                # Top misclassifications if text data is available
                if "text" in df_pred.columns:
                    _LOG.info(f"Processing top misclassifications for {name}")
                    misclass_df = _plot_top_misclassifications(
                        df_pred["text"].values,
                        split["y_true"],
                        y_pred,
                        split["out_dir"] / "top_misclassifications.csv"
                    )
                    model_results["top_misclassifications"] = str(split["out_dir"] / "top_misclassifications.csv")
                
                # t-SNE visualization if embeddings are available
                if "embeddings" in df_pred.columns:
                    _LOG.info(f"Processing t-SNE visualization for {name}")
                    _plot_tsne(
                        df_pred["embeddings"].values,
                        y_pred,
                        split["out_dir"] / "tsne_visualization.png"
                    )
                    model_results["tsne_visualization"] = str(split["out_dir"] / "tsne_visualization.png")

        # Compute overfitting metrics if both train and test metrics are available
        if "train_accuracy" in model_results and "test_accuracy" in model_results:
            model_results["accuracy_diff"] = model_results["train_accuracy"] - model_results["test_accuracy"]
            model_results["macro_f1_diff"] = model_results["train_macro_f1"] - model_results["test_macro_f1"]
            if "train_roc_auc" in model_results and "test_roc_auc" in model_results:
                model_results["roc_auc_diff"] = model_results["train_roc_auc"] - model_results["test_roc_auc"]
        
        # Store model results
        results[name] = model_results

    # Create a comprehensive summary table
    summary_df = pd.DataFrame(results).T
    
    # Add "best" indicators
    for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_roc_auc"]:
        if metric in summary_df.columns:
            best_idx = summary_df[metric].idxmax()
            summary_df[f"{metric}_best"] = False
            summary_df.loc[best_idx, f"{metric}_best"] = True
    
    # Save summary metrics
    summary_df.to_csv(output_dir / "summary_metrics.csv")
    
    # Generate model comparison visualizations if multiple models
    if len(model_names) > 1:
        _plot_model_comparisons(summary_df, model_names, output_dir)

    return results
