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
from typing import Sequence, List, Dict, Any, Literal, Optional

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

_LOG = logging.getLogger(__name__)

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _plot_label_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    out_path: Path,
):
    """Plot and compare the distribution of true vs predicted labels."""
    true_counts = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)
    
    df_dist = pd.DataFrame({
        'True': true_counts,
        'Predicted': pred_counts
    })
    
    # Calculate percentages for better comparison
    df_pct = df_dist.copy()
    df_pct['True'] = (df_pct['True'] / df_pct['True'].sum()) * 100
    df_pct['Predicted'] = (df_pct['Predicted'] / df_pct['Predicted'].sum()) * 100
    
    # Create plots: one for counts, one for percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute counts
    df_dist.plot(kind='bar', ax=ax1)
    ax1.set_title(f'Label Distribution - {split_name} (Counts)')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Class')
    ax1.legend(['True', 'Predicted'])
    
    # Percentage distribution
    df_pct.plot(kind='bar', ax=ax2)
    ax2.set_title(f'Label Distribution - {split_name} (Percentage)')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('Class')
    ax2.legend(['True', 'Predicted'])
    
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
):
    """Plot ROC curves for binary or multiclass classification."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if is_multiclass:
        # For multiclass, plot one curve per class
        auc_scores = []
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_i = _sk_auc(fpr, tpr)
            auc_scores.append(auc_i)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_i:.3f})")
        
        # Also plot micro-average and macro-average ROC curves
        all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_prob[:, i])[0] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= len(classes)
        macro_auc = _sk_auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, 'b--', 
                label=f'Macro-average (AUC={macro_auc:.3f})', 
                lw=2, alpha=0.8)
        
        # Compute micro-average ROC curve and ROC area
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
    
    # Common plot elements
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set(
        title=f"ROC Curves - {model_name} ({split_name})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path)
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
):
    """Plot precision-recall curves for binary or multiclass classification."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if is_multiclass:
        # For multiclass, plot one curve per class
        ap_scores = []
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            ap_scores.append(ap)
            ax.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
        
        # Also plot micro-average precision-recall curve
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
    
    # Common plot elements
    ax.set(
        title=f"Precision-Recall Curves - {model_name} ({split_name})",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    
    return np.mean(ap_scores) if is_multiclass else ap

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
):
    """Plot and save an enhanced confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with zero
    
    # Create figure with two subplots - absolute and normalized
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=classes, yticklabels=classes)
    ax1.set_title(f'Confusion Matrix - {model_name} ({split_name})')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Plot normalized values (%)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=classes, yticklabels=classes)
    ax2.set_title(f'Normalized Confusion Matrix - {model_name} ({split_name})')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    
    return cm

def run_evaluations(
    model_names: List[str],
    *,
    y_true: Sequence[Any],
    y_train_true: Optional[Sequence[Any]] = None,
    artefacts_root: str | Path = "artefacts",
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics from persisted *predictions*.

    Parameters
    ----------
    model_names
        List of model identifiers (must correspond to sub‑folders inside
        *artefacts_root*).
    y_true
        Ground‑truth labels for the test set – **must** be in the same order
        that was used when generating the predictions.
    y_train_true
        Optional ground-truth labels for the training set – if provided,
        training metrics will also be computed.
    artefacts_root
        Directory produced by `run_training`.
    output_dir
        Where to write evaluation artefacts (reports, matrices, ROC curves).
    verbose
        Whether to log progress.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        ``name → {metric → value}``
    """
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    is_multiclass = len(classes) > 2
    
    # Prepare binarized versions of labels for ROC and PR curves
    y_true_bin = label_binarize(y_true, classes=classes) if is_multiclass else y_true
    
    if y_train_true is not None:
        y_train_true = np.asarray(y_train_true)
        y_train_true_bin = label_binarize(y_train_true, classes=classes) if is_multiclass else y_train_true

    results: Dict[str, Dict[str, Any]] = {}
    artefacts_root = Path(artefacts_root)
    output_dir = _ensure_dir(output_dir)

    for name in model_names:
        if verbose:
            print(f"[run_evaluations] Processing {name}...", flush=True)

        model_dir = artefacts_root / name
        out_dir = _ensure_dir(output_dir / name)
        
        # Dictionary to store all results for this model
        model_results = {}

        # Process test predictions
        pred_path = model_dir / "test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)

        df_pred = pd.read_csv(pred_path)
        y_pred = df_pred["y_pred"].values

        # --- TEST METRICS --------------------------------------------------
        # Scalar metrics
        test_acc = accuracy_score(y_true, y_pred)
        test_macro_f1 = f1_score(y_true, y_pred, average="macro")
        test_weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        
        # Class-level metrics
        test_report_dict = classification_report(y_true, y_pred, output_dict=True)
        pd.DataFrame(test_report_dict).T.to_csv(out_dir / "test_report.csv")
        
        # Distribution analysis
        test_dist_df = _plot_label_distribution(
            y_true, 
            y_pred, 
            classes,
            "Test",
            out_dir / "test_label_distribution.png"
        )
        test_dist_df.to_csv(out_dir / "test_label_distribution.csv")
        
        # Enhanced confusion matrix
        test_cm = _plot_confusion_matrix(
            y_true,
            y_pred,
            classes,
            "Test",
            name,
            out_dir / "test_confusion.png"
        )
        
        # Add test metrics to results
        model_results.update({
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
        })
        
        # --- ROC CURVES (TEST) ---------------------------------------------
        prob_path = model_dir / "test_prob.npy"
        if prob_path.exists():
            y_prob = np.load(prob_path)
            
            # ROC Curves
            roc_auc = _plot_roc_curves(
                y_true_bin,
                y_prob,
                classes,
                "Test",
                name,
                out_dir / "test_roc_curves.png",
                is_multiclass
            )
            model_results["test_roc_auc"] = roc_auc
            
            # Precision-Recall Curves
            avg_precision = _plot_precision_recall_curves(
                y_true_bin,
                y_prob,
                classes,
                "Test",
                name,
                out_dir / "test_pr_curves.png",
                is_multiclass
            )
            model_results["test_avg_precision"] = avg_precision
        
        # --- TRAINING METRICS (if available) ------------------------------
        train_pred_path = model_dir / "train_predictions.csv"
        if train_pred_path.exists() and y_train_true is not None:
            if verbose:
                print(f"[run_evaluations] Processing {name} training metrics...", flush=True)
                
            df_train_pred = pd.read_csv(train_pred_path)
            y_train_pred = df_train_pred["y_pred"].values
            
            # Scalar metrics
            train_acc = accuracy_score(y_train_true, y_train_pred)
            train_macro_f1 = f1_score(y_train_true, y_train_pred, average="macro")
            train_weighted_f1 = f1_score(y_train_true, y_train_pred, average="weighted")
            
            # Class-level metrics
            train_report_dict = classification_report(y_train_true, y_train_pred, output_dict=True)
            pd.DataFrame(train_report_dict).T.to_csv(out_dir / "train_report.csv")
            
            # Distribution analysis
            train_dist_df = _plot_label_distribution(
                y_train_true, 
                y_train_pred, 
                classes,
                "Train",
                out_dir / "train_label_distribution.png"
            )
            train_dist_df.to_csv(out_dir / "train_label_distribution.csv")
            
            # Enhanced confusion matrix
            train_cm = _plot_confusion_matrix(
                y_train_true,
                y_train_pred,
                classes,
                "Train",
                name,
                out_dir / "train_confusion.png"
            )
            
            # Add train metrics to results
            model_results.update({
                "train_accuracy": train_acc,
                "train_macro_f1": train_macro_f1,
                "train_weighted_f1": train_weighted_f1,
            })
            
            # Compute overfitting metrics
            model_results["accuracy_diff"] = train_acc - test_acc
            model_results["macro_f1_diff"] = train_macro_f1 - test_macro_f1
            
            # ROC Curves for training data if probabilities are available
            train_prob_path = model_dir / "train_prob.npy"
            if train_prob_path.exists():
                y_train_prob = np.load(train_prob_path)
                
                # ROC Curves
                train_roc_auc = _plot_roc_curves(
                    y_train_true_bin,
                    y_train_prob,
                    classes,
                    "Train",
                    name,
                    out_dir / "train_roc_curves.png",
                    is_multiclass
                )
                model_results["train_roc_auc"] = train_roc_auc
                
                # Precision-Recall Curves
                train_avg_precision = _plot_precision_recall_curves(
                    y_train_true_bin,
                    y_train_prob,
                    classes,
                    "Train",
                    name,
                    out_dir / "train_pr_curves.png",
                    is_multiclass
                )
                model_results["train_avg_precision"] = train_avg_precision
                
                # Compute overfitting in AUC
                if "test_roc_auc" in model_results:
                    model_results["roc_auc_diff"] = train_roc_auc - model_results["test_roc_auc"]
        
        # Save model metrics to the results dictionary
        results[name] = model_results

    # Create a comprehensive summary table
    summary_df = pd.DataFrame(results).T
    
    # Add "best" indicators
    for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_roc_auc"]:
        if metric in summary_df.columns:
            best_idx = summary_df[metric].idxmax()
            summary_df[f"{metric}_best"] = False
            summary_df.loc[best_idx, f"{metric}_best"] = True
    
    # Persist global summary
    summary_df.to_csv(output_dir / "summary_metrics.csv")
    
    # Create summary visualizations if multiple models
    if len(model_names) > 1:
        # Bar chart comparing key metrics across models
        plt.figure(figsize=(12, 8))
        key_metrics = [col for col in summary_df.columns if not col.endswith('_best') 
                       and not col.endswith('_diff') and col.startswith('test_')]
        
        summary_df[key_metrics].plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance Comparison (Test)')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.tight_layout()
        plt.savefig(output_dir / "models_comparison.png")
        plt.close()
        
        # If we have training metrics, create train vs test comparison
        train_metrics = [col for col in summary_df.columns if col.startswith('train_') 
                         and not col.endswith('_best')]
        if train_metrics:
            # Rename columns to remove the prefix for cleaner plotting
            plot_df = summary_df.copy()
            test_columns = {}
            train_columns = {}
            for col in key_metrics:
                if col in plot_df.columns:
                    test_columns[col] = col.replace('test_', '')
            for col in train_metrics:
                if col in plot_df.columns:
                    train_columns[col] = col.replace('train_', '')
            
            # Create a figure for each metric comparing train vs test
            common_metrics = [m.replace('test_', '') for m in key_metrics 
                             if m.replace('test_', '') in [t.replace('train_', '') for t in train_metrics]]
            
            for metric in common_metrics:
                plt.figure(figsize=(10, 6))
                
                # Create a dataframe with 'Train' and 'Test' columns for this metric
                compare_df = pd.DataFrame(index=model_names)
                # Find the original column names
                train_col = [col for col in train_metrics if col.replace('train_', '') == metric][0]
                test_col = [col for col in key_metrics if col.replace('test_', '') == metric][0]
                
                compare_df['Train'] = summary_df.loc[model_names, train_col]
                compare_df['Test'] = summary_df.loc[model_names, test_col]
                
                # Plot the dataframe
                compare_df.plot(kind='bar', figsize=(10, 6))
                plt.title(f'{metric.title()} Comparison: Train vs Test')
                plt.xlabel('Model')
                plt.ylabel(metric.title())
                plt.tight_layout()
                plt.savefig(output_dir / f"{metric}_train_test_comparison.png")
                plt.close()

    return results
