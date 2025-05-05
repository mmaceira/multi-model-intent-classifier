"""
Visualization module.

This module provides functions for generating various evaluation visualizations
for text classification models.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc as _sk_auc,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score,
)
from sklearn.manifold import TSNE

# Configure matplotlib style
plt.style.use('default')
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

def plot_label_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
) -> pd.DataFrame:
    """Plot and compare the distribution of true vs predicted labels."""
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

def plot_precision_recall_curves(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
    is_multiclass: bool = True,
) -> float:
    """Plot precision-recall curves for binary or multiclass classification."""
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

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    model_name: str,
    out_path: Path,
) -> np.ndarray:
    """Plot and save an enhanced confusion matrix."""
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

def plot_model_comparisons(
    summary_df: pd.DataFrame,
    model_names: List[str],
    output_dir: Path,
) -> None:
    """Generate model comparison visualizations."""
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

def plot_top_misclassifications(
    texts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Plot and save top misclassifications with their text content."""
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

def plot_tsne(
    embeddings: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """Generate t-SNE visualization of embeddings colored by predictions."""
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