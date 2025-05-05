"""
Visualization module.

This module provides functions for generating various evaluation visualizations
for text classification models.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple

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
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from .metrics import analyze_text_features
from .utils import consistently_misclassified, analyse_error_patterns, load_all_prediction_files

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
    predictions_dict: Dict[str, Dict[str, pd.DataFrame]],
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
    
    # Get all unique labels from test sets
    all_labels = set()
    for splits in predictions_dict.values():
        if 'test' in splits:
            df = splits['test']
            all_labels.update(df['y_true'].unique())
            all_labels.update(df['y_pred'].unique())
    
    all_labels = sorted(all_labels)
    
    for model_name, splits in predictions_dict.items():
        if 'test' in splits:
            df = splits['test']
            
            # Compute label distributions
            true_counts = pd.Series(df['y_true']).value_counts().reindex(all_labels, fill_value=0)
            pred_counts = pd.Series(df['y_pred']).value_counts().reindex(all_labels, fill_value=0)
            
            df_dist = pd.DataFrame({
                'True': true_counts,
                'Predicted': pred_counts
            })
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            df_dist.plot(kind='bar', ax=ax)
            
            # Configure plot
            ax.set_title(f'Label Distribution - {model_name} (Test Set)')
            ax.set_ylabel('Count')
            ax.set_xlabel('Class')
            ax.legend(['True', 'Predicted'])
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name}_label_distribution.png')
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
    predictions_dict: Dict[str, Dict[str, pd.DataFrame]],
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
    
    # Get all unique labels from test sets
    all_labels = set()
    for splits in predictions_dict.values():
        if 'test' in splits:
            df = splits['test']
            all_labels.update(df['y_true'].unique())
    
    all_labels = sorted(all_labels)
    
    for model_name, splits in predictions_dict.items():
        if 'test' in splits and 'probabilities' in splits['test'].columns:
            df = splits['test']
            
            # Convert probabilities string to numpy array
            y_prob = np.array([eval(p) for p in df['probabilities']])
            
            # Create binary labels for each class
            y_true_bin = label_binarize(df['y_true'], classes=all_labels)
            
            # Plot curves
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # For each class, plot its curve
            ap_scores = []
            for i, cls in enumerate(all_labels):
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
            
            # Configure plot
            ax.set(
                title=f"Precision-Recall Curves - {model_name} (Test Set)",
                xlabel="Recall",
                ylabel="Precision",
            )
            ax.grid(alpha=0.3)
            ax.legend(loc="lower left")
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name}_precision_recall.png')
            plt.close(fig)

def plot_confusion_matrix(
    predictions_dict: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> None:
    """Plot confusion matrices for each model's test set predictions.
    
    Generates both normalized and non-normalized versions of the confusion matrix.
    
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
    
    # Get unique labels from all test sets
    classes = set()
    for model_predictions in predictions_dict.values():
        if 'test' in model_predictions:
            classes.update(model_predictions['test']['y_true'].unique())
    classes = sorted(classes)
    
    # Process each model
    for model_name, model_predictions in predictions_dict.items():
        if 'test' not in model_predictions:
            continue
            
        df = model_predictions['test']
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Create non-normalized confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            square=True
        )
        
        plt.title(f'Confusion Matrix - {model_name} (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save non-normalized plot
        plt.savefig(output_dir / f"{model_name}_confusion_matrix.png")
        plt.close()
        
        # Create normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            square=True
        )
        
        plt.title(f'Normalized Confusion Matrix - {model_name} (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save normalized plot
        plt.savefig(output_dir / f"{model_name}_confusion_matrix_normalized.png")
        plt.close()

def plot_model_comparisons(
    predictions_dict: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> None:
    """Generate comparison plots for multiple models.
    
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
    
    # Extract metrics for each model
    metrics = []
    for model_name, model_predictions in predictions_dict.items():
        if 'test' not in model_predictions:
            continue
            
        df = model_predictions['test']
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        metrics.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Macro F1': f1_score(y_true, y_pred, average='macro'),
            'Weighted F1': f1_score(y_true, y_pred, average='weighted')
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create bar plots for each metric
    for metric in ['Accuracy', 'Macro F1', 'Weighted F1']:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Model', y=metric)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_{metric.lower().replace(' ', '_')}.png")
        plt.close()

def plot_top_misclassifications(
    predictions_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Plot and save top misclassifications with their text content.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predictions and text data with 'text', 'y_true', and 'y_pred' columns
    output_path : Path
        Path to save the misclassifications CSV
    top_n : int, optional
        Number of top misclassifications to save, by default 10
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the top misclassifications
    """
    # Filter misclassifications
    df = predictions_df[predictions_df['y_true'] != predictions_df['y_pred']].copy()
    
    # Save to CSV
    df.head(top_n).to_csv(output_path, index=False)
    
    return df

def visualize_error_distribution(predictions_dict: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path):
    """Create visualizations of error distributions across models and classes.
    
    Saves visualizations to the output directory.
    
    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
    output_dir : Path
        Directory where visualizations will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all predictions for comparison
    model_results = []
    class_error_rates = []
    
    for model_name, splits in predictions_dict.items():
        # We'll only analyze test set for visualization
        if 'test' in splits:
            df = splits['test']
            # Calculate overall accuracy
            accuracy = (df['y_true'] == df['y_pred']).mean()
            model_results.append({'model': model_name, 'accuracy': accuracy})
            
            # Calculate per-class error rates
            for class_name in df['y_true'].unique():
                class_df = df[df['y_true'] == class_name]
                error_rate = (class_df['y_true'] != class_df['y_pred']).mean()
                class_error_rates.append({
                    'model': model_name,
                    'class': class_name,
                    'error_rate': error_rate,
                    'count': len(class_df)
                })
    
    if not model_results:
        return
    
    # Plot per-class error rates
    class_df = pd.DataFrame(class_error_rates)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='class', y='error_rate', hue='model', data=class_df)
    plt.title('Error Rate by Class and Model (Test Set)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rate_by_class.png', dpi=300)
    plt.close()

def generate_detailed_error_report(predictions_dict: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path):
    """Generate an HTML report with detailed analysis of classification errors.
    
    Includes example text snippets and patterns.
    
    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
    output_dir : Path
        Directory where the report will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get consistently misclassified examples
    misclass_df = consistently_misclassified(predictions_dict)
    
    # Get common error patterns
    error_patterns_df = analyse_error_patterns(predictions_dict)
    
    # Create HTML report
    html = []
    html.append('<html><head><title>Detailed Error Analysis</title>')
    html.append('<style>body{font-family:Arial;max-width:1200px;margin:0 auto;padding:20px}')
    html.append('table{border-collapse:collapse;width:100%;margin-bottom:20px}')
    html.append('th,td{border:1px solid #ddd;padding:8px}')
    html.append('th{background-color:#f2f2f2;text-align:left}')
    html.append('tr:nth-child(even){background-color:#f9f9f9}')
    html.append('h1,h2,h3{color:#333}</style></head><body>')
    
    html.append('<h1>Detailed Classification Error Analysis</h1>')
    
    # Common error patterns
    html.append('<h2>Common Error Patterns</h2>')
    if not error_patterns_df.empty:
        html.append('<table><tr><th>Error Type</th><th>Count</th></tr>')
        for _, row in error_patterns_df.head(10).iterrows():
            html.append(f'<tr><td>{row["error_type"]}</td><td>{row["total_count"]}</td></tr>')
        html.append('</table>')
    else:
        html.append('<p>No error patterns found.</p>')
    
    # Consistently misclassified examples
    html.append('<h2>Consistently Misclassified Examples</h2>')
    if not misclass_df.empty:
        html.append('<table><tr><th>Text</th><th>True Label</th><th>Predicted Label</th><th>Models</th></tr>')
        for _, row in misclass_df.head(20).iterrows():
            models = [name for name in predictions_dict.keys() if row.get(name, False)]
            html.append(f'<tr><td>{row["text"]}</td><td>{row["y_true"]}</td>')
            html.append(f'<td>{row["y_pred"]}</td><td>{", ".join(models)}</td></tr>')
        html.append('</table>')
    else:
        html.append('<p>No consistently misclassified examples found.</p>')
    
    # Model-specific analyses
    html.append('<h2>Model-Specific Error Analysis</h2>')
    for model_name, splits in predictions_dict.items():
        # We'll only analyze test set for the report
        if 'test' in splits:
            df = splits['test']
            errors = df[df['y_true'] != df['y_pred']]
            html.append(f'<h3>{model_name}</h3>')
            
            # Error count by class
            error_by_class = errors.groupby('y_true').size().reset_index(name='count')
            html.append('<h4>Error Count by True Class</h4>')
            html.append('<table><tr><th>Class</th><th>Error Count</th></tr>')
            for _, row in error_by_class.sort_values('count', ascending=False).iterrows():
                html.append(f'<tr><td>{row["y_true"]}</td><td>{row["count"]}</td></tr>')
            html.append('</table>')
            
            # Sample errors
            html.append('<h4>Sample Errors</h4>')
            html.append('<table><tr><th>Text</th><th>True Label</th><th>Predicted Label</th></tr>')
            for _, row in errors.head(5).iterrows():
                html.append(f'<tr><td>{row["text"]}</td><td>{row["y_true"]}</td><td>{row["y_pred"]}</td></tr>')
            html.append('</table>')
    
    html.append('</body></html>')
    
    # Save the report
    with open(output_dir / 'detailed_error_report.html', 'w') as f:
        f.write('\n'.join(html))

def plot_confusion_matrices(experiment_dir: str | Path, figsize: Tuple[int, int] = (15, 15)) -> None:
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
    all_labels = set()
    for splits in predictions.values():
        if 'test' in splits:
            df = splits['test']
            all_labels.update(df['y_true'].unique())
            all_labels.update(df['y_pred'].unique())
    
    all_labels = sorted(all_labels)
    
    for i, (model_name, splits) in enumerate(predictions.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Only analyze test set
            if 'test' in splits:
                df = splits['test']
                
                # Calculate confusion matrix
                cm = confusion_matrix(df['y_true'], df['y_pred'], labels=all_labels)
                
                # Normalize by row (true labels)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)
                
                # Plot heatmap
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                          xticklabels=all_labels, yticklabels=all_labels, ax=ax)
                
                ax.set_title(f'Confusion Matrix - {model_name} (Test Set)')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')
    
    # Remove empty subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_top_error_types(df: pd.DataFrame, output_path, n: int = 10):
    """
    Plot and save the top n error types (true_label -> pred_label) as a bar chart.
    Args:
        df (pd.DataFrame): DataFrame with columns 'true_label' and 'pred_label' (or 'y_true'/'y_pred').
        output_path (str or Path): Path to save the PNG plot.
        n (int): Number of top error types to plot (default 10).
    """
    # Accept both naming conventions
    if 'y_true' in df.columns and 'y_pred' in df.columns:
        y_true = df['y_true']
        y_pred = df['y_pred']
    elif 'y_true' in df.columns and 'y_pred' in df.columns:
        y_true = df['y_true']
        y_pred = df['y_pred']
    else:
        raise ValueError("DataFrame must contain either ('y_true', 'y_pred') or ('true_label', 'pred_label') columns.")

    # Create error type column
    error_df = df[y_true != y_pred].copy()
    error_df['error_type'] = y_true[y_true != y_pred].astype(str) + ' -> ' + y_pred[y_true != y_pred].astype(str)
    
    # Get top n error types
    error_counts = error_df['error_type'].value_counts().head(n)
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.bar(error_counts.index, error_counts.values)
    plt.title(f'Top {n} Error Types')
    plt.xlabel('Error Type (True → Predicted)')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 
