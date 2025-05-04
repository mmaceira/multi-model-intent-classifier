"""
Interactive error analysis for notebook use.

This script provides functions for examining misclassified examples
and analyzing error patterns in classification models.

Example usage in a notebook:
```python
from notebooks.error_analysis import analyze_errors, plot_confusion_matrices

# Load and analyze all model predictions
results = analyze_errors("experiment_with_07_classes", interactive=True)

# Plot confusion matrices for all models
plot_confusion_matrices("experiment_with_07_classes")

# Examine text characteristics of misclassified examples
analyze_text_characteristics(results['misclassified_examples'])
```
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix
import glob

# Add parent directory to path if running in notebook
import sys
sys.path.append('..')
try:
    from src.enhanced_qualitative_analysis import (
        load_all_prediction_files,
        analyze_error_patterns,
        identify_consistently_misclassified,
        analyze_text_features
    )
except ImportError:
    print("Error: Could not import enhanced_qualitative_analysis module. Make sure the path is correct.")


def analyze_errors(experiment_dir, interactive=False):
    """
    Analyze classification errors across all models.
    
    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing the experiment results
    interactive : bool
        If True, display interactive visualizations
        
    Returns
    -------
    dict
        Dictionary containing error patterns and misclassified examples
    """
    experiment_dir = Path(experiment_dir)
    predictions = load_all_prediction_files(experiment_dir)
    
    # Analyze error patterns
    error_patterns = analyze_error_patterns(predictions)
    
    # Find consistently misclassified examples
    misclassified = identify_consistently_misclassified(predictions)
    
    # Analyze text features
    text_features = analyze_text_features(predictions)
    
    if interactive:
        # Display interactive visualizations
        print(f"Loaded {len(predictions)} model prediction files")
        
        # Show error patterns
        if not error_patterns.empty:
            print("\nTop 10 Common Error Patterns:")
            display(error_patterns.head(10))
        
        # Show most consistently misclassified examples
        if not misclassified.empty:
            print(f"\nFound {len(misclassified)} examples misclassified by multiple models")
            print("Top misclassified examples:")
            display(misclassified.head(10))
        
        # Plot error type distribution
        if not error_patterns.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='error_type', y='total_count', data=error_patterns.head(10))
            plt.title('Top 10 Error Types')
            plt.xlabel('Error Type (True → Predicted)')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    return {
        'error_patterns': error_patterns,
        'misclassified_examples': misclassified,
        'text_features': text_features,
        'predictions': predictions
    }


def plot_confusion_matrices(experiment_dir, figsize=(15, 15)):
    """
    Plot confusion matrices for all models in the experiment.
    
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
    
    # Get all unique labels from all models
    all_labels = set()
    for df in predictions.values():
        all_labels.update(df['true_label'].unique())
        all_labels.update(df['pred_label'].unique())
    
    all_labels = sorted(all_labels)
    
    for i, (model_name, df) in enumerate(predictions.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Calculate confusion matrix
            cm = confusion_matrix(df['true_label'], df['pred_label'], labels=all_labels)
            
            # Normalize by row (true labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Plot heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                      xticklabels=all_labels, yticklabels=all_labels, ax=ax)
            
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for i in range(len(predictions), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_text_characteristics(misclassified_df):
    """
    Analyze text characteristics of misclassified examples.
    
    Parameters
    ----------
    misclassified_df : DataFrame
        DataFrame containing misclassified examples
    """
    if misclassified_df.empty:
        print("No misclassified examples to analyze")
        return
    
    # Add text length and word count
    analysis_df = misclassified_df.copy()
    analysis_df['text_length'] = analysis_df['text'].apply(lambda x: len(str(x)))
    analysis_df['word_count'] = analysis_df['text'].apply(lambda x: len(str(x).split()))
    
    # Analyze by true label
    by_label = analysis_df.groupby('true_label').agg({
        'id': 'count',
        'misclassified_count': 'mean',
        'text_length': 'mean',
        'word_count': 'mean'
    }).reset_index()
    
    by_label.columns = ['true_label', 'count', 'avg_misclassified_models', 'avg_text_length', 'avg_word_count']
    by_label = by_label.sort_values('count', ascending=False)
    
    print("Misclassification analysis by true label:")
    display(by_label)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot text length distribution
    sns.histplot(analysis_df['text_length'], ax=axes[0], kde=True)
    axes[0].set_title('Distribution of Text Length')
    axes[0].set_xlabel('Text Length (characters)')
    
    # Plot word count distribution
    sns.histplot(analysis_df['word_count'], ax=axes[1], kde=True)
    axes[1].set_title('Distribution of Word Count')
    axes[1].set_xlabel('Word Count')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation between text length and misclassification count
    corr = analysis_df[['text_length', 'word_count', 'misclassified_count']].corr()
    print("\nCorrelation between text characteristics and misclassification:")
    display(corr)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def export_analysis_results(results, output_dir):
    """
    Export analysis results to files.
    
    Parameters
    ----------
    results : dict
        Dictionary with analysis results
    output_dir : str or Path
        Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save error patterns
    if not results['error_patterns'].empty:
        results['error_patterns'].to_csv(output_dir / 'error_patterns.csv', index=False)
    
    # Save misclassified examples
    if not results['misclassified_examples'].empty:
        results['misclassified_examples'].to_csv(output_dir / 'misclassified_examples.csv', index=False)
    
    # Save text features
    if not results['text_features'].empty:
        results['text_features'].to_csv(output_dir / 'text_features.csv', index=False)
    
    print(f"Analysis results exported to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python error_analysis.py <experiment_dir> [output_dir]")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "error_analysis_results"
    
    results = analyze_errors(experiment_dir, interactive=False)
    export_analysis_results(results, output_dir)
    print("Analysis complete.") 