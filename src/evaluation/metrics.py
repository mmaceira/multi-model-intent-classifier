"""
Metrics computation module.

This module provides functions for computing various evaluation metrics
for text classification models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
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
        f"{split['prefix']}weighted_f1": f1_score(split["y_true"], y_pred, average="weighted"),
    }
    
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
    
    return y_pred, y_prob, metrics

def analyze_text_features(predictions_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze text features that might contribute to classification errors.
    
    Returns a dataframe with text statistics for correct and incorrect predictions.
    
    Parameters
    ----------
    predictions_dict : Dict[str, pd.DataFrame]
        Dictionary mapping model names to their prediction dataframes.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing text feature statistics for each model and prediction type.
    """
    all_rows = []
    
    for model_name, df in predictions_dict.items():
        # Add features
        df['is_correct'] = df['true_label'] == df['pred_label']
        df['text_length'] = df['text'].apply(lambda x: len(str(x)))
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        
        # Group by correct/incorrect
        for is_correct in [True, False]:
            subset = df[df['is_correct'] == is_correct]
            if len(subset) > 0:
                all_rows.append({
                    'model': model_name,
                    'is_correct': is_correct,
                    'count': len(subset),
                    'avg_text_length': subset['text_length'].mean(),
                    'avg_word_count': subset['word_count'].mean(),
                })
    
    return pd.DataFrame(all_rows)

def analyze_text_characteristics(misclassified_df: pd.DataFrame) -> None:
    """Analyze text characteristics of misclassified examples.
    
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
        'text_length': 'mean',
        'word_count': 'mean'
    }).reset_index()
    
    by_label.columns = ['true_label', 'count', 'avg_text_length', 'avg_word_count']
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
    
    # Correlation between text length and word count
    corr = analysis_df[['text_length', 'word_count']].corr()
    print("\nCorrelation between text characteristics:")
    display(corr)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show() 