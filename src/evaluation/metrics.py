"""
Metrics Module

This module provides comprehensive evaluation metrics computation for text classification models.
It supports both binary and multiclass classification scenarios, with special attention to
RAG (Retrieval-Augmented Generation) model evaluation.

Key Features:
- Standard classification metrics (accuracy, F1, etc.)
- Probability-based metrics (ROC AUC, average precision)
- Text feature analysis for error investigation
- Support for both binary and multiclass scenarios
- Proper handling of missing predictions
- Detailed logging and error reporting

Classes:
    None (Module-level functions only)

Functions:
    compute_metrics: Main function for computing all metrics
    analyze_text_features: Analyzes text characteristics of predictions
    analyze_text_characteristics: Detailed analysis of misclassified examples

Dependencies:
    numpy: For numerical operations
    pandas: For data manipulation
    sklearn.metrics: For metric computation
    matplotlib: For visualization
    seaborn: For enhanced visualization
    logging: For progress tracking

Example Usage:
    >>> from pathlib import Path
    >>> from evaluation.metrics import compute_metrics
    >>> 
    >>> # Compute metrics for a model
    >>> metrics = compute_metrics(
    ...     model_dir=Path("models/model1"),
    ...     split={"name": "test", "y_true": y_true, ...},
    ...     classes=np.array(["class1", "class2"]),
    ...     is_multiclass=True,
    ...     logger=logging.getLogger(__name__)
    ... )
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
    """
    Compute comprehensive metrics for a model's predictions.
    
    This function serves as the main entry point for metric computation. It handles:
    - Loading predictions and probabilities
    - Computing standard classification metrics
    - Generating classification reports
    - Computing probability-based metrics when available
    - Proper error handling and logging
    
    Parameters
    ----------
    model_dir : Path
        Directory containing model artifacts and predictions
    split : Dict[str, Any]
        Dictionary containing split information and data, including:
        - name: Split name (e.g., 'train', 'test')
        - pred_file: Path to predictions file
        - prob_file: Path to probabilities file
        - y_true: True labels
        - y_true_bin: Binarized true labels (for multiclass)
        - prefix: Prefix for metric names
        - out_dir: Output directory for reports
    classes : np.ndarray
        Array of unique class labels
    is_multiclass : bool
        Whether the task is multiclass classification
    logger : logging.Logger
        Logger instance for progress tracking
        
    Returns
    -------
    Optional[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]]
        Tuple containing:
        - y_pred: Predicted labels
        - y_prob: Predicted probabilities (if available)
        - metrics: Dictionary of computed metrics
        
    Raises
    ------
    FileNotFoundError
        If prediction file is missing for test split
    ValueError
        If probabilities are not properly normalized
        
    Example
    -------
    >>> split = {
    ...     "name": "test",
    ...     "pred_file": "test_predictions.csv",
    ...     "y_true": y_true,
    ...     "prefix": "test_",
    ...     "out_dir": Path("results")
    ... }
    >>> metrics = compute_metrics(
    ...     model_dir=Path("models/model1"),
    ...     split=split,
    ...     classes=np.array(["class1", "class2"]),
    ...     is_multiclass=True,
    ...     logger=logging.getLogger(__name__)
    ... )
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

def analyze_text_features(predictions_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze text features that might contribute to classification errors.
    
    This function examines various text characteristics (length, word count, etc.)
    to identify patterns in correct and incorrect predictions. It helps understand
    if certain text properties correlate with classification success or failure.
    
    Parameters
    ----------
    predictions_dict : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
        Each DataFrame should contain:
        - text: The input text
        - y_true: True labels
        - y_pred: Predicted labels
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing text feature statistics for each model and prediction type,
        including:
        - model: Model name
        - split: Data split (train/test)
        - is_correct: Whether prediction was correct
        - count: Number of examples
        - avg_text_length: Average text length
        - avg_word_count: Average word count
        
    Example
    -------
    >>> predictions = {
    ...     "model1": {
    ...         "train": train_df,
    ...         "test": test_df
    ...     }
    ... }
    >>> features = analyze_text_features(predictions)
    >>> print(features.head())
    """
    all_rows = []
    
    for model_name, splits in predictions_dict.items():
        # We'll analyze both train and test sets
        for split_name, df in splits.items():
            # Add features
            df['is_correct'] = df['y_true'] == df['y_pred']
            df['text_length'] = df['text'].apply(lambda x: len(str(x)))
            df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
            
            # Group by correct/incorrect
            for is_correct in [True, False]:
                subset = df[df['is_correct'] == is_correct]
                if len(subset) > 0:
                    all_rows.append({
                        'model': model_name,
                        'split': split_name,
                        'is_correct': is_correct,
                        'count': len(subset),
                        'avg_text_length': subset['text_length'].mean(),
                        'avg_word_count': subset['word_count'].mean(),
                    })
    
    return pd.DataFrame(all_rows)

def analyze_text_characteristics(misclassified_df: pd.DataFrame) -> None:
    """
    Analyze text characteristics of misclassified examples.
    
    This function provides a detailed analysis of misclassified examples, including:
    - Text length distributions
    - Word count distributions
    - Correlation between text features
    - Visualizations of these characteristics
    
    Parameters
    ----------
    misclassified_df : pd.DataFrame
        DataFrame containing misclassified examples with columns:
        - text: The input text
        - true_label: True class label
        - pred_label: Predicted class label
        
    Returns
    -------
    None
        This function generates visualizations but does not return any values.
        
    Example
    -------
    >>> misclassified = df[df['y_true'] != df['y_pred']]
    >>> analyze_text_characteristics(misclassified)
    """
    # Calculate text length and word count
    misclassified_df['text_length'] = misclassified_df['text'].apply(lambda x: len(str(x)))
    misclassified_df['word_count'] = misclassified_df['text'].apply(lambda x: len(str(x).split()))
    
    # Group by true label and compute statistics
    stats = misclassified_df.groupby('true_label').agg({
        'text_length': ['count', 'mean', 'std'],
        'word_count': ['mean', 'std']
    }).round(2)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Text length distribution
    sns.histplot(data=misclassified_df, x='text_length', hue='true_label', ax=ax1)
    ax1.set_title('Distribution of Text Length by True Label')
    ax1.set_xlabel('Text Length')
    
    # Word count distribution
    sns.histplot(data=misclassified_df, x='word_count', hue='true_label', ax=ax2)
    ax2.set_title('Distribution of Word Count by True Label')
    ax2.set_xlabel('Word Count')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation between text length and word count
    corr = misclassified_df[['text_length', 'word_count']].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='Blues', center=0,
                vmin=-1, vmax=1, square=True)
    plt.title('Correlation between Text Length and Word Count')
    plt.show() 