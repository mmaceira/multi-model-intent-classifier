"""\
Model results comparison module.

This module provides tools for comparing and visualizing the performance
of different text classification models. It includes functions for loading
evaluation results and creating visualizations to compare model metrics.

Functions:
- load_results: Load and aggregate model evaluation results
- plot_macro_f1: Create bar plot of model F1 scores

Created: 2025-05-03
"""

# model_results_comparison.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """Load and aggregate model evaluation results.
    
    This function loads evaluation results from CSV files in the
    specified directory and aggregates them into a single DataFrame.
    It assumes each model's results are stored in a separate CSV file.
    
    Args:
        results_dir: Directory containing result files (default: 'results')
        
    Returns:
        DataFrame containing aggregated evaluation metrics
        
    Example:
        >>> df = load_results('results')
        >>> print(f"Number of models: {len(df)}")
    """
    results = []
    
    # Load each model's results
    for filename in os.listdir(results_dir):
        if filename.endswith('_report.csv'):
            model_name = filename.replace('_report.csv', '')
            df = pd.read_csv(os.path.join(results_dir, filename))
            df['model'] = model_name
            results.append(df)
    
    # Combine all results
    if not results:
        return pd.DataFrame()
        
    return pd.concat(results, ignore_index=True)

def plot_macro_f1(df: pd.DataFrame, save_path: str = None) -> None:
    """Create bar plot of model macro-F1 scores.
    
    This function creates a bar plot comparing the macro-F1 scores
    of different models. The models are sorted by F1 score in
    descending order.
    
    Args:
        df: DataFrame containing model evaluation results
        save_path: Path to save the plot (default: None)
        
    Example:
        >>> df = load_results()
        >>> plot_macro_f1(df, save_path='results/f1_comparison.png')
    """
    if df.empty:
        return
        
    # Extract macro-F1 scores
    f1_scores = df[df['Unnamed: 0'] == 'macro avg']['f1-score']
    models = df[df['Unnamed: 0'] == 'macro avg']['model']
    
    # Sort by F1 score
    order = f1_scores.argsort()[::-1]
    f1_scores = f1_scores.iloc[order]
    models = models.iloc[order]
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=f1_scores)
    plt.title('Model Comparison (Macro-F1 Score)')
    plt.xlabel('Model')
    plt.ylabel('Macro-F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()