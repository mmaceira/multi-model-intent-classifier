"""\
Exploration module for analyzing and visualizing text datasets.

This module provides tools for analyzing and visualizing text datasets,
including functions for analyzing class distributions and text length
statistics. It helps understand data characteristics before model training.

Functions:
- class_frequency: Analyze and visualize class distribution
- length_distribution: Analyze and visualize text length distribution

Created: 2025-05-03
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd

def class_frequency(labels: np.ndarray, plot: bool = True,
                   save_path: str = None, top_n: int = None) -> Dict[str, Any]:
    """Analyze and visualize class distribution.
    
    This function analyzes the distribution of classes in the dataset
    and optionally creates a bar plot visualization. It can focus on
    the top N most frequent classes if specified.
    
    Args:
        labels: Array of class labels
        plot: Whether to create a visualization (default: True)
        save_path: Path to save the plot (default: None)
        top_n: Number of top classes to show (default: None)
        
    Returns:
        Dictionary containing:
        - counts: Series of class counts
        - proportions: Series of class proportions
        
    Example:
        >>> stats = class_frequency(y_train, plot=True, top_n=10)
        >>> print(f"Most frequent class: {stats['counts'].index[0]}")
    """
    # Compute class counts and proportions
    counts = pd.Series(labels).value_counts()
    proportions = counts / len(labels)
    
    # Optionally limit to top N classes
    if top_n is not None:
        counts = counts.head(top_n)
        proportions = proportions.head(top_n)
    
    # Create visualization if requested
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Create bar plot with better styling
        bars = plt.bar(counts.index, counts.values, color='steelblue')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # Style the plot
        plt.title('Distribution of Topics in Training Set', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
    
    return {
        'counts': counts,
        'proportions': proportions
    }

def length_distribution(texts: List[str], save_path: str = None) -> Dict[str, Any]:
    """Analyze and visualize text length distribution.
    
    This function analyzes the distribution of text lengths in the
    dataset and creates a histogram visualization. It provides
    summary statistics about the text lengths.
    
    Args:
        texts: List of text documents
        save_path: Path to save the plot (default: None)
        
    Returns:
        Dictionary containing:
        - lengths: Array of text lengths
        - stats: Dictionary of summary statistics
        
    Example:
        >>> stats = length_distribution(X_train)
        >>> print(f"Average length: {stats['stats']['mean']:.1f} words")
    """
    # Compute text lengths
    lengths = np.array([len(text.split()) for text in texts])
    
    # Compute summary statistics
    stats = {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths)
    }
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    
    # Add vertical lines for mean and median
    plt.axvline(stats['mean'], color='r', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
    plt.axvline(stats['median'], color='g', linestyle='--', label=f'Median: {stats["median"]:.1f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return {
        'lengths': lengths,
        'stats': stats
    }
