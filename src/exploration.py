"""
Text Dataset Exploration Module

This module provides comprehensive tools for analyzing and visualizing text classification datasets,
helping researchers and practitioners understand their data characteristics before model training.
It includes functions for analyzing class distributions, text length statistics, topic
co-occurrence patterns, and vocabulary drift between splits.

Key Features:
- Class distribution analysis and visualization
- Text length statistics and distribution plots
- Topic co-occurrence pattern analysis
- Vocabulary drift analysis between train/test splits
- Customizable visualizations with publication-ready styling
- Detailed statistical summaries
- CSV export capabilities for further analysis

Functions:
- class_frequency: Analyze and visualize class distribution with optional top-N filtering
- length_distribution: Analyze text length patterns with percentile statistics
- topic_cooccurrence: Create and visualize topic co-occurrence matrices
- vocabulary_drift: Analyze vocabulary differences between train and test splits

Statistical Outputs:
- Class frequencies and proportions
- Text length summary statistics (mean, median, std, min, max)
- Customizable percentile statistics
- Topic co-occurrence matrices
- Token frequency drift metrics

Visualizations:
- Bar plots of class distributions
- Histograms of text lengths
- Heatmaps of topic co-occurrence
- Publication-ready plots with proper styling

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- typing
- os
- collections

Example Usage:
    >>> # Analyze class distribution
    >>> class_stats = class_frequency(
    ...     labels=y_train,
    ...     plot=True,
    ...     save_path='class_dist.png',
    ...     top_n=10
    ... )
    
    >>> # Analyze text lengths
    >>> length_stats = length_distribution(
    ...     texts=X_train,
    ...     save_path='length_dist.png',
    ...     output_dir='stats',
    ...     percentiles=[25, 50, 75, 90, 95, 99]
    ... )
    
    >>> # Analyze topic co-occurrence
    >>> cooccurrence = topic_cooccurrence(
    ...     labels=y_train,
    ...     save_path='cooccurrence.png'
    ... )
    
    >>> # Analyze vocabulary drift
    >>> drift = vocabulary_drift(
    ...     train_texts=X_train,
    ...     test_texts=X_test,
    ...     top_k=2000,
    ...     output_path='vocab_drift.csv'
    ... )
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Sequence
import pandas as pd
import os
from pathlib import Path
from collections import Counter

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

def length_distribution(texts: List[str], save_path: str = None, output_dir: str = None, 
                     percentiles: List[int] = [25, 50, 75, 90, 95, 99]) -> Dict[str, Any]:
    """Analyze and visualize text length distribution.
    
    This function analyzes the distribution of text lengths in the
    dataset and creates a histogram visualization. It provides
    summary statistics about the text lengths, including percentiles.
    
    Args:
        texts: List of text documents
        save_path: Path to save the plot (default: None)
        output_dir: Directory to save additional output files like CSV (default: None)
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 90, 95, 99])
        
    Returns:
        Dictionary containing:
        - lengths: Array of text lengths
        - stats: Dictionary of summary statistics
        - percentile_stats: List of dictionaries with percentile information
        
    Example:
        >>> stats = length_distribution(X_train)
        >>> print(f"Average length: {stats['stats']['mean']:.1f} words")
        >>> print(f"90th percentile: {stats['percentile_stats'][3]['length']} words")
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
    
    # Calculate percentiles
    percentile_values = np.percentile(lengths, percentiles)
    percentile_stats = []
    
    for p, percentile in zip(percentiles, percentile_values):
        percentile_stats.append({'percentile': p, 'length': int(percentile)})
        
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
    
    # Save percentile statistics to CSV if output_dir is provided
    if output_dir:
        csv_path = os.path.join(output_dir, "document_length_stats.csv")
        pd.DataFrame(percentile_stats).to_csv(csv_path, index=False)
        print("\nDocument Length Percentiles:")
        for stat in percentile_stats:
            print(f"{stat['percentile']}th percentile: {stat['length']} tokens")
    
    return {
        'lengths': lengths,
        'stats': stats,
        'percentile_stats': percentile_stats
    }

def vocabulary_drift(
    train_texts: Sequence[str], 
    test_texts: Sequence[str], 
    *, 
    top_k: int = 2000,
    min_freq: int = 10,
    output_path: str|Path = None
) -> pd.DataFrame:
    """Compute token frequency drift between train and test splits.
    
    Parameters
    ----------
    train_texts : Sequence[str]
        Training documents
    test_texts : Sequence[str]
        Test documents
    top_k : int, optional
        Number of most frequent tokens to consider
    min_freq : int, optional
        Minimum number of times a token must appear in train + test to be included
    output_path : str|Path, optional
        Path to save results
    
    Returns
    -------
    pd.DataFrame
        DataFrame with token frequencies and drift metrics
    """
    def _tokenize(s: str):
        return s.lower().split()
    
    train_counts = Counter(token for doc in train_texts for token in _tokenize(doc))
    test_counts = Counter(token for doc in test_texts for token in _tokenize(doc))
    
    # restrict to top_k in either split
    vocab = set([w for w,_ in train_counts.most_common(top_k)] + [w for w,_ in test_counts.most_common(top_k)])
    rows = []
    
    for w in vocab:
        tr = train_counts.get(w, 0)
        te = test_counts.get(w, 0)
        total = tr + te
        if total < min_freq:
            continue
        rows.append({
            "token": w, 
            "train_freq": tr/total, 
            "test_freq": te/total, 
            "abs_diff": abs(tr/total - te/total)
        })
    
    df = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)
    
    # Save results if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df
