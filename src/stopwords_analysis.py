"""
Stopwords and Vocabulary Analysis Module

This module provides comprehensive tools for analyzing vocabulary distributions and stopword
impacts in text classification datasets. It includes functionality for filtering common words,
financial terms, and numerical tokens, along with visualization capabilities for understanding
word frequency patterns.

Key Features:
- Customizable stopword filtering
- Financial term detection and filtering
- Numerical token handling
- Word frequency analysis
- Class-specific vocabulary analysis
- Publication-ready visualizations
- Side-by-side comparisons

Constants:
- STOPWORDS: Set of common English stopwords
- FINANCIAL_TERMS: Set of common financial and numerical terms

Functions:
- is_numeric_or_financial: Detect numeric values and financial terms
- analyze_vocabulary: Analyze word distributions with filtering options
- compare_with_without_stopwords: Compare vocabulary with/without stopwords
- word_frequency_by_class: Analyze word frequencies per class

Filtering Options:
- Stopword removal
- Numeric token removal
- Financial term removal
- Minimum word length
- Custom term lists

Visualization Features:
- Word frequency bar plots
- Side-by-side comparisons
- Class-specific distributions
- Customizable styling
- Publication-ready outputs

Dependencies:
- re
- matplotlib
- numpy
- pandas
- collections
- typing
- seaborn

Example Usage:
    >>> # Basic vocabulary analysis
    >>> stats = analyze_vocabulary(
    ...     texts=documents,
    ...     remove_stopwords=True,
    ...     remove_numbers=True,
    ...     min_word_length=3,
    ...     n_most_common=30
    ... )
    >>> print(f"Vocabulary size: {stats['vocab_size']}")
    
    >>> # Compare with/without stopwords
    >>> compare_with_without_stopwords(
    ...     texts=documents,
    ...     n_words=20,
    ...     remove_financial_terms=True
    ... )
    
    >>> # Analyze word frequencies by class
    >>> class_freqs = word_frequency_by_class(
    ...     texts=documents,
    ...     labels=labels,
    ...     label_names=['Class A', 'Class B'],
    ...     n_words=10
    ... )
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional
import seaborn as sns

# Define common English stopwords
STOPWORDS = {
    "the", "of", "to", "in", "and", "a", "for", "it", "on", "its", "with", "as", "by", 
    "at", "from", "that", "this", "be", "is", "are", "was", "were", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "an", "the", 
    "but", "if", "or", "because", "until", "while", "about", "against", "between", 
    "into", "through", "during", "before", "after", "above", "below", "over", "under", 
    "than", "too", "very", "can", "will", "just", "should", "now", "i", "me", "my", 
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "itself", 
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", 
    "whose", "when", "where", "why", "how", "any", "both", "each", "few", "more", 
    "most", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "said"
}

# Extended financial/numeric terms to filter
FINANCIAL_TERMS = {
    "dlrs", "mln", "pct", "cts", "shr", "lt", "billion", "tonnes", "bpd", "000",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "v", "vs", "inc", "corp"
}

def is_numeric_or_financial(word: str) -> bool:
    """
    Determine if a word is numeric or a financial term.
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is a numeric value or financial term
    """
    # Check if it's entirely digits
    if word.isdigit():
        return True
    
    # Check if it's in our financial terms list
    if word in FINANCIAL_TERMS:
        return True
    
    # Check if it's a number with commas, decimal points, or other formatting
    if re.match(r'^[0-9,\.]+$', word):
        return True
    
    # Check if it starts with a number followed by other characters
    if re.match(r'^[0-9]+[a-zA-Z]*$', word):
        return True
    
    # Check for currency symbols
    if re.match(r'^[\$£€¥]?[0-9,\.]+$', word):
        return True
    
    return False

def analyze_vocabulary(texts: List[str], remove_stopwords: bool = True, 
                       remove_numbers: bool = True,
                       remove_financial_terms: bool = False,
                       min_word_length: int = 1,
                       n_most_common: int = 30, plot: bool = True, 
                       figsize: Tuple[int, int] = (12, 8)) -> Dict:
    """
    Analyze vocabulary distribution with various filtering options.
    
    Args:
        texts: List of text documents
        remove_stopwords: Whether to remove common English stopwords
        remove_numbers: Whether to remove numeric tokens
        remove_financial_terms: Whether to remove common financial terms
        min_word_length: Minimum length of words to include
        n_most_common: Number of most common words to return
        plot: Whether to create a visualization
        figsize: Figure size for the plot
        
    Returns:
        Dictionary containing:
        - vocab_size: Total unique words in vocabulary
        - word_counts: Counter object with word frequencies
        - top_words: DataFrame of n most common words
    """
    # Tokenize and count words
    words = []
    for doc in texts:
        words.extend(re.findall(r'\w+', doc.lower()))
    
    # Filter by minimum word length
    if min_word_length > 1:
        words = [word for word in words if len(word) >= min_word_length]
    
    # Filter stopwords if requested
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS]
    
    # Filter numbers and financial terms if requested
    filtered_words = []
    for word in words:
        if remove_numbers and is_numeric_or_financial(word):
            continue
        if remove_financial_terms and word in FINANCIAL_TERMS:
            continue
        filtered_words.append(word)
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get most common words
    most_common = word_counts.most_common(n_most_common)
    top_words = pd.DataFrame(most_common, columns=['word', 'count'])
    
    # Create visualization if requested
    if plot:
        plt.figure(figsize=figsize)
        sns.barplot(x='count', y='word', data=top_words.head(30), palette='viridis')
        title_parts = []
        if remove_stopwords:
            title_parts.append("Stopwords Removed")
        if remove_numbers:
            title_parts.append("Numbers Removed")
        if remove_financial_terms:
            title_parts.append("Financial Terms Removed")
        if min_word_length > 1:
            title_parts.append(f"Min Word Length: {min_word_length}")
        title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
        plt.title(f'Top Words in Corpus{title_suffix}')
        plt.xlabel('Count')
        plt.ylabel('Word')
        plt.tight_layout()
        plt.show()
    
    return {
        'vocab_size': len(word_counts),
        'word_counts': word_counts,
        'top_words': top_words
    }

def compare_with_without_stopwords(texts: List[str], n_words: int = 20, 
                                  remove_numbers: bool = True,
                                  remove_financial_terms: bool = False,
                                  min_word_length: int = 1) -> None:
    """
    Compare vocabulary analysis with and without stopwords side by side.
    
    Args:
        texts: List of text documents
        n_words: Number of top words to show
        remove_numbers: Whether to remove numeric tokens
        remove_financial_terms: Whether to remove common financial terms
        min_word_length: Minimum length of words to include
    """
    # Analyze with stopwords
    with_stopwords = analyze_vocabulary(texts, remove_stopwords=False, 
                                       remove_numbers=remove_numbers,
                                       remove_financial_terms=remove_financial_terms,
                                       min_word_length=min_word_length,
                                       n_most_common=n_words, plot=False)
    
    # Analyze without stopwords
    without_stopwords = analyze_vocabulary(texts, remove_stopwords=True,
                                         remove_numbers=remove_numbers,
                                         remove_financial_terms=remove_financial_terms,
                                         min_word_length=min_word_length,
                                         n_most_common=n_words, plot=False)
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Prepare title suffix
    title_parts = []
    if remove_numbers:
        title_parts.append("Numbers Removed")
    if remove_financial_terms:
        title_parts.append("Financial Terms Removed")
    if min_word_length > 1:
        title_parts.append(f"Min Length: {min_word_length}")
    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
    
    # Plot with stopwords
    sns.barplot(x='count', y='word', data=with_stopwords['top_words'], ax=ax1, palette='Blues_d')
    ax1.set_title(f'Top {n_words} Words (With Stopwords{title_suffix})')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Word')
    
    # Plot without stopwords
    sns.barplot(x='count', y='word', data=without_stopwords['top_words'], ax=ax2, palette='Reds_d')
    ax2.set_title(f'Top {n_words} Words (Stopwords Removed{title_suffix})')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('')  # No need for duplicate y-label
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"With stopwords: {with_stopwords['vocab_size']:,} unique words")
    print(f"Without stopwords: {without_stopwords['vocab_size']:,} unique words")
    
    # Create comparison DataFrame with proper padding for different length arrays
    with_words = [f"{word} ({count:,})" for word, count in with_stopwords['word_counts'].most_common(n_words)]
    without_words = [f"{word} ({count:,})" for word, count in without_stopwords['word_counts'].most_common(n_words)]
    
    # Ensure both arrays have the same length by padding with empty strings
    max_len = max(len(with_words), len(without_words))
    with_words += [''] * (max_len - len(with_words))
    without_words += [''] * (max_len - len(without_words))
    
    # Create DataFrame
    comparison = pd.DataFrame({
        'With Stopwords': with_words,
        'Without Stopwords': without_words
    })
    
    print("\nTop words comparison:")
    return comparison

def word_frequency_by_class(texts: List[str], labels: List[str], 
                           label_names: List[str] = None,
                           remove_stopwords: bool = True,
                           remove_numbers: bool = True,
                           remove_financial_terms: bool = False,
                           min_word_length: int = 1,
                           n_words: int = 10) -> pd.DataFrame:
    """
    Analyze most frequent words for each class.
    
    Args:
        texts: List of text documents
        labels: List of corresponding class labels
        label_names: List of label names (optional)
        remove_stopwords: Whether to remove common English stopwords
        remove_numbers: Whether to remove numeric tokens
        remove_financial_terms: Whether to remove common financial terms
        min_word_length: Minimum length of words to include
        n_words: Number of top words per class to show
        
    Returns:
        DataFrame with top words by class
    """
    if label_names is None:
        label_names = sorted(set(labels))
    
    results = {}
    max_length = 0
    
    for label in label_names:
        # Filter texts for this class
        class_texts = [text for text, l in zip(texts, labels) if l == label]
        
        # Skip if no texts for this class
        if not class_texts:
            continue
            
        # Analyze vocabulary for this class
        analysis = analyze_vocabulary(class_texts, remove_stopwords=remove_stopwords,
                                     remove_numbers=remove_numbers,
                                     remove_financial_terms=remove_financial_terms,
                                     min_word_length=min_word_length,
                                     n_most_common=n_words, plot=False)
        
        # Store results
        results[label] = [f"{word} ({count:,})" for word, count in 
                         analysis['word_counts'].most_common(n_words)]
        max_length = max(max_length, len(results[label]))
    
    # Pad shorter lists with empty strings to ensure consistent length
    for label in results:
        results[label] += [''] * (max_length - len(results[label]))
    
    # Convert to DataFrame
    return pd.DataFrame(results)

if __name__ == "__main__":
    # This section demonstrates how to use the functions
    # Example usage (replace with your actual data):
    # from src.dataset import load_data
    # X_train, y_train, X_test, y_test, label_names = load_data(10)
    
    # 1. Basic analysis without stopwords
    # results = analyze_vocabulary(X_train, remove_numbers=True, remove_financial_terms=True, min_word_length=3)
    # print(f"Total unique words (filtered): {results['vocab_size']:,}")
    
    # 2. Compare with and without stopwords
    # comparison = compare_with_without_stopwords(X_train, remove_numbers=True, remove_financial_terms=True, min_word_length=3)
    # print(comparison)
    
    # 3. Analyze by class
    # class_words = word_frequency_by_class(X_train, y_train, label_names, 
    #                                      remove_numbers=True, 
    #                                      remove_financial_terms=True,
    #                                      min_word_length=3)
    # print(class_words)
    
    print("Run this script after importing your data to analyze vocabulary with enhanced filtering options.") 