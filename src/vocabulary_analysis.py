#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union, Set
import seaborn as sns
import os

# Import analysis functions
try:
    from src.stopwords_analysis import analyze_vocabulary, compare_with_without_stopwords, word_frequency_by_class
    from src.datasets.dataset import load_data
    external_imports = True
except ImportError:
    external_imports = False

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

# Additional domain-specific stopwords for Reuters news
ADDITIONAL_STOPS = {
    "said", "would", "also", "one", "two", "three", "last", "first", "new", 
    "may", "could", "will", "says", "told", "reuters"
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

def _analyze_vocabulary(texts: List[str], remove_stopwords: bool = True, 
                       remove_numbers: bool = True,
                       remove_financial_terms: bool = False,
                       additional_stopwords: Set[str] = None,
                       min_word_length: int = 1,
                       n_most_common: int = 30, plot: bool = True, 
                       figsize: Tuple[int, int] = (12, 8),
                       title: str = 'Top Words in Corpus') -> Dict:
    """
    Internal vocabulary analysis function with all filtering options.
    
    Args:
        texts: List of text documents
        remove_stopwords: Whether to remove common English stopwords
        remove_numbers: Whether to remove numeric tokens
        remove_financial_terms: Whether to remove common financial terms
        additional_stopwords: Optional set of additional stopwords to remove
        min_word_length: Minimum length of words to include
        n_most_common: Number of most common words to return
        plot: Whether to create a visualization
        figsize: Figure size for the plot
        title: Plot title
        
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
    
    # Filter additional stopwords if provided
    if additional_stopwords:
        words = [word for word in words if word not in additional_stopwords]
    
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
        if additional_stopwords:
            title_parts.append("Additional Stopwords Removed")
        if min_word_length > 1:
            title_parts.append(f"Min Word Length: {min_word_length}")
        title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
        plt.title(f'{title}{title_suffix}')
        plt.xlabel('Count')
        plt.ylabel('Word')
        plt.tight_layout()
        plt.show()
    
    return {
        'vocab_size': len(word_counts),
        'word_counts': word_counts,
        'top_words': top_words
    }

def _word_frequency_by_class(texts: List[str], labels: List[str], 
                           label_names: List[str] = None,
                           remove_stopwords: bool = True,
                           remove_numbers: bool = True,
                           remove_financial_terms: bool = False,
                           additional_stopwords: Set[str] = None,
                           min_word_length: int = 1,
                           n_words: int = 10) -> pd.DataFrame:
    """
    Internal function to analyze most frequent words for each class.
    
    Args:
        texts: List of text documents
        labels: List of corresponding class labels
        label_names: List of label names (optional)
        remove_stopwords: Whether to remove common English stopwords
        remove_numbers: Whether to remove numeric tokens
        remove_financial_terms: Whether to remove common financial terms
        additional_stopwords: Optional set of additional stopwords to remove
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
        analysis = _analyze_vocabulary(
            class_texts, 
            remove_stopwords=remove_stopwords,
            remove_numbers=remove_numbers,
            remove_financial_terms=remove_financial_terms,
            additional_stopwords=additional_stopwords,
            min_word_length=min_word_length,
            n_most_common=n_words, 
            plot=False
        )
        
        # Store results
        results[label] = [f"{word} ({count:,})" for word, count in 
                         analysis['word_counts'].most_common(n_words)]
        max_length = max(max_length, len(results[label]))
    
    # Pad shorter lists with empty strings to ensure consistent length
    for label in results:
        results[label] += [''] * (max_length - len(results[label]))
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def comprehensive_analysis(
    texts: List[str],
    labels: Optional[List[str]] = None,
    label_names: Optional[List[str]] = None,
    output_dir: str = '.',
    min_word_length: int = 3,
    top_n: int = 30,
    create_visualizations: bool = True,
    create_csv: bool = True
) -> Dict:
    """
    Perform comprehensive vocabulary analysis with enhanced filtering.
    
    This function runs a series of analyses to understand the vocabulary
    distribution in the provided texts, applying various filtering techniques
    to remove noise and focus on meaningful content words.
    
    Args:
        texts: List of text documents to analyze
        labels: Optional list of class labels for class-specific analysis
        label_names: Optional list of label names for the classes
        output_dir: Directory to save output files (default: current directory)
        min_word_length: Minimum length of words to include (default: 3)
        top_n: Number of top words to return (default: 30)
        create_visualizations: Whether to create and save visualizations (default: True)
        create_csv: Whether to save results to CSV files (default: True)
        
    Returns:
        Dictionary containing the results of all analyses
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print(f"Running comprehensive vocabulary analysis on {len(texts)} documents...")
    
    # 1. Basic analysis (only removing stopwords)
    print("\n======= BASIC FILTERING =======")
    print("Settings: Remove stopwords only")
    
    basic_results = _analyze_vocabulary(
        texts,
        remove_stopwords=True,
        remove_numbers=False,
        remove_financial_terms=False,
        additional_stopwords=None,
        min_word_length=1,
        n_most_common=top_n,
        plot=False
    )
    
    results['basic'] = basic_results
    print(f"Total unique words (basic filtering): {basic_results['vocab_size']:,}")
    print("\nTop 10 words (basic filtering):")
    for word, count in basic_results['word_counts'].most_common(10):
        print(f"  {word}: {count:,}")
    
    if create_csv:
        pd.DataFrame(basic_results['word_counts'].most_common(top_n), 
                     columns=['word', 'count']).to_csv(
            os.path.join(output_dir, 'basic_filtering.csv'), index=False
        )
    
    # 2. Standard analysis (removing stopwords, numbers, and financial terms)
    print("\n======= STANDARD FILTERING =======")
    print("Settings: Remove stopwords, numbers, and financial terms")
    
    standard_results = _analyze_vocabulary(
        texts,
        remove_stopwords=True,
        remove_numbers=True,
        remove_financial_terms=True,
        additional_stopwords=None,
        min_word_length=min_word_length,
        n_most_common=top_n,
        plot=False
    )
    
    results['standard'] = standard_results
    print(f"Total unique words (standard filtering): {standard_results['vocab_size']:,}")
    print("\nTop 10 words (standard filtering):")
    for word, count in standard_results['word_counts'].most_common(10):
        print(f"  {word}: {count:,}")
    
    if create_csv:
        pd.DataFrame(standard_results['word_counts'].most_common(top_n), 
                     columns=['word', 'count']).to_csv(
            os.path.join(output_dir, 'standard_filtering.csv'), index=False
        )
    
    # 3. Advanced analysis (all filters including additional stopwords)
    print("\n======= ADVANCED FILTERING =======")
    print("Settings: Remove stopwords, numbers, financial terms, and additional domain-specific stopwords")
    
    advanced_results = _analyze_vocabulary(
        texts,
        remove_stopwords=True,
        remove_numbers=True,
        remove_financial_terms=True,
        additional_stopwords=ADDITIONAL_STOPS,
        min_word_length=min_word_length,
        n_most_common=top_n,
        plot=False
    )
    
    results['advanced'] = advanced_results
    print(f"Total unique words (advanced filtering): {advanced_results['vocab_size']:,}")
    print("\nTop 10 words (advanced filtering):")
    for word, count in advanced_results['word_counts'].most_common(10):
        print(f"  {word}: {count:,}")
    
    if create_csv:
        pd.DataFrame(advanced_results['word_counts'].most_common(top_n), 
                     columns=['word', 'count']).to_csv(
            os.path.join(output_dir, 'advanced_filtering.csv'), index=False
        )
    
    # Create top words visualization
    if create_visualizations:
        top_words = pd.DataFrame(advanced_results['word_counts'].most_common(25), 
                                columns=['word', 'count'])
        plt.figure(figsize=(12, 10))
        plt.barh(top_words['word'], top_words['count'], color='steelblue')
        plt.title('Top 25 Words After Comprehensive Filtering')
        plt.xlabel('Count')
        plt.ylabel('Word')
        plt.gca().invert_yaxis()  # Put the largest at the top
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_words_comprehensive.png'))
        plt.close()
        print("Visualization saved to 'top_words_comprehensive.png'")
    
    # Class-specific analysis if labels are provided
    if labels and len(set(labels)) > 1:
        print("\n======= CLASS-SPECIFIC WORD ANALYSIS =======")
        
        # Basic class analysis (only removing stopwords)
        basic_class_results = _word_frequency_by_class(
            texts,
            labels,
            label_names=label_names,
            remove_stopwords=True,
            remove_numbers=False,
            remove_financial_terms=False,
            additional_stopwords=None,
            min_word_length=1,
            n_words=top_n
        )
        
        results['basic_class'] = basic_class_results
        
        if create_csv:
            basic_class_results.to_csv(
                os.path.join(output_dir, 'basic_class_words.csv'), index=False
            )
        
        # Standard class analysis (removing stopwords, numbers, and financial terms)
        standard_class_results = _word_frequency_by_class(
            texts,
            labels,
            label_names=label_names,
            remove_stopwords=True,
            remove_numbers=True,
            remove_financial_terms=True,
            additional_stopwords=None,
            min_word_length=min_word_length,
            n_words=top_n
        )
        
        results['standard_class'] = standard_class_results
        
        if create_csv:
            standard_class_results.to_csv(
                os.path.join(output_dir, 'standard_class_words.csv'), index=False
            )
        
        # Advanced class analysis (all filters including additional stopwords)
        advanced_class_results = _word_frequency_by_class(
            texts,
            labels,
            label_names=label_names,
            remove_stopwords=True,
            remove_numbers=True,
            remove_financial_terms=True,
            additional_stopwords=ADDITIONAL_STOPS,
            min_word_length=min_word_length,
            n_words=15
        )
        
        results['advanced_class'] = advanced_class_results
        
        print("Class-specific top words (after comprehensive filtering):")
        print(advanced_class_results)
        
        if create_csv:
            advanced_class_results.to_csv(
                os.path.join(output_dir, 'advanced_class_words.csv'), index=False
            )
    
    print("\nComprehensive vocabulary analysis complete!")
    return results

# Example usage
if __name__ == "__main__":
    if external_imports:
        # Load data from the module if available
        print("Loading Reuters dataset...")
        X_train, y_train, X_test, y_test, label_names = load_data(10)  # Use top 10 classes
        print(f"Loaded {len(X_train)} training documents across {len(label_names)} classes")
        
        # Run comprehensive analysis
        analysis_results = comprehensive_analysis(
            texts=X_train,
            labels=y_train,
            label_names=label_names,
            output_dir='vocabulary_analysis',
            min_word_length=3,
            top_n=30,
            create_visualizations=True,
            create_csv=True
        )
    else:
        # Fallback to sample data
        print("Using sample data for demonstration...")
        sample_texts = [
            "The company reported earnings of 5.2 mln dlrs or 45 cts vs 3.7 mln dlrs in Q1.",
            "Oil prices increased 2 pct to 65 dlrs per barrel, according to Reuters.",
            "The bank announced a rate increase of 25 basis points to 4.5 pct.",
            "Net income fell to 1.2 billion from 1.5 billion last year."
        ]
        sample_labels = ["earn", "crude", "interest", "earn"]
        sample_label_names = ["earn", "crude", "interest"]
        
        # Run comprehensive analysis on sample data
        analysis_results = comprehensive_analysis(
            texts=sample_texts,
            labels=sample_labels,
            label_names=sample_label_names,
            output_dir='vocabulary_analysis',
            min_word_length=3
        ) 