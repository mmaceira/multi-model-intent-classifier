"""Dataset exploration utilities for text classification experiments."""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define common English stopwords
STOPWORDS = {
    "the",
    "of",
    "to",
    "in",
    "and",
    "a",
    "for",
    "it",
    "on",
    "its",
    "with",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "be",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "an",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "over",
    "under",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "said",
}

# Extended financial/numeric terms to filter
FINANCIAL_TERMS = {
    "dlrs",
    "mln",
    "pct",
    "cts",
    "shr",
    "lt",
    "billion",
    "tonnes",
    "bpd",
    "000",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "v",
    "vs",
    "inc",
    "corp",
}


def class_frequency(
    labels: np.ndarray,
    plot: bool = True,
    save_path: str | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Analyze and visualize class distribution.

    Args:
        labels: Array of class labels
        plot: Whether to create visualization
        save_path: Path to save plot
        top_n: Number of top classes to show

    Returns:
        Dict with:
        - counts: Series of class counts
        - proportions: Series of class proportions

    Raises:
        ValueError: Empty labels array
        TypeError: Invalid input type
        FileNotFoundError: Invalid save path
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
        bars = plt.bar(counts.index, counts.values, color="steelblue")

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )

        # Style the plot
        plt.title("Distribution of Topics in Training Set", fontsize=14)
        plt.grid(axis="y", alpha=0.3)
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()  # Always close the figure to prevent pop-ups

    return {"counts": counts, "proportions": proportions}


def length_distribution(
    texts: list[str],
    save_path: str | None = None,
    output_dir: str | None = None,
    percentiles: list[int] | None = None,
) -> dict[str, Any]:
    """Analyze text length distribution.

    Args:
        texts: List of text documents
        save_path: Path to save plot
        output_dir: Directory for output files
        percentiles: List of percentiles to calculate

    Returns:
        Dict with:
        - lengths: Array of text lengths
        - stats: Summary statistics
        - percentile_stats: Percentile information

    Raises:
        ValueError: Empty texts list
        TypeError: Invalid input type
        FileNotFoundError: Invalid save path
    """
    # Compute text lengths
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]
    lengths = np.array([len(text.split()) for text in texts])

    # Compute summary statistics
    stats = {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
    }

    # Calculate percentiles
    percentile_values = np.percentile(lengths, percentiles)
    percentile_stats = []

    for p, percentile in zip(percentiles, percentile_values, strict=False):
        percentile_stats.append({"percentile": p, "length": int(percentile)})

    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50)
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")

    # Add vertical lines for mean and median
    plt.axvline(stats["mean"], color="r", linestyle="--", label=f"Mean: {stats['mean']:.1f}")
    plt.axvline(stats["median"], color="g", linestyle="--", label=f"Median: {stats['median']:.1f}")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()  # Always close the figure to prevent pop-ups

    # Save percentile statistics to CSV if output_dir is provided
    if output_dir:
        csv_path = os.path.join(output_dir, "document_length_stats.csv")
        pd.DataFrame(percentile_stats).to_csv(csv_path, index=False)
        print("\nDocument Length Percentiles:")
        for stat in percentile_stats:
            print(f"{stat['percentile']}th percentile: {stat['length']} tokens")

    return {"lengths": lengths, "stats": stats, "percentile_stats": percentile_stats}


def _is_numeric_or_financial(word: str) -> bool:
    """Determine if a word is numeric or a financial term.

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
    if re.match(r"^[0-9,\.]+$", word):
        return True

    # Check if it starts with a number followed by other characters
    if re.match(r"^[0-9]+[a-zA-Z]*$", word):
        return True

    # Check for currency symbols
    if re.match(r"^[\$£€¥]?[0-9,\.]+$", word):
        return True

    return False


def vocabulary_analysis(
    texts: list[str],
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    remove_financial_terms: bool = False,
    min_word_length: int = 1,
    n_most_common: int = 30,
    plot: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> dict:
    """Analyze vocabulary distribution.

    Args:
        texts: List of text documents
        remove_stopwords: Remove common stopwords
        remove_numbers: Remove numeric tokens
        remove_financial_terms: Remove financial terms
        min_word_length: Minimum word length
        n_most_common: Number of top words
        plot: Create visualization
        figsize: Figure size

    Returns:
        Dict with:
        - vocab_size: Total unique words
        - word_counts: Word frequencies
        - top_words: DataFrame of common words

    Raises:
        ValueError: Invalid input
        TypeError: Invalid input type
    """
    # Tokenize and count words
    words = []
    for doc in texts:
        words.extend(re.findall(r"\w+", doc.lower()))

    # Filter by minimum word length
    if min_word_length > 1:
        words = [word for word in words if len(word) >= min_word_length]

    # Filter stopwords if requested
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS]

    # Filter numbers and financial terms if requested
    filtered_words = []
    for word in words:
        if remove_numbers and _is_numeric_or_financial(word):
            continue
        if remove_financial_terms and word in FINANCIAL_TERMS:
            continue
        filtered_words.append(word)

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Get most common words
    most_common = word_counts.most_common(n_most_common)
    top_words = pd.DataFrame(most_common, columns=["word", "count"])

    # Create visualization if requested
    if plot:
        plt.figure(figsize=figsize)
        sns.barplot(x="count", y="word", data=top_words.head(30), palette="viridis")
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
        plt.title(f"Top Words in Corpus{title_suffix}")
        plt.xlabel("Count")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.close()  # Close figure instead of showing to prevent pop-ups

    return {"vocab_size": len(word_counts), "word_counts": word_counts, "top_words": top_words}


def vocabulary_drift(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    top_k: int = 2000,
    min_freq: int = 10,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Analyze vocabulary differences between train and test sets.

    Args:
        train_texts: Training documents
        test_texts: Test documents
        top_k: Number of frequent tokens
        min_freq: Minimum token frequency
        output_path: Path to save results

    Returns:
        DataFrame with token frequencies and drift metrics

    Raises:
        ValueError: Invalid input
        TypeError: Invalid input type
        FileNotFoundError: Invalid output path
    """

    def _tokenize(s: str):
        return s.lower().split()

    train_counts = Counter(token for doc in train_texts for token in _tokenize(doc))
    test_counts = Counter(token for doc in test_texts for token in _tokenize(doc))

    # restrict to top_k in either split
    vocab = set(
        [w for w, _ in train_counts.most_common(top_k)]
        + [w for w, _ in test_counts.most_common(top_k)]
    )
    rows = []

    for w in vocab:
        tr = train_counts.get(w, 0)
        te = test_counts.get(w, 0)
        total = tr + te
        if total < min_freq:
            continue
        rows.append(
            {
                "token": w,
                "train_freq": tr / total,
                "test_freq": te / total,
                "abs_diff": abs(tr / total - te / total),
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)

    # Save results if output path provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def comprehensive_analysis(
    texts: list[str],
    labels: list[str] | None = None,
    label_names: list[str] | None = None,
    output_dir: str = ".",
    min_word_length: int = 3,
    top_n: int = 30,
    create_visualizations: bool = True,
    create_csv: bool = True,
) -> dict:
    """Perform comprehensive text analysis.

    Args:
        texts: List of text documents
        labels: Optional class labels
        label_names: Optional label names
        output_dir: Output directory
        min_word_length: Minimum word length
        top_n: Number of top words
        create_visualizations: Create plots
        create_csv: Save CSV files

    Returns:
        Dict with results from all analyses

    Raises:
        ValueError: Invalid input
        TypeError: Invalid input type
        FileNotFoundError: Invalid output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"Running comprehensive vocabulary analysis on {len(texts)} documents...")

    # 1. Basic analysis (only removing stopwords)
    print("\n======= BASIC FILTERING =======")
    basic_results = vocabulary_analysis(
        texts,
        remove_stopwords=True,
        remove_numbers=False,
        remove_financial_terms=False,
        min_word_length=1,
        n_most_common=top_n,
        plot=False,
    )

    results["basic"] = basic_results
    print(f"Total unique words (basic filtering): {basic_results['vocab_size']:,}")

    if create_csv:
        pd.DataFrame(
            basic_results["word_counts"].most_common(top_n), columns=["word", "count"]
        ).to_csv(os.path.join(output_dir, "basic_filtering.csv"), index=False)

    # 2. Standard analysis (removing stopwords, numbers, and financial terms)
    print("\n======= STANDARD FILTERING =======")
    standard_results = vocabulary_analysis(
        texts,
        remove_stopwords=True,
        remove_numbers=True,
        remove_financial_terms=True,
        min_word_length=min_word_length,
        n_most_common=top_n,
        plot=False,
    )

    results["standard"] = standard_results
    print(f"Total unique words (standard filtering): {standard_results['vocab_size']:,}")

    if create_csv:
        pd.DataFrame(
            standard_results["word_counts"].most_common(top_n), columns=["word", "count"]
        ).to_csv(os.path.join(output_dir, "standard_filtering.csv"), index=False)

    # 3. Advanced analysis (all filters including additional stopwords)
    print("\n======= ADVANCED FILTERING =======")
    advanced_results = vocabulary_analysis(
        texts,
        remove_stopwords=True,
        remove_numbers=True,
        remove_financial_terms=True,
        min_word_length=min_word_length,
        n_most_common=top_n,
        plot=False,
    )

    results["advanced"] = advanced_results
    print(f"Total unique words (advanced filtering): {advanced_results['vocab_size']:,}")

    if create_csv:
        pd.DataFrame(
            advanced_results["word_counts"].most_common(top_n), columns=["word", "count"]
        ).to_csv(os.path.join(output_dir, "advanced_filtering.csv"), index=False)

    # Create top words visualization
    if create_visualizations:
        top_words = pd.DataFrame(
            advanced_results["word_counts"].most_common(25), columns=["word", "count"]
        )
        plt.figure(figsize=(12, 10))
        plt.barh(top_words["word"], top_words["count"], color="steelblue")
        plt.title("Top 25 Words After Comprehensive Filtering")
        plt.xlabel("Count")
        plt.ylabel("Word")
        plt.gca().invert_yaxis()  # Put the largest at the top
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_words_comprehensive.png"))
        plt.close()
        print("Visualization saved to 'top_words_comprehensive.png'")

    # Class-specific analysis if labels are provided
    if labels is not None and label_names is not None:
        print("\n======= CLASS-SPECIFIC ANALYSIS =======")

        # Create a DataFrame to store class-specific word frequencies
        class_word_freqs = {}

        for class_name in label_names:
            # Filter texts for this class
            # Handle both single-label (str) and multi-label (list/str with commas) formats
            class_texts = []
            for text, label in zip(texts, labels, strict=False):
                # Handle multilabel: if label is a list or comma-separated string
                if isinstance(label, (list, tuple)):
                    if class_name in label:
                        class_texts.append(text)
                elif isinstance(label, str) and "," in label:
                    # Comma-separated multilabel string
                    label_set = {tag.strip() for tag in label.split(",") if tag.strip()}
                    if class_name in label_set:
                        class_texts.append(text)
                elif label == class_name:
                    # Single-label match
                    class_texts.append(text)

            if not class_texts:
                continue

            # Analyze vocabulary for this class
            class_analysis = vocabulary_analysis(
                class_texts,
                remove_stopwords=True,
                remove_numbers=True,
                remove_financial_terms=True,
                min_word_length=min_word_length,
                n_most_common=top_n,
                plot=False,
            )

            # Store top words for this class
            class_word_freqs[class_name] = [
                f"{word} ({count:,})"
                for word, count in class_analysis["word_counts"].most_common(top_n)
            ]

        # Convert to DataFrame
        # Handle empty class_word_freqs (can happen with tiny datasets or multi-label data)
        if class_word_freqs:
            max_length = max(len(words) for words in class_word_freqs.values())
            for class_name in class_word_freqs:
                class_word_freqs[class_name] += [""] * (
                    max_length - len(class_word_freqs[class_name])
                )
            results["advanced_class"] = pd.DataFrame(class_word_freqs)
        else:
            # Create empty DataFrame if no class-specific data
            results["advanced_class"] = pd.DataFrame()

        if create_csv:
            # Guard rail: ensure we write at least a header if DataFrame is empty
            if results["advanced_class"].empty:
                # Create a proper structure: one row per class with top words
                # If no class-specific data, create a single info row
                info_df = pd.DataFrame(
                    [
                        {
                            "class": "__info__",
                            "word": "No tokens passed the advanced filters",
                            "count": 0,
                        }
                    ]
                )
                results["advanced_class"] = info_df
            else:
                # Convert the wide format (columns = classes) to long format
                # Each class column contains strings like "word (count)"
                long_rows = []
                for class_name in results["advanced_class"].columns:
                    words_col = results["advanced_class"][class_name]
                    for word_str in words_col:
                        if word_str and word_str.strip():
                            # Parse "word (count)" format
                            import re

                            match = re.match(r"^(.+?)\s*\((\d+)\)$", str(word_str))
                            if match:
                                word, count_str = match.groups()
                                long_rows.append(
                                    {
                                        "class": class_name,
                                        "word": word.strip(),
                                        "count": int(count_str),
                                    }
                                )
                            else:
                                # Fallback: treat as word without count
                                long_rows.append(
                                    {"class": class_name, "word": str(word_str).strip(), "count": 0}
                                )
                if long_rows:
                    results["advanced_class"] = pd.DataFrame(long_rows)
                else:
                    # Fallback if parsing fails
                    results["advanced_class"] = pd.DataFrame(
                        [
                            {
                                "class": "__info__",
                                "word": "Failed to parse class-specific words",
                                "count": 0,
                            }
                        ]
                    )
            results["advanced_class"].to_csv(  # type: ignore[attr-defined]
                os.path.join(output_dir, "advanced_class_words.csv"), index=False
            )

    return results
