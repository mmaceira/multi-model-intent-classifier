#!/usr/bin/env python
"""
CLINC150 Intent Classification - Exploratory Data Analysis

This script performs detailed exploratory data analysis on the CLINC150 intent
classification dataset. We'll analyze various aspects of the data to better
understand its characteristics and potential challenges.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

# Import config setup
from config.notebook_setup import *

# Import dataset and exploration modules
from src.datasets.dataset import get_dataset
from src.exploration import (
    class_frequency,
    comprehensive_analysis,
    length_distribution,
    vocabulary_drift,
)

# Configure logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    """Main function to perform exploratory data analysis."""

    print("=" * 60)
    print("CLINC150 Intent Classification - Exploratory Data Analysis")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        use_oos=config_vars.get("DATASET_USE_OOS", False),
        max_classes=config_vars.get("DATASET_MAX_CLASSES", None),
        max_train_samples=config_vars.get("DATASET_MAX_TRAIN_SAMPLES", None),
        max_test_samples=config_vars.get("DATASET_MAX_TEST_SAMPLES", None),
        seed=config_vars.get("GENERAL_SEED", 42),
    )

    print(f"Loaded {len(X_train)} training utterances with {len(classes)} intent classes")
    print(f"Test set contains {len(X_test)} utterances")

    # 1. Intent Distribution Analysis
    print("\n" + "=" * 60)
    print("1. Intent Distribution Analysis")
    print("=" * 60)
    print("\nAnalyzing the distribution of intents in our dataset to understand class imbalance...")

    class_stats = class_frequency(
        labels=y_train,
        plot=True,
        save_path=os.path.join(DATA_EXPLORATION_DIR, "class_distribution.png"),
        top_n=len(classes),  # Show all intents
    )

    # 2. Utterance Length Analysis
    print("\n" + "=" * 60)
    print("2. Utterance Length Analysis")
    print("=" * 60)
    print("\nUnderstanding the length distribution of utterances...")

    plt.figure(figsize=(10, 6))
    stats = length_distribution(
        X_train,
        save_path=os.path.join(DATA_EXPLORATION_DIR, "document_length_distribution.png"),
        output_dir=DATA_EXPLORATION_DIR,
    )
    print(f"Mean length: {stats['stats']['mean']:.1f} tokens, median: {stats['stats']['median']}")

    # Calculate length percentiles
    lengths = [len(doc.split()) for doc in X_train]
    percentiles = np.percentile(lengths, [25, 50, 75, 90, 95, 99])
    print("\nUtterance Length Percentiles:")
    length_stats = []
    for p, percentile in zip([25, 50, 75, 90, 95, 99], percentiles, strict=False):
        print(f"{p}th percentile: {percentile:.0f} tokens")
        length_stats.append({"percentile": p, "length": int(percentile)})

    # Save utterance length statistics to CSV
    pd.DataFrame(length_stats).to_csv(
        os.path.join(DATA_EXPLORATION_DIR, "document_length_stats.csv"), index=False
    )

    # 3. Vocabulary Analysis
    print("\n" + "=" * 60)
    print("3. Vocabulary Analysis")
    print("=" * 60)
    print("\nExamining the vocabulary characteristics of our dataset...")

    # Run comprehensive analysis with custom settings
    analysis_results = comprehensive_analysis(
        texts=X_train,  # Your text utterances
        labels=y_train,  # Intent labels for intent-specific analysis
        label_names=classes,  # Names of the intents
        output_dir=os.path.join(DATA_EXPLORATION_DIR),  # Output directory for results
        min_word_length=3,  # Minimum word length (filters short tokens)
        top_n=50,  # Number of top words to analyze
        create_visualizations=True,  # Create and save plots
        create_csv=True,  # Save results to CSV files
    )

    print("\nAnalysis complete! Results saved to the output directory.")

    # Example: Get the top 5 words after advanced filtering
    print("\nTop 5 words (advanced filtering):")
    for word, count in analysis_results["advanced"]["word_counts"].most_common(5):
        print(f"  {word}: {count:,}")

    # Example: Get the most distinctive word for each intent
    print("\nMost distinctive word by intent (advanced filtering):")
    for intent_name in classes:
        if intent_name in analysis_results["advanced_class"].columns:
            top_word = analysis_results["advanced_class"][intent_name][0]
            if top_word:  # Check that it's not an empty string
                print(f"  {intent_name}: {top_word}")

    # 4. Vocabulary Drift Analysis
    print("\n" + "=" * 60)
    print("4. Vocabulary Drift Analysis")
    print("=" * 60)
    print("\nAnalyzing vocabulary drift between train and test sets...")

    vocab_drift = vocabulary_drift(
        train_texts=X_train,
        test_texts=X_test,
        top_k=2000,
        min_freq=10,  # Lower threshold for intent classification (utterances are shorter)
        output_path=Path(RESULTS_DIR) / "vocabulary_drift.csv",
    )

    # Display top 20 tokens with highest drift
    print("\nTop 20 tokens with highest frequency drift:")
    print(vocab_drift.head(20))

    # Display intent distribution statistics
    print("\nIntent distribution statistics:")
    print(class_stats["counts"])

    print("\n✅ Exploratory analysis complete!")
    print(f"Results saved to: {DATA_EXPLORATION_DIR}")


if __name__ == "__main__":
    main()
