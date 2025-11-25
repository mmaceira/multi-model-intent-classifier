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
from config.notebook_setup import (  # noqa: E402
    DATA_EXPLORATION_DIR,
    RESULTS_DIR,
    config_vars,
)

# Import dataset and exploration modules
from src.datasets.dataset import get_dataset  # noqa: E402
from src.exploration import (  # noqa: E402
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
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        use_oos=config_vars.get("DATASET_USE_OOS", False),
        max_classes=config_vars.get("DATASET_MAX_CLASSES", None),
        max_train_samples=config_vars.get("DATASET_MAX_TRAIN_SAMPLES", None),
        max_test_samples=config_vars.get("DATASET_MAX_TEST_SAMPLES", None),
        seed=config_vars.get("GENERAL_SEED", 42),
    )

    # Merge validation into training for exploratory analysis
    # Note: This is for analysis only. The training pipeline (03_model_training.py)
    # keeps validation separate for proper ML practices (hyperparameter tuning, early stopping).
    X_train = X_train + X_val
    y_train = y_train + y_val

    print(
        f"Loaded {len(X_train)} training utterances "
        f"(train+val merged for analysis) with {len(classes)} intent classes"
    )
    print(f"Test set contains {len(X_test)} utterances")

    # 0. Dataset Overview Statistics
    print("\n" + "=" * 60)
    print("0. Dataset Overview")
    print("=" * 60)

    from collections import Counter

    train_class_counts = Counter(y_train)
    test_class_counts = Counter(y_test)

    # Create summary statistics
    summary_stats = {
        "dataset_name": config_vars.get("DATASET_NAME", "clinc150"),
        "use_oos": config_vars.get("DATASET_USE_OOS", False),
        "max_classes": config_vars.get("DATASET_MAX_CLASSES"),
        "max_train_samples": config_vars.get("DATASET_MAX_TRAIN_SAMPLES"),
        "max_test_samples": config_vars.get("DATASET_MAX_TEST_SAMPLES"),
        "total_classes": len(classes),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_classes": len(set(y_train)),
        "test_classes": len(set(y_test)),
        "classes_in_both": len(set(y_train) & set(y_test)),
        "classes_only_train": len(set(y_train) - set(y_test)),
        "classes_only_test": len(set(y_test) - set(y_train)),
        "min_samples_per_class_train": (
            min(train_class_counts.values()) if train_class_counts else 0
        ),
        "max_samples_per_class_train": (
            max(train_class_counts.values()) if train_class_counts else 0
        ),
        "mean_samples_per_class_train": (
            sum(train_class_counts.values()) / len(train_class_counts) if train_class_counts else 0
        ),
        "min_samples_per_class_test": min(test_class_counts.values()) if test_class_counts else 0,
        "max_samples_per_class_test": max(test_class_counts.values()) if test_class_counts else 0,
        "mean_samples_per_class_test": (
            sum(test_class_counts.values()) / len(test_class_counts) if test_class_counts else 0
        ),
    }

    print("\nDataset Configuration:")
    print(f"  - Dataset: {summary_stats['dataset_name']}")
    print(f"  - Use OOS: {summary_stats['use_oos']}")
    if summary_stats["max_classes"]:
        print(f"  - Max classes: {summary_stats['max_classes']}")
    if summary_stats["max_train_samples"]:
        print(f"  - Max train samples: {summary_stats['max_train_samples']}")
    if summary_stats["max_test_samples"]:
        print(f"  - Max test samples: {summary_stats['max_test_samples']}")

    print("\nDataset Statistics:")
    print(f"  - Total classes: {summary_stats['total_classes']}")
    print(f"  - Training samples: {summary_stats['train_samples']:,}")
    print(f"  - Test samples: {summary_stats['test_samples']:,}")
    print(f"  - Classes in train: {summary_stats['train_classes']}")
    print(f"  - Classes in test: {summary_stats['test_classes']}")
    print(f"  - Classes in both: {summary_stats['classes_in_both']}")

    print("\nClass Distribution:")
    print(
        f"  - Train - Min: {summary_stats['min_samples_per_class_train']}, "
        f"Max: {summary_stats['max_samples_per_class_train']}, "
        f"Mean: {summary_stats['mean_samples_per_class_train']:.1f}"
    )
    print(
        f"  - Test - Min: {summary_stats['min_samples_per_class_test']}, "
        f"Max: {summary_stats['max_samples_per_class_test']}, "
        f"Mean: {summary_stats['mean_samples_per_class_test']:.1f}"
    )

    # Save summary statistics to CSV
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(DATA_EXPLORATION_DIR, "dataset_summary.csv"), index=False)

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

    # Calculate length percentiles for both train and test
    train_lengths = [len(doc.split()) for doc in X_train]
    test_lengths = [len(doc.split()) for doc in X_test]

    train_percentiles = np.percentile(train_lengths, [25, 50, 75, 90, 95, 99])
    test_percentiles = np.percentile(test_lengths, [25, 50, 75, 90, 95, 99])

    print("\nUtterance Length Percentiles (Train):")
    length_stats = []
    for p, percentile in zip([25, 50, 75, 90, 95, 99], train_percentiles, strict=False):
        print(f"{p}th percentile: {percentile:.0f} tokens")
        length_stats.append({"split": "train", "percentile": p, "length": int(percentile)})

    print("\nUtterance Length Percentiles (Test):")
    for p, percentile in zip([25, 50, 75, 90, 95, 99], test_percentiles, strict=False):
        print(f"{p}th percentile: {percentile:.0f} tokens")
        length_stats.append({"split": "test", "percentile": p, "length": int(percentile)})

    # Add summary statistics
    length_stats.append(
        {"split": "train", "percentile": "mean", "length": float(np.mean(train_lengths))}
    )
    length_stats.append(
        {"split": "train", "percentile": "median", "length": float(np.median(train_lengths))}
    )
    length_stats.append(
        {"split": "train", "percentile": "std", "length": float(np.std(train_lengths))}
    )
    length_stats.append(
        {"split": "test", "percentile": "mean", "length": float(np.mean(test_lengths))}
    )
    length_stats.append(
        {"split": "test", "percentile": "median", "length": float(np.median(test_lengths))}
    )
    length_stats.append(
        {"split": "test", "percentile": "std", "length": float(np.std(test_lengths))}
    )

    # Save utterance length statistics to CSV
    pd.DataFrame(length_stats).to_csv(
        os.path.join(DATA_EXPLORATION_DIR, "document_length_stats.csv"), index=False
    )

    print("\nTrain/Test Length Comparison:")
    print(
        f"  - Train - Mean: {np.mean(train_lengths):.1f}, Median: {np.median(train_lengths):.1f}, "
        f"Min: {min(train_lengths)}, Max: {max(train_lengths)}"
    )
    print(
        f"  - Test - Mean: {np.mean(test_lengths):.1f}, Median: {np.median(test_lengths):.1f}, "
        f"Min: {min(test_lengths)}, Max: {max(test_lengths)}"
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

    # 5. Train/Test Class Distribution Comparison
    print("\n" + "=" * 60)
    print("5. Train/Test Class Distribution Comparison")
    print("=" * 60)

    # Create comparison DataFrame
    comparison_data = []
    for cls in sorted(classes):
        train_count = train_class_counts.get(cls, 0)
        test_count = test_class_counts.get(cls, 0)
        total = train_count + test_count
        train_ratio = train_count / total if total > 0 else 0
        test_ratio = test_count / total if total > 0 else 0

        comparison_data.append(
            {
                "class": cls,
                "train_count": train_count,
                "test_count": test_count,
                "total_count": total,
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "ratio_diff": abs(train_ratio - test_ratio),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("total_count", ascending=False)

    # Save comparison to CSV
    comparison_df.to_csv(
        os.path.join(DATA_EXPLORATION_DIR, "train_test_class_comparison.csv"), index=False
    )

    print("\nClass distribution comparison saved to CSV.")
    print(f"Total classes: {len(comparison_df)}")
    print(
        f"Classes with perfect balance (80/20 split): "
        f"{sum((comparison_df['train_ratio'] > 0.79) & (comparison_df['train_ratio'] < 0.81))}"
    )

    # Show classes with largest imbalance
    print("\nTop 10 classes by total samples:")
    print(
        comparison_df.head(10)[["class", "train_count", "test_count", "total_count"]].to_string(
            index=False
        )
    )

    if len(comparison_df) > 10:
        print("\nBottom 10 classes by total samples:")
        print(
            comparison_df.tail(10)[["class", "train_count", "test_count", "total_count"]].to_string(
                index=False
            )
        )

    print("\n✅ Exploratory analysis complete!")
    print(f"Results saved to: {DATA_EXPLORATION_DIR}")


if __name__ == "__main__":
    main()
