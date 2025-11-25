#!/usr/bin/env python
"""
CLINC150 Intent Classification - Data Loading

This script handles the initial data loading and preprocessing for the CLINC150
intent classification task. We'll be using the CLINC150 dataset, a collection of
user utterances labeled with intent categories.

Dataset Overview:
The CLINC150 dataset is a collection of user utterances that have been labeled
with intent categories. Key features:
- 150 in-scope intents across 10 domains (banking, credit cards, work, etc.)
- Real-world user queries and commands
- Out-of-scope (OOS) examples available as an optional class
- Benchmark dataset for intent classification research
"""

import logging
import os
import sys
import warnings
from pathlib import Path

# Infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

# Import config setup
from config.notebook_setup import (  # noqa: E402
    DATA_EXPLORATION_DIR,
    config_vars,
)

# Import dataset and exploration modules
from src.datasets.dataset import get_dataset  # noqa: E402
from src.exploration import class_frequency, length_distribution  # noqa: E402

# Configure logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    """Main function to load and validate the dataset."""

    print("=" * 60)
    print("CLINC150 Intent Classification - Data Loading")
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
    X_train_merged = X_train + X_val
    y_train_merged = y_train + y_val

    print(
        f"Loaded {len(X_train)} training, {len(X_val)} validation, "
        f"{len(X_test)} test utterances"
    )
    print(
        f"Total training (train+val merged for analysis): "
        f"{len(X_train_merged)} utterances with {len(classes)} intent classes"
    )

    # Minimal check on the data
    print("\n" + "=" * 60)
    print("Data Validation")
    print("=" * 60)
    print("\nExamining the distribution of intents and utterance lengths...")

    # Intent distribution plot
    class_frequency(
        y_train_merged,
        top_n=20,
        plot=True,
        save_path=os.path.join(DATA_EXPLORATION_DIR, "class_distribution_validation.png"),
    )

    # Utterance length distribution
    stats = length_distribution(
        X_train_merged,
        save_path=os.path.join(DATA_EXPLORATION_DIR, "document_length_distribution_validation.png"),
        output_dir=DATA_EXPLORATION_DIR,
    )
    print(f"\nMean length: {stats['stats']['mean']:.1f} tokens, median: {stats['stats']['median']}")

    print("\n✅ Data loading complete!")
    print(f"Results saved to: {DATA_EXPLORATION_DIR}")


if __name__ == "__main__":
    main()
