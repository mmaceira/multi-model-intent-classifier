#!/usr/bin/env python
"""
Intent Classification - Data Loading

This script handles the initial data loading and preprocessing for intent
classification tasks. It supports multiple datasets configured via config files.
"""

import logging
import os
import sys
import warnings

# Import path utilities
from intent_classifier.utils.paths import get_repo_root  # noqa: E402

# Get repo root and add to path for config imports (config is not part of the installed package)
repo_root = get_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import config setup
from config.notebook_setup import (  # noqa: E402
    DATA_EXPLORATION_DIR,
    config_vars,
)

# Import dataset and exploration modules
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.exploration import class_frequency, length_distribution  # noqa: E402

# Configure logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    """Main function to load and validate the dataset."""

    dataset_name = config_vars.get("DATASET_NAME", "clinc150")
    print("=" * 60)
    print(f"Intent Classification - Data Loading ({dataset_name})")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name=config_vars.get("DATASET_NAME", "clinc150"),
        use_oos=config_vars.get("DATASET_USE_OOS", False),
        multilabel=config_vars.get("DATASET_MULTILABEL", False),
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
    # For multi-label data, flatten labels first (count all tags across all samples)
    from intent_classifier.utils.label_utils import is_multilabel

    if is_multilabel(y_train_merged):
        # Multi-label: flatten all tags for frequency analysis
        flattened_labels = []
        for label_list in y_train_merged:
            flattened_labels.extend(label_list)
        labels_for_analysis = flattened_labels
    else:
        # Single-label: use as-is
        labels_for_analysis = y_train_merged

    class_frequency(
        labels_for_analysis,
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
