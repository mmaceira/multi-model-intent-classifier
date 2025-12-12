#!/usr/bin/env python
"""
Test script for the NLU++ dataset loader.

This script loads the NLU++ dataset and prints basic statistics.
It verifies that the dataset loads correctly in multilabel format.

Run with: uv run python scripts/test_nlu_plus_dataset.py
"""

import sys
from collections import Counter

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()
sys.path.insert(0, str(repo_root))

from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.utils.label_utils import is_multilabel


def main():
    """Test loading the NLU++ dataset."""
    print("=" * 70)
    print("Testing NLU++ Dataset Loader")
    print("=" * 70)

    # Load dataset
    print("\n📦 Loading NLU++ dataset...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name="nlu_plus",
            seed=42,
            max_train_samples=100,  # Limit for quick testing
            max_test_samples=30,
            max_val_samples=20,
        )

        print("\n✅ Dataset loaded successfully!")
        print("\n📊 Dataset Statistics:")
        print(f"   - Training samples:   {len(X_train):,}")
        print(f"   - Validation samples: {len(X_val):,}")
        print(f"   - Test samples:        {len(X_test):,}")
        print(f"   - Total samples:       {len(X_train) + len(X_val) + len(X_test):,}")
        print(f"   - Number of classes:   {len(classes)}")
        print(f"   - Classes (first 10): {classes[:10]}")

        # Verify multi-label format
        print("\n🔍 Verifying label format...")
        is_multi = is_multilabel(y_train)
        print(f"   - Is multi-label? {is_multi}")
        if not is_multi:
            print("   ❌ ERROR: Expected multi-label format, but got single-label!")
            sys.exit(1)
        print("   ✅ Labels are in multi-label format (list of lists)")

        # Show sample labels
        print("\n📝 Sample Labels:")
        print(f"   y_train[0]: {y_train[0]} (type: {type(y_train[0])}, length: {len(y_train[0])})")
        if len(y_train) > 1:
            print(
                f"   y_train[1]: {y_train[1]} (type: {type(y_train[1])}, length: {len(y_train[1])})"
            )
        if len(y_train) > 2:
            print(
                f"   y_train[2]: {y_train[2]} (type: {type(y_train[2])}, length: {len(y_train[2])})"
            )

        # Count labels per sample
        labels_per_sample = [len(labels) for labels in y_train]
        if labels_per_sample:
            import numpy as np

            print("\n📈 Label Statistics:")
            print(f"   - Average labels per sample: {np.mean(labels_per_sample):.2f}")
            print(f"   - Min labels per sample: {min(labels_per_sample)}")
            print(f"   - Max labels per sample: {max(labels_per_sample)}")

        # Show class distribution (flatten all labels)
        print("\n📈 Class Distribution (top 10 classes):")
        all_labels = []
        for label_list in y_train + y_val + y_test:
            all_labels.extend(label_list)
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common(10):
            print(f"   - {label:40s}: {count:4,} samples")

        # Show sample data
        print("\n📝 Sample Data:")
        print(f"   Example text (first 100 chars): {X_train[0][:100]}...")
        print(f"   Labels: {y_train[0]} ({len(y_train[0])} labels)")
        if len(X_train) > 1:
            print(f"\n   Example text 2 (first 100 chars): {X_train[1][:100]}...")
            print(f"   Labels: {y_train[1]} ({len(y_train[1])} labels)")

        print("\n✅ Test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
