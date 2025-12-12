#!/usr/bin/env python
"""
Test script for multi-label dataset loader.

This script loads a multi-label dataset and prints basic statistics.

Run with: uv run python scripts/test_tandem_go_dataset.py
(Note: This script tests multi-label dataset loading functionality)
"""

import sys

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()
sys.path.insert(0, str(repo_root))

# This import must come after sys.path manipulation
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402


def main():
    """Test loading a multi-label dataset."""
    print("=" * 70)
    print("Testing Multi-Label Dataset Loader")
    print("=" * 70)

    # Load dataset
    print("\n📦 Loading multi-label dataset...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name="nlu_plus",
            seed=42,
        )

        print("\n✅ Dataset loaded successfully!")
        print("\n📊 Dataset Statistics:")
        print(f"   - Training samples:   {len(X_train):,}")
        print(f"   - Validation samples: {len(X_val):,}")
        print(f"   - Test samples:        {len(X_test):,}")
        print(f"   - Total samples:       {len(X_train) + len(X_val) + len(X_test):,}")
        print(f"   - Number of classes:   {len(classes)}")

        # Show class distribution
        print("\n📈 Class Distribution (top 10 classes):")
        from collections import Counter

        all_labels = y_train + y_val + y_test
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common(10):
            print(f"   - {label:30s}: {count:4,} samples")

        # Show sample data
        print("\n📝 Sample Data:")
        print(f"   Example text (first 100 chars): {X_train[0][:100]}...")
        print(f"   Label: {y_train[0]}")
        if len(X_train) > 1:
            print(f"\n   Example text 2 (first 100 chars): {X_train[1][:100]}...")
            print(f"   Label: {y_train[1]}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
