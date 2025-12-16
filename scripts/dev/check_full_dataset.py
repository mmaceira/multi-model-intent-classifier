#!/usr/bin/env python
"""
Script to check the full CLINC150 dataset and verify the full_dataset config.

This script loads the dataset with no limits to see what the full dataset contains.
"""

from collections import Counter

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()

from intent_classifier.datasets.dataset import get_dataset  # noqa: E402


def main():
    """Load and analyze the full CLINC150 dataset."""

    print("=" * 70)
    print("CLINC150 Full Dataset Analysis")
    print("=" * 70)

    print("\nLoading full dataset (no limits)...")
    print("Configuration:")
    print("  - use_oos: False")
    print("  - max_classes: None (all classes)")
    print("  - max_train_samples: None (all samples)")
    print("  - max_test_samples: None (all samples)")

    # Load with no limits - this is what config_full_dataset.yaml should do
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        use_oos=False,
        max_classes=None,
        max_train_samples=None,
        max_test_samples=None,
        seed=42,
    )

    # Merge validation into training for analysis
    X_train = X_train + X_val
    y_train = y_train + y_val

    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)

    print("\n📊 Training Set:")
    print(f"  - Total samples: {len(X_train):,}")
    print(f"  - Unique classes: {len(set(y_train))}")

    print("\n📊 Test Set:")
    print(f"  - Total samples: {len(X_test):,}")
    print(f"  - Unique classes: {len(set(y_test))}")

    print("\n📊 Overall:")
    print(f"  - Total classes: {len(classes)}")
    print(f"  - Classes in train: {len(set(y_train))}")
    print(f"  - Classes in test: {len(set(y_test))}")
    print(f"  - Classes in both: {len(set(y_train) & set(y_test))}")
    print(f"  - Classes only in train: {len(set(y_train) - set(y_test))}")
    print(f"  - Classes only in test: {len(set(y_test) - set(y_train))}")

    # Class distribution
    train_class_counts = Counter(y_train)
    test_class_counts = Counter(y_test)

    print("\n📊 Class Distribution:")
    print(f"  - Min samples per class (train): {min(train_class_counts.values())}")
    print(f"  - Max samples per class (train): {max(train_class_counts.values())}")
    mean_train = sum(train_class_counts.values()) / len(train_class_counts)
    print(f"  - Mean samples per class (train): {mean_train:.1f}")
    print(f"  - Min samples per class (test): {min(test_class_counts.values())}")
    print(f"  - Max samples per class (test): {max(test_class_counts.values())}")
    mean_test = sum(test_class_counts.values()) / len(test_class_counts)
    print(f"  - Mean samples per class (test): {mean_test:.1f}")

    # Sample lengths
    train_lengths = [len(text.split()) for text in X_train]
    test_lengths = [len(text.split()) for text in X_test]

    print("\n📊 Text Length Statistics:")
    print(
        f"  - Train - Mean: {sum(train_lengths) / len(train_lengths):.1f}, "
        f"Median: {sorted(train_lengths)[len(train_lengths) // 2]}, "
        f"Min: {min(train_lengths)}, Max: {max(train_lengths)}"
    )
    print(
        f"  - Test - Mean: {sum(test_lengths) / len(test_lengths):.1f}, "
        f"Median: {sorted(test_lengths)[len(test_lengths) // 2]}, "
        f"Min: {min(test_lengths)}, Max: {max(test_lengths)}"
    )

    # Show first few classes
    print("\n📊 First 10 Classes (alphabetically):")
    for i, cls in enumerate(sorted(classes)[:10], 1):
        train_count = train_class_counts.get(cls, 0)
        test_count = test_class_counts.get(cls, 0)
        print(f"  {i:2d}. {cls:30s} - Train: {train_count:4d}, Test: {test_count:4d}")

    print("\n✅ Full dataset loaded successfully!")
    print("\n💡 The config_full_dataset.yaml will use:")
    print(f"  - All {len(classes)} classes")
    print(f"  - All {len(X_train):,} training samples")
    print(f"  - All {len(X_test):,} test samples")
    print("  - No OOS examples (use_oos: false)")


if __name__ == "__main__":
    main()
