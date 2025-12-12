#!/usr/bin/env python
"""
Test script for config-based dataset loaders.

This script tests loading datasets using the new config-based loader system.
It verifies that:
- Datasets can be loaded from config/dataset/{name}/loader.yaml
- Single-label and multi-label datasets work correctly
- All splits (train/val/test) are properly loaded

Run with: uv run python scripts/test_config_loaders.py
"""

import sys

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()
sys.path.insert(0, str(repo_root))

from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.utils.label_utils import is_multilabel


def test_dataset(dataset_name: str, is_multilabel_dataset: bool, use_oos: bool = False):
    """Test loading a dataset."""
    print("=" * 70)
    print(f"Testing Dataset: {dataset_name}")
    print(f"Multi-label: {is_multilabel_dataset}")
    print("=" * 70)

    try:
        # Load with small limits for quick testing
        print(f"\n📦 Loading {dataset_name} dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name=dataset_name,
            use_oos=use_oos,
            max_train_samples=100,
            max_val_samples=20,
            max_test_samples=30,
            max_classes=10,  # Limit classes for quick testing
            seed=42,
        )

        print("\n✅ Dataset loaded successfully!")
        print("\n📊 Dataset Statistics:")
        print(f"   - Training samples:   {len(X_train):,}")
        print(f"   - Validation samples: {len(X_val):,}")
        print(f"   - Test samples:        {len(X_test):,}")
        print(f"   - Number of classes:   {len(classes)}")
        print(f"   - Classes (first 10): {classes[:10]}")

        # Verify label format
        print("\n🔍 Verifying label format...")
        is_multi = is_multilabel(y_train)
        print(f"   - Is multi-label? {is_multi}")
        if is_multi != is_multilabel_dataset:
            print(
                f"   ❌ ERROR: Expected {'multi' if is_multilabel_dataset else 'single'}-label, but got {'multi' if is_multi else 'single'}-label!"
            )
            return False
        print("   ✅ Labels are in correct format")

        # Show sample data
        print("\n📝 Sample Data:")
        print(f"   Text: {X_train[0][:80]}...")
        print(f"   Label: {y_train[0]}")
        if len(X_train) > 1:
            print(f"\n   Text 2: {X_train[1][:80]}...")
            print(f"   Label 2: {y_train[1]}")

        # Verify all splits have data
        if len(X_train) == 0:
            print("   ❌ ERROR: Training set is empty!")
            return False
        if len(X_val) == 0:
            print("   ⚠️  WARNING: Validation set is empty")
        if len(X_test) == 0:
            print("   ❌ ERROR: Test set is empty!")
            return False

        print("\n✅ All checks passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test all available datasets."""
    print("=" * 70)
    print("Testing Config-Based Dataset Loaders")
    print("=" * 70)

    results = []

    # Test CLINC150 (single-label)
    print("\n" + "=" * 70)
    results.append(
        (
            "clinc150 (single-label)",
            test_dataset("clinc150", is_multilabel_dataset=False, use_oos=False),
        )
    )

    # Test CLINC150 with OOS
    print("\n" + "=" * 70)
    results.append(
        ("clinc150 (with OOS)", test_dataset("clinc150", is_multilabel_dataset=False, use_oos=True))
    )

    # Test NLU++ (multi-label)
    print("\n" + "=" * 70)
    results.append(
        (
            "nlu_plus (multi-label)",
            test_dataset("nlu_plus", is_multilabel_dataset=True, use_oos=False),
        )
    )

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
