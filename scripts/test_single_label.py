#!/usr/bin/env python
"""
Test script for single-label classification.

This script tests single-label classification using a tiny subset of the CLINC150 dataset.
It verifies that:
- Dataset loads correctly in single-label format
- Model trains successfully
- Predictions are in single-label format (list of strings)

Run with: uv run python scripts/test_single_label.py
"""

import sys

# Import path utilities
from intent_classifier.utils.paths import get_repo_root  # noqa: E402

# Get repo root
repo_root = get_repo_root()
sys.path.insert(0, str(repo_root))

from sklearn.metrics import accuracy_score

from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.utils.label_utils import is_multilabel


def main():
    """Test single-label classification with CLINC150 tiny subset."""
    print("=" * 70)
    print("Testing Single-Label Classification (CLINC150 Tiny Subset)")
    print("=" * 70)

    # Load tiny subset of CLINC150 dataset
    print("\n📦 Loading CLINC150 dataset (tiny subset)...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name="clinc150",
            max_train_samples=50,  # Tiny subset for quick testing
            max_test_samples=20,
            max_val_samples=10,
            max_classes=5,  # Only 5 classes for quick testing
            seed=42,
        )

        print("\n✅ Dataset loaded successfully!")
        print("\n📊 Dataset Statistics:")
        print(f"   - Training samples:   {len(X_train):,}")
        print(f"   - Validation samples: {len(X_val):,}")
        print(f"   - Test samples:        {len(X_test):,}")
        print(f"   - Number of classes:   {len(classes)}")
        print(f"   - Classes: {classes}")

        # Verify single-label format
        print("\n🔍 Verifying label format...")
        is_multi = is_multilabel(y_train)
        print(f"   - Is multi-label? {is_multi}")
        if is_multi:
            print("   ❌ ERROR: Expected single-label format, but got multi-label!")
            sys.exit(1)
        print("   ✅ Labels are in single-label format (list of strings)")

        # Show sample labels
        print("\n📝 Sample Labels:")
        print(f"   y_train[0]: {y_train[0]} (type: {type(y_train[0])})")
        print(f"   y_train[1]: {y_train[1]} (type: {type(y_train[1])})")
        print(f"   y_train[2]: {y_train[2]} (type: {type(y_train[2])})")

        # Train model
        print("\n🤖 Training Naive Bayes classifier...")
        clf = NaiveBayesClassifier(max_features=1000, alpha=0.1)
        clf.fit(X_train, y_train)
        print("   ✅ Model trained successfully!")

        # Verify model detected single-label
        print("\n🔍 Verifying model configuration...")
        print(f"   - Model _is_multilabel: {clf._is_multilabel}")
        if clf._is_multilabel:
            print("   ❌ ERROR: Model incorrectly detected multi-label!")
            sys.exit(1)
        print("   ✅ Model correctly detected single-label format")

        # Make predictions
        print("\n🔮 Making predictions on test set...")
        y_pred = clf.predict(X_test)
        print(f"   ✅ Generated {len(y_pred)} predictions")

        # Verify prediction format
        print("\n🔍 Verifying prediction format...")
        is_multi_pred = is_multilabel(y_pred)
        print(f"   - Predictions are multi-label? {is_multi_pred}")
        if is_multi_pred:
            print("   ❌ ERROR: Predictions should be single-label format!")
            sys.exit(1)
        print("   ✅ Predictions are in single-label format (list of strings)")

        # Show sample predictions
        print("\n📝 Sample Predictions:")
        for i in range(min(5, len(X_test))):
            print(f"   Text: {X_test[i][:60]}...")
            print(f"   True label: {y_test[i]}")
            print(f"   Predicted: {y_pred[i]}")
            print()

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\n✅ Single-label classification test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
