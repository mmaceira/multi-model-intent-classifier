#!/usr/bin/env python
"""
Test script for multi-label classification.

This script tests multi-label classification using the NLU++ dataset.
It verifies that:
- Dataset loads correctly in multi-label format (list of lists)
- Model trains successfully with multi-label data
- Predictions are in multi-label format (list of lists)

Run with: uv run python scripts/test_multi_label.py
"""

import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()
sys.path.insert(0, str(repo_root))

from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.utils.label_utils import is_multilabel


def main():
    """Test multi-label classification with NLU++ dataset."""
    print("=" * 70)
    print("Testing Multi-Label Classification (NLU++ Dataset)")
    print("=" * 70)

    # Load NLU++ dataset in multi-label mode
    print("\n📦 Loading NLU++ dataset (multi-label mode)...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name="nlu_plus",
            multilabel=True,  # Enable multi-label mode
            max_train_samples=100,  # Limit for quick testing
            max_test_samples=30,
            max_val_samples=20,
            # Note: max_classes filtering has a bug with multi-label, so we skip it for now
            # max_classes=10,  # Limit classes for quick testing
            seed=42,
        )

        print("\n✅ Dataset loaded successfully!")
        print("\n📊 Dataset Statistics:")
        print(f"   - Training samples:   {len(X_train):,}")
        print(f"   - Validation samples: {len(X_val):,}")
        print(f"   - Test samples:        {len(X_test):,}")
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
        print(f"   y_train[1]: {y_train[1]} (type: {type(y_train[1])}, length: {len(y_train[1])})")
        if len(y_train) > 2:
            print(
                f"   y_train[2]: {y_train[2]} (type: {type(y_train[2])}, length: {len(y_train[2])})"
            )

        # Count labels per sample
        labels_per_sample = [len(labels) for labels in y_train]
        print("\n📈 Label Statistics:")
        print(f"   - Average labels per sample: {np.mean(labels_per_sample):.2f}")
        print(f"   - Min labels per sample: {min(labels_per_sample)}")
        print(f"   - Max labels per sample: {max(labels_per_sample)}")

        # Train model
        print("\n🤖 Training Naive Bayes classifier...")
        clf = NaiveBayesClassifier(max_features=1000, alpha=0.1)
        clf.fit(X_train, y_train)
        print("   ✅ Model trained successfully!")

        # Verify model detected multi-label
        print("\n🔍 Verifying model configuration...")
        print(f"   - Model _is_multilabel: {clf._is_multilabel}")
        if not clf._is_multilabel:
            print("   ❌ ERROR: Model incorrectly detected single-label!")
            sys.exit(1)
        print("   ✅ Model correctly detected multi-label format")

        # Make predictions
        print("\n🔮 Making predictions on test set...")
        y_pred = clf.predict(X_test)
        print(f"   ✅ Generated {len(y_pred)} predictions")

        # Verify prediction format
        print("\n🔍 Verifying prediction format...")
        is_multi_pred = is_multilabel(y_pred)
        print(f"   - Predictions are multi-label? {is_multi_pred}")
        if not is_multi_pred:
            print("   ❌ ERROR: Predictions should be multi-label format!")
            sys.exit(1)
        print("   ✅ Predictions are in multi-label format (list of lists)")

        # Show sample predictions
        print("\n📝 Sample Predictions:")
        for i in range(min(5, len(X_test))):
            print(f"   Text: {X_test[i][:60]}...")
            print(f"   True labels: {y_test[i]} ({len(y_test[i])} labels)")
            print(f"   Predicted: {y_pred[i]} ({len(y_pred[i])} labels)")
            print()

        # Calculate metrics
        # For multi-label, we need to convert to binary format for sklearn metrics
        from intent_classifier.utils.label_utils import binarize_labels

        y_test_binary, _ = binarize_labels(y_test, classes=classes)
        y_pred_binary, _ = binarize_labels(y_pred, classes=classes)

        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(y_test_binary, y_pred_binary)
        print("\n📊 Evaluation Metrics:")
        print(
            f"   - Subset Accuracy (exact match): {subset_accuracy:.4f} ({subset_accuracy*100:.2f}%)"
        )

        # Hamming loss (lower is better)
        hamming = hamming_loss(y_test_binary, y_pred_binary)
        print(f"   - Hamming Loss: {hamming:.4f} (lower is better)")

        # F1 scores
        f1_macro = f1_score(y_test_binary, y_pred_binary, average="macro", zero_division=0)
        f1_micro = f1_score(y_test_binary, y_pred_binary, average="micro", zero_division=0)
        print(f"   - F1 Score (macro): {f1_macro:.4f}")
        print(f"   - F1 Score (micro): {f1_micro:.4f}")

        print("\n✅ Multi-label classification test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
