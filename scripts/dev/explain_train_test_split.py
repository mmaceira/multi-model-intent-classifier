#!/usr/bin/env python
"""
Script to explain how the train/test split works in CLINC150 dataset.

This shows that the splits come pre-defined from HuggingFace.
"""

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root
repo_root = get_repo_root()

from datasets import load_dataset  # noqa: E402


def main():
    """Show how the dataset splits work."""

    print("=" * 70)
    print("CLINC150 Dataset Split Explanation")
    print("=" * 70)

    print("\n📦 Loading dataset from HuggingFace...")
    print("   Dataset: clinc_oos (config: 'plus')")

    # Load the dataset directly from HuggingFace
    ds = load_dataset("clinc_oos", "plus")

    print("\n✅ Dataset loaded! The dataset comes with pre-defined splits:")
    print(f"   - Available splits: {list(ds.keys())}")

    # Show split sizes
    print("\n📊 Split Sizes (from HuggingFace):")
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            split_size = len(ds[split_name])
            print(f"   - {split_name:12s}: {split_size:6,} samples")

    # Show what the code does
    print("\n" + "=" * 70)
    print("What the code does:")
    print("=" * 70)

    print("\n1️⃣  Extracts splits from HuggingFace dataset:")
    print("   - X_train, y_train = extract('train')")
    print("   - X_valid, y_valid = extract('validation')")
    print("   - X_test, y_test = extract('test')")

    print("\n2️⃣  Merges validation into training:")
    print("   - X_train_extended = X_train + X_valid")
    print("   - y_train_extended = y_train + y_valid")
    print("   (This is done to maximize training data)")

    print("\n3️⃣  Keeps test set separate:")
    print("   - X_test, y_test remain unchanged")
    print("   (Test set is never used for training)")

    # Show actual numbers
    train_size = len(ds["train"])
    valid_size = len(ds["validation"])
    test_size = len(ds["test"])
    total_train = train_size + valid_size

    print("\n" + "=" * 70)
    print("Final Split Sizes (after merging validation into train):")
    print("=" * 70)
    print(f"\n   Training set: {train_size:,} + {valid_size:,} = {total_train:,} samples")
    print(f"   Test set:     {test_size:,} samples")
    print(f"   Total:        {total_train + test_size:,} samples")

    # Show class distribution in each split
    print("\n" + "=" * 70)
    print("Class Distribution Across Splits:")
    print("=" * 70)

    def get_classes(split_name):
        """Get unique classes in a split."""
        classes = set()
        for row in ds[split_name]:
            intent = row["intent"]
            # Handle both int and string intents
            if isinstance(intent, int):
                intent_feature = ds[split_name].features["intent"]
                if hasattr(intent_feature, "names"):
                    intent = intent_feature.names[intent]
            classes.add(str(intent))
        return classes

    train_classes = get_classes("train")
    valid_classes = get_classes("validation")
    test_classes = get_classes("test")

    print(f"\n   Train classes:      {len(train_classes)}")
    print(f"   Validation classes: {len(valid_classes)}")
    print(f"   Test classes:       {len(test_classes)}")
    print(f"   Classes in all:     {len(train_classes & valid_classes & test_classes)}")

    print("\n" + "=" * 70)
    print("Key Points:")
    print("=" * 70)
    print("\n✅ The train/test split comes PRE-DEFINED from HuggingFace")
    print("✅ The code does NOT create the split - it uses existing splits")
    print("✅ Validation set is merged into training (common practice)")
    print("✅ Test set remains completely separate (never used for training)")
    print("✅ This ensures reproducible, standard evaluation")

    print("\n💡 Why this matters:")
    print("   - Standard benchmark splits allow fair comparison")
    print("   - Test set is never seen during training")
    print("   - Results are reproducible across different runs")


if __name__ == "__main__":
    main()
