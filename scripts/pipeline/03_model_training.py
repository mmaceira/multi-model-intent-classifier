#!/usr/bin/env python
"""
03 – Model Training

This script trains the selected models and persists all artefacts. It delegates
all heavy-lifting to the unified `src.training.run_training` helper.
"""

import sys
from pathlib import Path

# Infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

# Import config setup
from config.notebook_setup import *

# Import dataset and training modules
from src.datasets.dataset import get_dataset
from src.training import run_training
from src.utils.model_loader import load_models_from_config


def main():
    """Main function to train models."""

    print("=" * 60)
    print("Model Training")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        use_oos=False,  # Set to True to include out-of-scope examples as an extra class
    )

    print(f"Loaded {len(X_train)} training utterances with {len(classes)} intent classes")
    print(f"Test set contains {len(X_test)} utterances")

    # Load models
    print("\n" + "=" * 60)
    print("Loading Models from Configuration")
    print("=" * 60)
    try:
        models = load_models_from_config()
        if not models:
            print("⚠️  Warning: No models were loaded. Check your config/models_config.yaml file.")
            return
        print(f"✅ Successfully loaded {len(models)} model(s) for training")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("   Please check your configuration files.")
        raise

    # Check which models already exist and filter them out
    print("\n" + "=" * 60)
    print("Checking for Existing Models")
    print("=" * 60)
    models_to_train = {}
    skipped_models = []

    for name, model in models.items():
        model_path = MODELS_DIR / name / "model.pkl"
        if model_path.exists():
            print(f"⏭️  Skipping {name} - model already exists at {model_path}")
            skipped_models.append(name)
        else:
            models_to_train[name] = model

    if skipped_models:
        print(f"\n⏭️  Skipped {len(skipped_models)} model(s) that are already trained")

    if not models_to_train:
        print("\n✅ All models are already trained. Nothing to do!")
        print(f"Trained models are in: {MODELS_DIR}")
        return

    # Train and persist
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    print(f"Training {len(models_to_train)} model(s)...")
    try:
        trained = run_training(
            models_to_train,
            X_train=X_train,
            y_train=y_train,
            output_dir=MODELS_DIR,
        )
        print(
            f"\n✅ Successfully trained {len(trained) if trained else len(models_to_train)} model(s)"
        )
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

    print("\n✅ Model training complete!")
    print(f"Trained models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
