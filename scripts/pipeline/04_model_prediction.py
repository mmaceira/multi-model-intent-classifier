#!/usr/bin/env python
"""
04 – Model Prediction

This script performs prediction using trained models. It delegates all heavy-lifting
to the unified `src.prediction.run_prediction` helper.
"""

import sys
from pathlib import Path

# Infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

# Import config setup
from config.notebook_setup import *

# Import dataset and prediction modules
from src.datasets.dataset import get_dataset
from src.prediction import run_prediction
from src.utils.model_loader import load_models_from_config
from src.utils.model_utils import load_model_paths


def main():
    """Main function to run predictions."""

    print("=" * 60)
    print("Model Prediction")
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
        model_paths = load_model_paths(models, MODELS_DIR)

        if not model_paths:
            print("⚠️  Warning: No model paths found. Make sure models have been trained first.")
            print(f"   Expected models in: {MODELS_DIR}")
            return

        print(f"✅ Found {len(model_paths)} trained model(s)")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

    # Check which models already have predictions and filter them out
    print("\n" + "=" * 60)
    print("Checking for Existing Predictions")
    print("=" * 60)
    models_to_predict = {}
    skipped_models = []

    for name, model_path in model_paths.items():
        test_pred_path = PREDICTIONS_DIR / name / "test_predictions.csv"
        if test_pred_path.exists():
            print(f"⏭️  Skipping {name} - predictions already exist at {test_pred_path}")
            skipped_models.append(name)
        else:
            models_to_predict[name] = model_path

    if skipped_models:
        print(f"\n⏭️  Skipped {len(skipped_models)} model(s) that already have predictions")

    if not models_to_predict:
        print("\n✅ All models already have predictions. Nothing to do!")
        print(f"Predictions are in: {PREDICTIONS_DIR}")
        return

    # Predict
    print("\n" + "=" * 60)
    print("Running Predictions")
    print("=" * 60)
    print(f"Generating predictions for {len(X_test)} test samples...")
    try:
        run_prediction(
            models_to_predict,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=PREDICTIONS_DIR,
        )
        print(f"\n✅ Successfully generated predictions for {len(models_to_predict)} model(s)")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise

    print("\n✅ Prediction complete!")
    print(f"Predictions saved to: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
