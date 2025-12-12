#!/usr/bin/env python
"""
04 – Model Prediction

This script performs prediction using trained models. It delegates all heavy-lifting
to the unified `intent_classifier.prediction.run_prediction` helper.
"""

import sys
from pathlib import Path

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root and add to path for config imports (config is not part of the installed package)
repo_root = get_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Suppress verbose warnings before other imports
from intent_classifier.utils.warnings_config import suppress_pydantic_warnings  # noqa: E402

suppress_pydantic_warnings()

# Import config setup
# Import from prediction.py file (not prediction/ directory)
# These imports must come after sys.path manipulation
from config.notebook_setup import (  # noqa: E402
    MODELS_DIR,
    PREDICTIONS_DIR,
    config_vars,
)

# Import dataset and prediction modules
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402

prediction_module_path = Path(__file__).parent.parent.parent / "intent_classifier" / "prediction.py"
if str(prediction_module_path.parent) not in sys.path:
    sys.path.insert(0, str(prediction_module_path.parent))
from intent_classifier.prediction import run_prediction  # noqa: E402
from intent_classifier.utils.model_loader import load_models_from_config  # noqa: E402
from intent_classifier.utils.model_utils import load_model_paths  # noqa: E402


def main():
    """Main function to run predictions."""

    print("=" * 60)
    print("Model Prediction")
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

    # For prediction, we merge train+val to match what models were trained on
    # (Models are trained on train only, but for consistency in prediction we use train+val)
    # Note: Currently not used, but kept for potential future use
    # X_train_combined = X_train + X_val
    # y_train_combined = y_train + y_val

    print(f"Loaded {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test utterances")
    print(f"Total: {len(classes)} intent classes")
    print("\nNote: For prediction, train+val are combined for consistency with training data.")

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
    print(
        "Note: Only test set predictions are generated "
        "(training predictions skipped for efficiency)"
    )
    try:
        run_prediction(
            models_to_predict,
            X_train=None,  # Skip training predictions - they're slow and usually not needed
            y_train=None,
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
