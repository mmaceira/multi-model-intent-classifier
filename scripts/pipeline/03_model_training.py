#!/usr/bin/env python
"""
03 – Model Training

This script trains the selected models and persists all artefacts. It delegates
all heavy-lifting to the unified `intent_classifier.training.run_training` helper.
"""

import sys

# Import path utilities
from intent_classifier.utils.paths import get_repo_root

# Get repo root and add to path for config imports (config is not part of the installed package)
repo_root = get_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import config setup
from config.notebook_setup import (  # noqa: E402
    MODELS_DIR,
    config_vars,
)

# Import dataset and training modules
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.training import run_training  # noqa: E402
from intent_classifier.utils.file_ops import sanitize_model_name  # noqa: E402
from intent_classifier.utils.method_logger import get_logger  # noqa: E402
from intent_classifier.utils.model_loader import load_models_from_config  # noqa: E402


def main():
    # Disable method logging during training
    get_logger().disable()
    """Main function to train models."""

    print("=" * 60)
    print("Model Training")
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

    print(f"Loaded {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test utterances")
    print(f"Total: {len(classes)} intent classes")
    print("\nNote: Validation set is kept separate for hyperparameter tuning and model selection.")
    print("      Models using cross-validation internally will use the training set for CV.")

    # Load models
    print("\n" + "=" * 60)
    print("Loading Models from Configuration")
    print("=" * 60)
    try:
        # Check if hyperparameters exist (will be checked by model loader based on config name)
        from intent_classifier.utils.config_loader import load_config_with_metadata

        metadata = load_config_with_metadata()
        config_name = metadata["config_name"]
        hyperparams_dir = repo_root / "config" / "algorithm" / "hyperparameters" / config_name
        # Check if any hyperparameter files exist
        if hyperparams_dir.exists() and any(hyperparams_dir.glob("best_*.yaml")):
            num_files = len(list(hyperparams_dir.glob("best_*.yaml")))
            print(
                f"✅ Found {num_files} tuned hyperparameter file(s) for config "
                f"'{config_name}' - will use them for model initialization"
            )
        else:
            print(
                f"ℹ️  No tuned hyperparameters found for config '{config_name}' - "
                f"using defaults from config"
            )
            print("   (Run scripts/tune_hyperparams.py first to optimize hyperparameters)")

        models = load_models_from_config()
        if not models:
            print(
                "⚠️  Warning: No models were loaded. "
                "Check your config/algorithm/models_config.yaml file."
            )
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
        # Use the same sanitisation logic as the training helper so that
        # models whose display names contain path separators (e.g.
        # "Embedding + LogReg (Qwen/Ollama)") map to the correct
        # filesystem directory and can be detected as already trained.
        safe_name = sanitize_model_name(name)
        model_path = MODELS_DIR / safe_name / "model.pkl"
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
            X_val=X_val,
            y_val=y_val,
            output_dir=MODELS_DIR,
        )
        num_trained = len(trained) if trained else len(models_to_train)
        print(f"\n✅ Successfully trained {num_trained} model(s)")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

    print("\n✅ Model training complete!")
    print(f"Trained models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
