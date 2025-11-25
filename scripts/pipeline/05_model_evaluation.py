#!/usr/bin/env python
"""
05 – Model Evaluation

This script evaluates persisted predictions – no models are loaded. It generates
comprehensive evaluation metrics and visualizations.
"""

import sys
from pathlib import Path

# Infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[2]
# Add repo root to path for config imports (config is not part of the installed package)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import config setup
from config.notebook_setup import (  # noqa: E402
    PREDICTIONS_DIR,
    RESULTS_DIR,
    config_vars,
)

# Import dataset and evaluation modules
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.evaluation import display_detailed_results, run_evaluations  # noqa: E402
from intent_classifier.utils.model_loader import load_models_from_config  # noqa: E402


def main():
    """Main function to evaluate models."""

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        use_oos=config_vars.get("DATASET_USE_OOS", False),
        max_classes=config_vars.get("DATASET_MAX_CLASSES", None),
        max_train_samples=config_vars.get("DATASET_MAX_TRAIN_SAMPLES", None),
        max_test_samples=config_vars.get("DATASET_MAX_TEST_SAMPLES", None),
        seed=config_vars.get("GENERAL_SEED", 42),
    )

    print(f"Loaded {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test utterances")
    print(f"Total: {len(classes)} intent classes")
    print(
        "\nNote: Evaluation uses test set only. Train/val splits are kept separate for reference."
    )

    # Load models (for metadata)
    print("\n" + "=" * 60)
    print("Loading Model Configuration")
    print("=" * 60)
    try:
        models = load_models_from_config()
        if not models:
            print("⚠️  Warning: No models found in configuration.")
            return
        print(f"✅ Loaded configuration for {len(models)} model(s)")
    except Exception as e:
        print(f"❌ Error loading model configuration: {e}")
        raise

    # Check if predictions exist
    if not PREDICTIONS_DIR.exists() or not any(PREDICTIONS_DIR.iterdir()):
        print(f"\n⚠️  Warning: No predictions found in {PREDICTIONS_DIR}")
        print("   Please run 04_model_prediction.py first.")
        return

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running Evaluations")
    print("=" * 60)
    try:
        results = run_evaluations(
            models,
            artefacts_root=PREDICTIONS_DIR,
            output_dir=RESULTS_DIR,
            verbose=True,
        )
        if not results:
            print("⚠️  Warning: No evaluation results generated.")
            return
        print(f"✅ Evaluated {len(results)} model(s)")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

    # Make comparison plots sorted by model name
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    # Define the desired model order
    # Note: Use actual model names from configuration
    # (may include suffixes like "(local-embeddings)")
    model_order = [
        "Naive Bayes",
        "Linear SVM",
        "TF-IDF bigrams + SVM",
        "MiniLM + LogReg",
        "Embedding + LogReg",
        "RAG-CentroidNN",
        "RAG-kMajority",
        "RAG-LLM (local-embeddings)",  # RAG-LLM with local SBERT embeddings
        "RAG-LLM (OpenAI-embeddings)",  # RAG-LLM with OpenAI embeddings
    ]

    # Display results in the specified order
    try:
        display_detailed_results(results, model_order=model_order, output_dir=RESULTS_DIR)
        print("✅ Comparison plots generated successfully")
    except Exception as e:
        print(f"⚠️  Warning: Error generating comparison plots: {e}")
        print("   Evaluation results are still available.")

    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
