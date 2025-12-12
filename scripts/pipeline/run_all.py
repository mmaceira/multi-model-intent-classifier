#!/usr/bin/env python
"""
Run All Pipeline Scripts

This script runs all pipeline scripts in sequence. It's a convenience script
for executing the complete training pipeline from start to finish.

This script uses the PipelineOrchestrator to call pipeline steps as
Python functions, making it easier to test, debug, and reuse.

Supports:
- CONFIG_FILE environment variable to override default config
- --tune flag to run hyperparameter tuning before training
- --save-model flag to save trained model to a specific path
"""

import argparse
import os
import sys
from pathlib import Path

# Set deterministic seed before any imports
from intent_classifier.utils.warnings_config import suppress_pydantic_warnings

# Suppress verbose Pydantic warnings globally
suppress_pydantic_warnings()

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent


def load_config() -> dict:
    """Load configuration from YAML file."""
    from intent_classifier.utils.config_loader import load_config as load_config_centralized

    return load_config_centralized()


def save_model(model_path: str):
    """Save the trained model to the specified path."""
    # This is a placeholder - actual model saving happens in 03_model_training.py
    # We'll copy the model from the default location to the specified path

    config_file = os.environ.get("CONFIG_FILE")
    if config_file:
        from intent_classifier.utils.config_loader import parse_config_path

        _, config_name = parse_config_path(config_file)
    else:
        # Fallback if no CONFIG_FILE is set
        config_name = "default"

    # Find the most recent model (this is a simplified approach)
    # In practice, the model should be saved during training
    models_dir = repo_root / "output" / f"experiment_{config_name}" / "models"
    if not models_dir.exists():
        print(f"⚠️  Warning: Models directory not found: {models_dir}")
        return

    # For now, we'll just note where models are saved
    # The actual saving should be done in the training script
    print(f"💡 Models are saved in: {models_dir}")
    print(f"   To copy to {model_path}, use the training script's save functionality")


def main():
    """Run all pipeline scripts in sequence with optional tuning."""
    parser = argparse.ArgumentParser(
        description="Run the complete training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run with default config\n"
            "  python scripts/pipeline/run_all.py\n\n"
            "  # Run with custom config\n"
            "  CONFIG_FILE=config/dataset/clinc150/tiny.yaml "
            "python scripts/pipeline/run_all.py\n\n"
            "  # Run with hyperparameter tuning\n"
            "  python scripts/pipeline/run_all.py --tune\n\n"
            "  # Run with tuning and save model\n"
            "  CONFIG_FILE=config/dataset/clinc150/tiny.yaml "
            "python scripts/pipeline/run_all.py --tune "
            "--save-model artifacts/model.pkl\n"
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before training",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model (optional)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
        config = {}

    # Create orchestrator and run pipeline
    from intent_classifier.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(repo_root=repo_root)
    success = orchestrator.run_all(skip_tuning=not args.tune, config=config)

    # Save model if requested
    if args.save_model and success:
        print("\n" + "=" * 60)
        print("Saving Model")
        print("=" * 60)
        save_model(args.save_model)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
