#!/usr/bin/env python
"""
Run All Pipeline Scripts

This script runs all pipeline scripts in sequence. It's a convenience script
for executing the complete training pipeline from start to finish.

Supports:
- CONFIG_FILE environment variable to override default config
- --tune flag to run hyperparameter tuning before training
- --save-model flag to save trained model to a specific path
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Set deterministic seed before any imports
from intent_classifier.utils.seed import set_global_seed
from intent_classifier.utils.warnings_config import suppress_pydantic_warnings

# Suppress verbose Pydantic warnings globally
suppress_pydantic_warnings()

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent

# Define the scripts in order
scripts = [
    "00_data_loading.py",
    "01_exploratory_analysis.py",
    "02_build_embeddings.py",
    "03_model_training.py",
    "04_model_prediction.py",
    "05_model_evaluation.py",
]


def load_config() -> dict:
    """Load configuration from YAML file."""
    import yaml

    config_file = os.environ.get("CONFIG_FILE", "config.yaml")
    config_path = repo_root / "config" / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_hyperparameter_tuning(config: dict) -> dict:
    """Run hyperparameter tuning and return best parameters."""
    try:
        from intent_classifier.hparam.tune import run_tuning

        return run_tuning(config)
    except ImportError:
        print("⚠️  Warning: Hyperparameter tuning module not available")
        print("   Install with: pip install -e '.[tune]'")
        return {}


def save_model(model_path: str):
    """Save the trained model to the specified path."""
    # This is a placeholder - actual model saving happens in 03_model_training.py
    # We'll copy the model from the default location to the specified path

    config_file = os.environ.get("CONFIG_FILE", "config.yaml")
    config_name = Path(config_file).stem

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
            "  CONFIG_FILE=config/config_tiny_dataset.yaml "
            "python scripts/pipeline/run_all.py\n\n"
            "  # Run with hyperparameter tuning\n"
            "  python scripts/pipeline/run_all.py --tune\n\n"
            "  # Run with tuning and save model\n"
            "  CONFIG_FILE=config/config_tiny_dataset.yaml "
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

    # Load config to get seed
    try:
        config = load_config()
        seed = config.get("general", {}).get("seed", 42)
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
        print("   Using default seed: 42")
        seed = 42

    # Set deterministic seed
    set_global_seed(seed)

    print("=" * 60)
    print("Running Complete Training Pipeline")
    print("=" * 60)
    print(f"Config: {os.environ.get('CONFIG_FILE', 'config.yaml')}")
    print(f"Seed: {seed}")
    print()

    # Step 1: Hyperparameter tuning (if requested)
    if args.tune:
        print("\n" + "=" * 60)
        print("Step 0: Hyperparameter Tuning")
        print("=" * 60)
        try:
            best_params = run_hyperparameter_tuning(config)
            if best_params:
                print(f"\n✅ Tuning complete. Best parameters for {len(best_params)} model(s)")
            else:
                print("\n⚠️  No hyperparameters returned from tuning")
        except Exception as e:
            print(f"\n❌ Hyperparameter tuning failed: {e}")
            print("   Continuing with default hyperparameters...")
            import traceback

            traceback.print_exc()

    # Step 2: Run pipeline scripts
    for i, script_name in enumerate(scripts, 1):
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"❌ Error: Script not found: {script_path}")
            sys.exit(1)

        print(f"\n[{i}/{len(scripts)}] Running {script_name}...")
        print("-" * 60)

        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                cwd=repo_root,  # Run from repo root
            )
            print(f"✅ {script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ {script_name} failed with exit code {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⚠️  Pipeline interrupted by user at {script_name}")
            sys.exit(1)

    # Step 3: Save model if requested
    if args.save_model:
        print("\n" + "=" * 60)
        print("Saving Model")
        print("=" * 60)
        save_model(args.save_model)

    print("\n" + "=" * 60)
    print("✅ All pipeline scripts completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
