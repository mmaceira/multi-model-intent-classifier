#!/usr/bin/env python
"""
Run All Experiments

This script runs the complete pipeline for all available experiment configurations.
It optionally runs hyperparameter tuning before training.

Usage:
    # Run all experiments (no tuning)
    python scripts/run_all_experiments.py

    # Run with hyperparameter tuning
    python scripts/run_all_experiments.py --tune

    # Tune with custom number of samples
    python scripts/run_all_experiments.py --tune --num-samples 50
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# List of all experiment configs
EXPERIMENTS = [
    "config_tiny_dataset.yaml",
    "config_10_classes.yaml",
    "config_25_classes.yaml",
    "config_full_dataset.yaml",
    "config.yaml",
]


def check_prerequisites():
    """Check if prerequisites are met."""
    print("Checking prerequisites...")

    # Check Python
    if sys.executable is None:
        print("❌ Error: Python not found")
        return False

    # Check if Ollama is running (optional, just warn)
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print("✅ Ollama is running")
    except Exception:
        print("⚠️  Warning: Ollama doesn't seem to be running")
        print("   RAG-LLM models may fail. Start Ollama with: ollama serve")

    print("✅ Prerequisites check passed\n")
    return True


def run_tuning(config_file, num_samples):
    """Run hyperparameter tuning for a config."""
    print("=" * 60)
    print(f"Tuning hyperparameters: {config_file}")
    print("=" * 60)

    env = {"CONFIG_FILE": config_file}
    cmd = [
        sys.executable,
        "scripts/tune_hyperparams.py",
        "--config",
        f"config/{config_file}",
        "--all",
        "--num-samples",
        str(num_samples),
    ]

    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, **env},
            check=True,
        )
        print(f"✅ Hyperparameter tuning completed for {config_file}\n")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Hyperparameter tuning failed for {config_file}\n")
        return False


def run_pipeline(config_file):
    """Run full pipeline for a config."""
    print("=" * 60)
    print(f"Running pipeline: {config_file}")
    print("=" * 60)

    env = {"CONFIG_FILE": config_file}
    cmd = [sys.executable, "scripts/pipeline/run_all.py"]

    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, **env},
            check=True,
        )
        print(f"✅ Pipeline completed for {config_file}\n")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Pipeline failed for {config_file}\n")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (no tuning)
  python scripts/run_all_experiments.py

  # Run with hyperparameter tuning
  python scripts/run_all_experiments.py --tune

  # Tune with custom number of samples
  python scripts/run_all_experiments.py --tune --num-samples 50
        """,
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of samples for hyperparameter tuning (default: 30)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Running All Experiments")
    print("=" * 60)
    print()
    print("Experiments to run:")
    for exp in EXPERIMENTS:
        print(f"  - {exp}")
    print()

    if args.tune:
        print("Hyperparameter tuning: ENABLED")
        print(f"Number of samples per model: {args.num_samples}")
    else:
        print("Hyperparameter tuning: DISABLED")
        print("(Use --tune to enable hyperparameter tuning)")
    print()

    if not check_prerequisites():
        sys.exit(1)

    # Track results
    successful = []
    failed = []

    # Run experiments
    for exp_config in EXPERIMENTS:
        print()
        print("=" * 60)
        print(f"Processing: {exp_config}")
        print("=" * 60)
        print()

        # Step 1: Hyperparameter tuning (if enabled)
        if args.tune:
            if not run_tuning(exp_config, args.num_samples):
                print("⚠️  Hyperparameter tuning failed, continuing anyway...\n")

        # Step 2: Run full pipeline
        if run_pipeline(exp_config):
            successful.append(exp_config)
        else:
            failed.append(exp_config)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()

    if successful:
        print(f"✅ Successful experiments ({len(successful)}):")
        for exp in successful:
            print(f"  ✓ {exp}")
        print()

    if failed:
        print(f"❌ Failed experiments ({len(failed)}):")
        for exp in failed:
            print(f"  ✗ {exp}")
        print()
        sys.exit(1)
    else:
        print("All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
