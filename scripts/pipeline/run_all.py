#!/usr/bin/env python
"""
Run All Pipeline Scripts

This script runs all pipeline scripts in sequence. It's a convenience script
for executing the complete training pipeline from start to finish.
"""

import subprocess
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Define the scripts in order
scripts = [
    "00_data_loading.py",
    "01_exploratory_analysis.py",
    "02_build_embeddings.py",
    "03_model_training.py",
    "04_model_prediction.py",
    "05_model_evaluation.py",
]


def main():
    """Run all pipeline scripts in sequence."""

    print("=" * 60)
    print("Running Complete Training Pipeline")
    print("=" * 60)
    print()

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
                cwd=script_dir.parent.parent,  # Run from repo root
            )
            print(f"✅ {script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ {script_name} failed with exit code {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⚠️  Pipeline interrupted by user at {script_name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All pipeline scripts completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
