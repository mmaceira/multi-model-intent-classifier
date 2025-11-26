"""Run hyperparameter tuning and expose results to the main pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def run_tuning(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run hyperparameter tuning and return best parameters.

    This function runs the hyperparameter tuning script and returns the best
    hyperparameters found. The results are also saved to artifacts/best_params.json
    and config/hyperparameters/{config_name}/best_*.yaml.

    Args:
        config: Configuration dictionary (must include dataset, general, model sections)

    Returns:
        Dictionary mapping model names to their best hyperparameters

    Raises:
        RuntimeError: If tuning fails
    """
    repo_root = Path(__file__).resolve().parents[3]

    # Get config file name from environment or infer from config
    config_file = os.environ.get("CONFIG_FILE", "config.yaml")
    config_path = repo_root / "config" / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Extract config name (without extension)
    config_name = config_path.stem

    # Run the tuning script
    tune_script = repo_root / "scripts" / "tune_hyperparams.py"
    if not tune_script.exists():
        raise FileNotFoundError(f"Tuning script not found: {tune_script}")

    print("=" * 60)
    print("Running Hyperparameter Tuning")
    print("=" * 60)
    print(f"Config: {config_file}")
    print()

    try:
        # Run tuning with --all flag to tune all models
        subprocess.run(
            [
                sys.executable,
                str(tune_script),
                "--config",
                str(config_path),
                "--all",
                "--num-samples",
                "30",  # Default number of samples
            ],
            cwd=repo_root,
            check=True,
            capture_output=False,  # Show output in real-time
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Hyperparameter tuning failed with exit code {e.returncode}") from e

    # Load best hyperparameters from the config/hyperparameters directory
    hyperparams_dir = repo_root / "config" / "hyperparameters" / config_name
    best_params = {}

    if hyperparams_dir.exists():
        for file_path in hyperparams_dir.glob("best_*.yaml"):
            model_name = file_path.stem.replace("best_", "")
            with open(file_path) as f:
                params = yaml.safe_load(f)
                best_params[model_name] = params

    # Also save to artifacts/best_params.json for easy access
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    best_params_json = artifacts_dir / "best_params.json"

    with open(best_params_json, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n✅ Best hyperparameters saved to {best_params_json}")
    print(f"   Also available in {hyperparams_dir}")

    return best_params
