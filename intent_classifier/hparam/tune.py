"""Run hyperparameter tuning and expose results to the main pipeline."""

import json
import subprocess
import sys
from typing import Any

import yaml


def run_tuning(config: dict[str, Any]) -> dict[str, Any]:
    """Run hyperparameter tuning and return best parameters.

    This function runs the hyperparameter tuning script and returns the best
    hyperparameters found. The results are also saved to artifacts/best_params.json
    and config/algorithm/hyperparameters/{config_name}/best_*.yaml.

    Args:
        config: Configuration dictionary (must include dataset, general, model sections)

    Returns:
        Dictionary mapping model names to their best hyperparameters

    Raises:
        RuntimeError: If tuning fails
    """
    from intent_classifier.utils.paths import get_repo_root

    repo_root = get_repo_root()

    # Get config file name from environment or discover default
    from intent_classifier.utils.config_loader import discover_config_file
    from intent_classifier.utils.paths import get_config_path

    config_file = discover_config_file()
    config_path = get_config_path(config_file)

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

    # Load best hyperparameters from the config/algorithm/hyperparameters directory
    hyperparams_dir = repo_root / "config" / "algorithm" / "hyperparameters" / config_name
    best_params = {}

    if hyperparams_dir.exists():
        for file_path in hyperparams_dir.glob("best_*.yaml"):
            model_name = file_path.stem.replace("best_", "")
            with open(file_path, encoding="utf-8") as f:
                params = yaml.safe_load(f)
                best_params[model_name] = params

    # Also save to artifacts/best_params.json for easy access
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    best_params_json = artifacts_dir / "best_params.json"

    with open(best_params_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n✅ Best hyperparameters saved to {best_params_json}")
    print(f"   Also available in {hyperparams_dir}")

    return best_params
