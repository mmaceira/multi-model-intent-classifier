"""Script to debug predictions from different algorithms.

This script trains models with different algorithms and validates that all
predictions are within the valid classes from the dataset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.algorithms.transformer_logreg import TransformerLogReg
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.prediction import run_prediction
from intent_classifier.training import run_training


def validate_predictions(
    predictions: dict,
    valid_classes: set[str],
    is_multilabel: bool,
    algorithm_name: str,
) -> tuple[bool, list[str]]:
    """Validate that all predictions are in valid classes.

    Args:
        predictions: Dictionary with 'train' and 'test' predictions
        valid_classes: Set of valid class names
        is_multilabel: Whether predictions are multi-label
        algorithm_name: Name of the algorithm being tested

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    is_valid = True

    for split_name, preds in predictions.items():
        if preds is None:
            continue

        for i, pred in enumerate(preds):
            if is_multilabel:
                if not isinstance(pred, list):
                    errors.append(
                        f"{algorithm_name} {split_name}[{i}]: Expected list, got {type(pred)}"
                    )
                    is_valid = False
                    continue

                if len(pred) == 0:
                    errors.append(
                        f"{algorithm_name} {split_name}[{i}]: Empty prediction (should have at least one label)"
                    )
                    is_valid = False
                    continue

                for j, label in enumerate(pred):
                    if not isinstance(label, str):
                        errors.append(
                            f"{algorithm_name} {split_name}[{i}][{j}]: Expected string, got {type(label)}"
                        )
                        is_valid = False
                    elif label not in valid_classes:
                        errors.append(
                            f"{algorithm_name} {split_name}[{i}][{j}]: Invalid class '{label}' "
                            f"not in {sorted(valid_classes)}"
                        )
                        is_valid = False
            else:
                if not isinstance(pred, str):
                    errors.append(
                        f"{algorithm_name} {split_name}[{i}]: Expected string, got {type(pred)}"
                    )
                    is_valid = False
                elif len(pred) == 0:
                    errors.append(f"{algorithm_name} {split_name}[{i}]: Empty prediction")
                    is_valid = False
                elif pred not in valid_classes:
                    errors.append(
                        f"{algorithm_name} {split_name}[{i}]: Invalid class '{pred}' "
                        f"not in {sorted(valid_classes)}"
                    )
                    is_valid = False

    return is_valid, errors


def debug_algorithm(
    algorithm_class,
    algorithm_kwargs: dict,
    algorithm_name: str,
    multilabel: bool = False,
    max_train_samples: int = 100,
    max_val_samples: int = 20,
    max_classes: int = 10,
    seed: int = 42,
) -> bool:
    """Debug a single algorithm.

    Args:
        algorithm_class: Algorithm class to test
        algorithm_kwargs: Keyword arguments for algorithm initialization
        algorithm_name: Name of the algorithm
        multilabel: Whether to use multi-label dataset
        max_train_samples: Maximum training samples
        max_val_samples: Maximum validation samples
        max_classes: Maximum number of classes
        seed: Random seed

    Returns:
        True if all predictions are valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {algorithm_name}")
    print(f"{'='*60}")

    # Try to get dataset name from config file if available
    dataset_name = None
    config_file = os.environ.get("CONFIG_FILE")
    if config_file:
        try:
            from pathlib import Path

            import yaml

            from intent_classifier.utils.paths import get_config_path, get_repo_root

            config_path = get_config_path(config_file)
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                    dataset_name = cfg.get("dataset", {}).get("name")
        except Exception:
            pass  # Fall back to default selection

    # Fallback: select dataset based on multilabel requirement
    # Try to discover available datasets from config/dataset/ directory
    if dataset_name is None:
        try:
            from pathlib import Path

            from intent_classifier.utils.paths import get_repo_root

            repo_root = get_repo_root()
            dataset_dir = repo_root / "config" / "dataset"
            if dataset_dir.exists():
                available_datasets = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
                # Prefer datasets that match the multilabel requirement
                # For now, use first available as fallback (could be improved to check config)
                if available_datasets:
                    dataset_name = available_datasets[0]
        except Exception:
            pass

    # Final fallback (should rarely be needed)
    if dataset_name is None:
        dataset_name = "clinc150" if not multilabel else "nlu_plus"

    X_train, y_train, X_val, y_val, _, _, classes = get_dataset(
        dataset_name=dataset_name,
        multilabel=multilabel,
        max_train_samples=max_train_samples,
        max_test_samples=20,
        max_val_samples=max_val_samples,
        max_classes=max_classes,
        seed=seed,
    )

    valid_classes = set(classes)
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Classes: {sorted(valid_classes)}")
    print(f"Multi-label: {multilabel}")

    # Train model
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        models = {algorithm_name: algorithm_class(**algorithm_kwargs)}

        print(f"\nTraining {algorithm_name}...")
        run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=tmp_dir,
            save_models=True,
            verbose=False,
        )

        model_path = Path(tmp_dir) / algorithm_name / "model.pkl"

        # Make predictions
        print(f"Making predictions with {algorithm_name}...")
        with tempfile.TemporaryDirectory() as pred_dir:
            predictions = run_prediction(
                models_or_paths={algorithm_name: str(model_path)},
                X_train=X_train[:10],
                y_train=y_train[:10],
                X_test=X_val,
                y_test=y_val,
                output_dir=pred_dir,
                save_train_predictions=True,
                save_test_predictions=True,
                verbose=False,
            )

            # Validate predictions
            model_preds = predictions[algorithm_name]
            is_valid, errors = validate_predictions(
                model_preds,
                valid_classes,
                multilabel,
                algorithm_name,
            )

            if is_valid:
                print(f"? {algorithm_name}: All predictions are valid!")
                print(f"   Train predictions: {len(model_preds.get('train', []))}")
                print(f"   Test predictions: {len(model_preds.get('test', []))}")
                return True
            else:
                print(f"? {algorithm_name}: Found {len(errors)} validation errors:")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"   - {error}")
                if len(errors) > 10:
                    print(f"   ... and {len(errors) - 10} more errors")
                return False


def main():
    """Main function to debug all algorithms."""
    print("=" * 60)
    print("Debugging Predictions - All Algorithms")
    print("=" * 60)

    # Single-label algorithms
    single_label_algorithms = [
        (LinearSVMClassifier, {"max_features": 100, "C": 0.1, "calibrate": False}, "LinearSVM"),
        (NaiveBayesClassifier, {"max_features": 100, "alpha": 0.5}, "NaiveBayes"),
        (
            TransformerLogReg,
            {"model_name": "all-MiniLM-L6-v2", "max_iter": 100},
            "TransformerLogReg",
        ),
    ]

    # Multi-label algorithms
    multilabel_algorithms = [
        (LinearSVMClassifier, {"max_features": 100, "C": 0.1, "calibrate": False}, "LinearSVM"),
        (NaiveBayesClassifier, {"max_features": 100, "alpha": 0.5}, "NaiveBayes"),
        (
            TransformerLogReg,
            {"model_name": "all-MiniLM-L6-v2", "max_iter": 100},
            "TransformerLogReg",
        ),
    ]

    results = []

    # Test single-label algorithms
    print("\n" + "=" * 60)
    print("SINGLE-LABEL ALGORITHMS")
    print("=" * 60)
    for alg_class, alg_kwargs, alg_name in single_label_algorithms:
        result = debug_algorithm(
            alg_class,
            alg_kwargs,
            alg_name,
            multilabel=False,
            max_train_samples=50,
            max_val_samples=10,
            max_classes=5,
        )
        results.append((alg_name, "single-label", result))

    # Test multi-label algorithms
    print("\n" + "=" * 60)
    print("MULTI-LABEL ALGORITHMS")
    print("=" * 60)
    for alg_class, alg_kwargs, alg_name in multilabel_algorithms:
        result = debug_algorithm(
            alg_class,
            alg_kwargs,
            alg_name,
            multilabel=True,
            max_train_samples=50,
            max_val_samples=10,
            max_classes=5,
        )
        results.append((alg_name, "multi-label", result))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for alg_name, label_type, passed in results:
        status = "? PASS" if passed else "? FAIL"
        print(f"{status}: {alg_name} ({label_type})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n?? All algorithms produce valid predictions!")
        return 0
    else:
        print("\n??  Some algorithms have validation errors. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
