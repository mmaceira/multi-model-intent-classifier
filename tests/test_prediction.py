"""Tests for prediction pipeline (single-label and multi-label)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.prediction import run_prediction
from intent_classifier.training import run_training


@pytest.fixture
def single_label_trained_model(tmp_path):
    """Create a trained single-label model for testing."""
    X_train, y_train, X_val, y_val, _, _, _ = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}

    run_training(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=tmp_path,
        save_models=True,
        verbose=False,
    )

    return tmp_path, X_train, y_train, X_val, y_val


@pytest.fixture
def multilabel_trained_model(tmp_path):
    """Create a trained multi-label model for testing."""
    X_train, y_train, X_val, y_val, _, _, _ = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}

    run_training(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=tmp_path,
        save_models=True,
        verbose=False,
    )

    return tmp_path, X_train, y_train, X_val, y_val


def test_prediction_single_label(single_label_trained_model):
    """Test prediction with single-label model."""
    model_dir, X_train, y_train, X_val, y_val = single_label_trained_model

    # Load model path
    model_path = model_dir / "linear_svm" / "model.pkl"

    # Run prediction
    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=X_train[:10],
            y_train=y_train[:10],
            X_test=X_val[:5],
            y_test=y_val[:5],
            output_dir=pred_dir,
            save_train_predictions=True,
            save_test_predictions=True,
            verbose=False,
        )

        # Check predictions structure
        assert "linear_svm" in predictions
        assert "train" in predictions["linear_svm"]
        assert "test" in predictions["linear_svm"]

        # Check prediction format (single-label: list of strings)
        train_preds = predictions["linear_svm"]["train"]
        test_preds = predictions["linear_svm"]["test"]

        assert len(train_preds) == 10
        assert len(test_preds) == 5
        assert all(isinstance(p, str) for p in train_preds)
        assert all(isinstance(p, str) for p in test_preds)

        # Check that CSV files were created
        assert (Path(pred_dir) / "linear_svm" / "train_predictions.csv").exists()
        assert (Path(pred_dir) / "linear_svm" / "test_predictions.csv").exists()


def test_prediction_multilabel(multilabel_trained_model):
    """Test prediction with multi-label model."""
    model_dir, X_train, y_train, X_val, y_val = multilabel_trained_model

    # Load model path
    model_path = model_dir / "linear_svm" / "model.pkl"

    # Run prediction
    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=X_train[:10],
            y_train=y_train[:10],
            X_test=X_val[:5],
            y_test=y_val[:5],
            output_dir=pred_dir,
            save_train_predictions=True,
            save_test_predictions=True,
            verbose=False,
        )

        # Check predictions structure
        assert "linear_svm" in predictions
        assert "train" in predictions["linear_svm"]
        assert "test" in predictions["linear_svm"]

        # Check prediction format (multi-label: list of lists)
        train_preds = predictions["linear_svm"]["train"]
        test_preds = predictions["linear_svm"]["test"]

        assert len(train_preds) == 10
        assert len(test_preds) == 5
        assert all(isinstance(p, list) for p in train_preds)
        assert all(isinstance(p, list) for p in test_preds)

        # Verify multi-label format: each prediction is a list with at least one string label
        assert all(len(p) > 0 for p in train_preds), (
            "All predictions should have at least one label"
        )
        assert all(len(p) > 0 for p in test_preds), "All predictions should have at least one label"
        assert all(isinstance(label, str) for pred in train_preds for label in pred)
        assert all(isinstance(label, str) for pred in test_preds for label in pred)

        # Check that CSV files were created
        assert (Path(pred_dir) / "linear_svm" / "train_predictions.csv").exists()
        assert (Path(pred_dir) / "linear_svm" / "test_predictions.csv").exists()


def test_prediction_with_model_object(single_label_trained_model):
    """Test prediction with model object instead of path."""
    model_dir, X_train, y_train, X_val, y_val = single_label_trained_model

    # Load model directly
    import cloudpickle

    model_path = model_dir / "linear_svm" / "model.pkl"
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)

    # Run prediction with model object
    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": model},
            X_train=None,
            y_train=None,
            X_test=X_val[:5],
            y_test=y_val[:5],
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # Should work the same way
        assert "linear_svm" in predictions
        assert "test" in predictions["linear_svm"]


def test_prediction_validation_errors():
    """Test that prediction raises errors for invalid inputs."""
    # Create a dummy model
    model = LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)
    X_train, y_train, _, _, _, _, _ = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=10,
        max_test_samples=5,
        seed=42,
    )
    model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as pred_dir:
        # Should raise error if X_train provided but y_train is not
        with pytest.raises(ValueError, match="X_train and y_train must both be provided"):
            run_prediction(
                models_or_paths={"test_model": model},
                X_train=X_train[:5],
                y_train=None,
                X_test=None,
                y_test=None,
                output_dir=pred_dir,
                verbose=False,
            )

        # Should raise error if no data provided
        with pytest.raises(ValueError, match="At least one of X_train or X_test must be provided"):
            run_prediction(
                models_or_paths={"test_model": model},
                X_train=None,
                y_train=None,
                X_test=None,
                y_test=None,
                output_dir=pred_dir,
                verbose=False,
            )


def test_prediction_multiple_models(single_label_trained_model):
    """Test prediction with multiple models."""
    model_dir, X_train, y_train, X_val, y_val = single_label_trained_model

    # Train another model
    models = {"naive_bayes": NaiveBayesClassifier(max_features=100, alpha=0.5)}
    run_training(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=model_dir,
        save_models=True,
        verbose=False,
    )

    # Run prediction with both models
    model_paths = {
        "linear_svm": str(model_dir / "linear_svm" / "model.pkl"),
        "naive_bayes": str(model_dir / "naive_bayes" / "model.pkl"),
    }

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths=model_paths,
            X_train=None,
            y_train=None,
            X_test=X_val[:5],
            y_test=y_val[:5],
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # Both models should have predictions
        assert len(predictions) == 2
        assert "linear_svm" in predictions
        assert "naive_bayes" in predictions

        # Both should have test predictions
        assert "test" in predictions["linear_svm"]
        assert "test" in predictions["naive_bayes"]
