"""Tests for training pipeline (single-label and multi-label)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.training import run_training
from intent_classifier.utils.label_utils import is_multilabel


@pytest.fixture
def single_label_data():
    """Get single-label dataset for testing."""
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )
    return X_train, y_train, X_val, y_val, classes


@pytest.fixture
def multilabel_data():
    """Get multi-label dataset for testing."""
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )
    return X_train, y_train, X_val, y_val, classes


def test_training_single_label(single_label_data):
    """Test training with single-label data."""
    X_train, y_train, X_val, y_val, classes = single_label_data

    # Verify single-label format
    assert not is_multilabel(y_train)
    assert not is_multilabel(y_val)

    # Create models
    models = {
        "naive_bayes": NaiveBayesClassifier(max_features=100, alpha=0.5),
        "linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False),
    }

    # Train with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=tmpdir,
            save_models=True,
            verbose=False,
        )

        # Check that models were trained
        assert len(trained) == len(models)
        assert "naive_bayes" in trained
        assert "linear_svm" in trained

        # Check that models are fitted
        assert hasattr(trained["naive_bayes"], "predict")
        assert hasattr(trained["linear_svm"], "predict")

        # Check that model files were saved
        assert (Path(tmpdir) / "naive_bayes" / "model.pkl").exists()
        assert (Path(tmpdir) / "linear_svm" / "model.pkl").exists()

        # Check that training times were saved
        assert (Path(tmpdir) / "training_times.txt").exists()


def test_training_multilabel(multilabel_data):
    """Test training with multi-label data."""
    X_train, y_train, X_val, y_val, classes = multilabel_data

    # Verify multi-label format
    assert is_multilabel(y_train)
    assert is_multilabel(y_val)

    # Create models (these should support multi-label)
    models = {
        "naive_bayes": NaiveBayesClassifier(max_features=100, alpha=0.5),
        "linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False),
    }

    # Train with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=tmpdir,
            save_models=True,
            verbose=False,
        )

        # Check that models were trained
        assert len(trained) == len(models)
        assert "naive_bayes" in trained
        assert "linear_svm" in trained

        # Check that models are fitted
        assert hasattr(trained["naive_bayes"], "predict")
        assert hasattr(trained["linear_svm"], "predict")

        # Test predictions are multi-label format (list of lists)
        sample_pred = trained["naive_bayes"].predict(X_train[:1])
        # Multi-label predictions should be list of lists
        assert isinstance(sample_pred, list)
        assert isinstance(sample_pred[0], list)

        # Check that model files were saved
        assert (Path(tmpdir) / "naive_bayes" / "model.pkl").exists()
        assert (Path(tmpdir) / "linear_svm" / "model.pkl").exists()


def test_training_validation_set_consistency():
    """Test that training raises error if validation set is inconsistently provided."""
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should raise ValueError if X_val is provided but y_val is not
        with pytest.raises(ValueError, match="X_val and y_val must both be provided"):
            run_training(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_val=X_train[:10],  # Provide X_val but not y_val
                y_val=None,
                output_dir=tmpdir,
                verbose=False,
            )


def test_training_without_validation_set(single_label_data):
    """Test training without validation set."""
    X_train, y_train, _, _, _ = single_label_data

    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}

    with tempfile.TemporaryDirectory() as tmpdir:
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=None,
            y_val=None,
            output_dir=tmpdir,
            save_models=True,
            verbose=False,
        )

        # Should still train successfully
        assert len(trained) == 1
        assert "linear_svm" in trained


def test_training_multiple_models(single_label_data):
    """Test training multiple models in one call."""
    X_train, y_train, X_val, y_val, _ = single_label_data

    models = {
        "naive_bayes": NaiveBayesClassifier(max_features=100, alpha=0.5),
        "linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=tmpdir,
            save_models=True,
            verbose=False,
        )

        # All models should be trained
        assert len(trained) == 2
        assert all(name in trained for name in models)

        # All model files should exist
        for name in models:
            assert (Path(tmpdir) / name / "model.pkl").exists()


def test_training_predictions_work(single_label_data):
    """Test that trained models can make predictions."""
    X_train, y_train, X_val, y_val, _ = single_label_data

    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}

    with tempfile.TemporaryDirectory() as tmpdir:
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=tmpdir,
            save_models=True,
            verbose=False,
        )

        # Make predictions
        predictions = trained["linear_svm"].predict(X_train[:5])
        assert len(predictions) == 5
        assert all(isinstance(p, str) for p in predictions)
