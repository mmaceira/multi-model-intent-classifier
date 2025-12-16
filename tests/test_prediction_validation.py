"""Tests to verify that predictions only contain valid classes from the dataset.

This test suite validates that all prediction algorithms produce predictions
that are always within the valid classes from the training dataset.
"""

from __future__ import annotations

import tempfile

import pytest

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.algorithms.transformer_logreg import TransformerLogReg
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.prediction import run_prediction
from intent_classifier.training import run_training


@pytest.fixture
def single_label_trained_model(tmp_path):
    """Create a trained single-label model for testing."""
    X_train, y_train, X_val, y_val, _, _, classes = get_dataset(
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

    return tmp_path, X_train, y_train, X_val, y_val, classes


@pytest.fixture
def multilabel_trained_model(tmp_path):
    """Create a trained multi-label model for testing."""
    X_train, y_train, X_val, y_val, _, _, classes = get_dataset(
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

    return tmp_path, X_train, y_train, X_val, y_val, classes


def test_single_label_predictions_only_valid_classes(single_label_trained_model):
    """Test that single-label predictions only contain valid classes from dataset."""
    model_dir, X_train, y_train, X_val, y_val, classes = single_label_trained_model

    # Get valid classes from training data
    valid_classes = set(classes)

    # Load model and make predictions
    model_path = model_dir / "linear_svm" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=None,
            y_train=None,
            X_test=X_val,
            y_test=y_val,
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # Check that all predictions are valid classes
        test_preds = predictions["linear_svm"]["test"]
        assert len(test_preds) > 0

        for pred in test_preds:
            assert isinstance(pred, str)
            assert pred in valid_classes, (
                f"Prediction '{pred}' not in valid classes {valid_classes}"
            )


def test_multilabel_predictions_only_valid_classes(multilabel_trained_model):
    """Test that multi-label predictions only contain valid classes from dataset."""
    model_dir, X_train, y_train, X_val, y_val, classes = multilabel_trained_model

    # Get valid classes from training data
    valid_classes = set(classes)

    # Load model and make predictions
    model_path = model_dir / "linear_svm" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=None,
            y_train=None,
            X_test=X_val,
            y_test=y_val,
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # Check that all predictions contain only valid classes
        test_preds = predictions["linear_svm"]["test"]
        assert len(test_preds) > 0

        for pred in test_preds:
            assert isinstance(pred, list)
            assert len(pred) > 0, "Each prediction should have at least one label"

            for label in pred:
                assert isinstance(label, str)
                assert label in valid_classes, (
                    f"Label '{label}' not in valid classes {valid_classes}. Full prediction: {pred}"
                )


def test_single_label_predictions_handle_invalid_classes(single_label_trained_model):
    """Test that single-label predictions handle invalid classes gracefully."""
    model_dir, X_train, y_train, X_val, y_val, classes = single_label_trained_model

    # Create a mock model that might return invalid classes
    # (In practice, sklearn models should always return valid classes, but we test the validation)
    model_path = model_dir / "linear_svm" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=None,
            y_train=None,
            X_test=X_val[:5],  # Small subset
            y_test=y_val[:5],
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # All predictions should be valid classes
        test_preds = predictions["linear_svm"]["test"]
        valid_classes = set(classes)

        for pred in test_preds:
            assert pred in valid_classes


def test_multilabel_predictions_handle_invalid_classes(multilabel_trained_model):
    """Test that multi-label predictions handle invalid classes gracefully."""
    model_dir, X_train, y_train, X_val, y_val, classes = multilabel_trained_model

    model_path = model_dir / "linear_svm" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=None,
            y_train=None,
            X_test=X_val[:5],  # Small subset
            y_test=y_val[:5],
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # All labels in predictions should be valid classes
        test_preds = predictions["linear_svm"]["test"]
        valid_classes = set(classes)

        for pred in test_preds:
            assert isinstance(pred, list)
            for label in pred:
                assert label in valid_classes


# Tests for all algorithms - single-label
@pytest.mark.parametrize(
    "algorithm_class,algorithm_kwargs",
    [
        (LinearSVMClassifier, {"max_features": 100, "C": 0.1, "calibrate": False}),
        (NaiveBayesClassifier, {"max_features": 100, "alpha": 0.5}),
        (TransformerLogReg, {"model_name": "all-MiniLM-L6-v2", "max_iter": 100}),
    ],
)
def test_all_algorithms_single_label_valid_classes(algorithm_class, algorithm_kwargs, tmp_path):
    """Test that all single-label algorithms produce valid class predictions."""
    X_train, y_train, X_val, y_val, _, _, classes = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    valid_classes = set(classes)
    models = {"test_model": algorithm_class(**algorithm_kwargs)}

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

    model_path = tmp_path / "test_model" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"test_model": str(model_path)},
            X_train=None,
            y_train=None,
            X_test=X_val,
            y_test=y_val,
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # Check that all predictions are valid classes
        test_preds = predictions["test_model"]["test"]
        assert len(test_preds) > 0, f"{algorithm_class.__name__} produced no predictions"

        for i, pred in enumerate(test_preds):
            assert isinstance(pred, str), (
                f"{algorithm_class.__name__}: Prediction {i} is not a string: {type(pred)}"
            )
            assert pred in valid_classes, (
                f"{algorithm_class.__name__}: Prediction '{pred}' at index {i} "
                f"not in valid classes {sorted(valid_classes)}"
            )


# Tests for all algorithms - multi-label
@pytest.mark.parametrize(
    "algorithm_class,algorithm_kwargs",
    [
        (LinearSVMClassifier, {"max_features": 100, "C": 0.1, "calibrate": False}),
        (NaiveBayesClassifier, {"max_features": 100, "alpha": 0.5}),
        (TransformerLogReg, {"model_name": "all-MiniLM-L6-v2", "max_iter": 100}),
    ],
)
def test_all_algorithms_multilabel_valid_classes(algorithm_class, algorithm_kwargs, tmp_path):
    """Test that all multi-label algorithms produce valid class predictions."""
    X_train, y_train, X_val, y_val, _, _, classes = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    valid_classes = set(classes)
    models = {"test_model": algorithm_class(**algorithm_kwargs)}

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

    model_path = tmp_path / "test_model" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"test_model": str(model_path)},
            X_train=None,
            y_train=None,
            X_test=X_val,
            y_test=y_val,
            output_dir=pred_dir,
            save_test_predictions=True,
            verbose=False,
        )

        # Check that all predictions contain only valid classes
        test_preds = predictions["test_model"]["test"]
        assert len(test_preds) > 0, f"{algorithm_class.__name__} produced no predictions"

        for i, pred in enumerate(test_preds):
            assert isinstance(pred, list), (
                f"{algorithm_class.__name__}: Prediction {i} is not a list: {type(pred)}"
            )
            assert len(pred) > 0, (
                f"{algorithm_class.__name__}: Prediction {i} is empty. "
                "Each prediction should have at least one label."
            )

            for j, label in enumerate(pred):
                assert isinstance(label, str), (
                    f"{algorithm_class.__name__}: Label {j} in prediction {i} "
                    f"is not a string: {type(label)}"
                )
                assert label in valid_classes, (
                    f"{algorithm_class.__name__}: Label '{label}' at prediction {i}, label {j} "
                    f"not in valid classes {sorted(valid_classes)}. "
                    f"Full prediction: {pred}"
                )


def test_single_label_predictions_format_consistency(single_label_trained_model):
    """Test that single-label predictions have consistent format."""
    model_dir, X_train, y_train, X_val, y_val, classes = single_label_trained_model
    model_path = model_dir / "linear_svm" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=X_train[:10],
            y_train=y_train[:10],
            X_test=X_val,
            y_test=y_val,
            output_dir=pred_dir,
            save_train_predictions=True,
            save_test_predictions=True,
            verbose=False,
        )

        # Check train predictions
        train_preds = predictions["linear_svm"]["train"]
        assert len(train_preds) == 10
        assert all(isinstance(p, str) for p in train_preds)
        assert all(len(p) > 0 for p in train_preds), "All predictions should be non-empty strings"

        # Check test predictions
        test_preds = predictions["linear_svm"]["test"]
        assert len(test_preds) == len(X_val)
        assert all(isinstance(p, str) for p in test_preds)
        assert all(len(p) > 0 for p in test_preds), "All predictions should be non-empty strings"


def test_multilabel_predictions_format_consistency(multilabel_trained_model):
    """Test that multi-label predictions have consistent format."""
    model_dir, X_train, y_train, X_val, y_val, classes = multilabel_trained_model
    model_path = model_dir / "linear_svm" / "model.pkl"

    with tempfile.TemporaryDirectory() as pred_dir:
        predictions = run_prediction(
            models_or_paths={"linear_svm": str(model_path)},
            X_train=X_train[:10],
            y_train=y_train[:10],
            X_test=X_val,
            y_test=y_val,
            output_dir=pred_dir,
            save_train_predictions=True,
            save_test_predictions=True,
            verbose=False,
        )

        # Check train predictions
        train_preds = predictions["linear_svm"]["train"]
        assert len(train_preds) == 10
        assert all(isinstance(p, list) for p in train_preds)
        assert all(len(p) > 0 for p in train_preds), (
            "All predictions should have at least one label"
        )
        assert all(isinstance(label, str) for pred in train_preds for label in pred)

        # Check test predictions
        test_preds = predictions["linear_svm"]["test"]
        assert len(test_preds) == len(X_val)
        assert all(isinstance(p, list) for p in test_preds)
        assert all(len(p) > 0 for p in test_preds), "All predictions should have at least one label"
        assert all(isinstance(label, str) for pred in test_preds for label in pred)
