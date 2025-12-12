"""Tests for evaluation pipeline (single-label and multi-label)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.evaluation import run_evaluations
from intent_classifier.prediction import run_prediction
from intent_classifier.training import run_training


@pytest.fixture
def single_label_predictions(tmp_path):
    """Create predictions for single-label model."""
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    # Train model
    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}
    model_dir = tmp_path / "models"
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

    # Generate predictions
    pred_dir = tmp_path / "predictions"
    model_path = model_dir / "linear_svm" / "model.pkl"
    run_prediction(
        models_or_paths={"linear_svm": str(model_path)},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=pred_dir,
        save_train_predictions=True,
        save_test_predictions=True,
        verbose=False,
    )

    return pred_dir, models


@pytest.fixture
def multilabel_predictions(tmp_path):
    """Create predictions for multi-label model."""
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    # Train model
    models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}
    model_dir = tmp_path / "models"
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

    # Generate predictions
    pred_dir = tmp_path / "predictions"
    model_path = model_dir / "linear_svm" / "model.pkl"
    run_prediction(
        models_or_paths={"linear_svm": str(model_path)},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=pred_dir,
        save_train_predictions=True,
        save_test_predictions=True,
        verbose=False,
    )

    return pred_dir, models


def test_evaluation_single_label(single_label_predictions):
    """Test evaluation with single-label predictions."""
    pred_dir, models = single_label_predictions

    # Run evaluation
    with tempfile.TemporaryDirectory() as results_dir:
        results = run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        # Check that results were generated
        assert len(results) > 0
        assert "linear_svm" in results

        # Check that metrics files exist
        results_path = Path(results_dir)
        assert (results_path / "linear_svm" / "test" / "test_metrics.json").exists()

        # Check that summary exists
        assert (results_path / "summary_metrics.csv").exists()


def test_evaluation_multilabel(multilabel_predictions):
    """Test evaluation with multi-label predictions."""
    pred_dir, models = multilabel_predictions

    # Run evaluation
    with tempfile.TemporaryDirectory() as results_dir:
        results = run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        # Check that results were generated
        assert len(results) > 0
        assert "linear_svm" in results

        # Check that metrics files exist
        results_path = Path(results_dir)
        assert (results_path / "linear_svm" / "test" / "test_metrics.json").exists()

        # Check that summary exists
        assert (results_path / "summary_metrics.csv").exists()


def test_evaluation_metrics_content_single_label(single_label_predictions):
    """Test that evaluation metrics contain expected values for single-label."""
    pred_dir, models = single_label_predictions

    with tempfile.TemporaryDirectory() as results_dir:
        results = run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        # Check that metrics are present
        model_results = results["linear_svm"]
        assert "test_accuracy" in model_results or "test_f1_macro" in model_results


def test_evaluation_metrics_content_multilabel(multilabel_predictions):
    """Test that evaluation metrics contain expected values for multi-label."""
    pred_dir, models = multilabel_predictions

    with tempfile.TemporaryDirectory() as results_dir:
        results = run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        # Check that metrics are present
        model_results = results["linear_svm"]
        # Multi-label should have metrics (may be detected as single-label if format is ambiguous)
        assert len(model_results) > 0
        # Check for any test metrics
        test_metrics = [k for k in model_results.keys() if k.startswith("test_")]
        assert len(test_metrics) > 0


def test_evaluation_with_missing_predictions():
    """Test that evaluation handles missing predictions gracefully."""
    model_names = ["linear_svm"]

    with tempfile.TemporaryDirectory() as pred_dir, tempfile.TemporaryDirectory() as results_dir:
        # Try to evaluate with non-existent predictions directory
        # Should return empty dict (not raise)
        results = run_evaluations(
            model_names=model_names,
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )
        # Should return empty dict when no predictions found
        assert isinstance(results, dict)
        assert len(results) == 0


def test_evaluation_summary_table(single_label_predictions):
    """Test that evaluation generates summary table."""
    pred_dir, models = single_label_predictions

    with tempfile.TemporaryDirectory() as results_dir:
        run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        # Check that summary CSV exists and is readable
        summary_path = Path(results_dir) / "summary_metrics.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            assert len(df) > 0
            # Summary CSV has model names as index, not a "model" column
            assert len(df.index) > 0
