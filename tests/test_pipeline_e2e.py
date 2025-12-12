"""End-to-end tests for the complete pipeline (training -> prediction -> evaluation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.algorithms.naive_bayes import NaiveBayesClassifier
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.evaluation import run_evaluations
from intent_classifier.prediction import run_prediction
from intent_classifier.training import run_training
from intent_classifier.utils.label_utils import is_multilabel


def test_pipeline_single_label_e2e():
    """Test complete pipeline for single-label classification."""
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    # Verify single-label format
    assert not is_multilabel(y_train)
    assert not is_multilabel(y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        models_dir = tmp_path / "models"
        pred_dir = tmp_path / "predictions"
        results_dir = tmp_path / "results"

        # Step 1: Training
        models = {
            "linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False),
            "naive_bayes": NaiveBayesClassifier(max_features=100, alpha=0.5),
        }

        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=models_dir,
            save_models=True,
            verbose=False,
        )

        assert len(trained) == 2
        assert (models_dir / "linear_svm" / "model.pkl").exists()
        assert (models_dir / "naive_bayes" / "model.pkl").exists()

        # Step 2: Prediction
        model_paths = {
            "linear_svm": str(models_dir / "linear_svm" / "model.pkl"),
            "naive_bayes": str(models_dir / "naive_bayes" / "model.pkl"),
        }

        predictions = run_prediction(
            models_or_paths=model_paths,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=pred_dir,
            save_train_predictions=True,
            save_test_predictions=True,
            verbose=False,
        )

        assert len(predictions) == 2
        assert "linear_svm" in predictions
        assert "naive_bayes" in predictions

        # Verify predictions are single-label
        assert not is_multilabel(predictions["linear_svm"]["test"])
        assert not is_multilabel(predictions["naive_bayes"]["test"])

        # Check CSV files exist
        assert (pred_dir / "linear_svm" / "test_predictions.csv").exists()
        assert (pred_dir / "naive_bayes" / "test_predictions.csv").exists()

        # Step 3: Evaluation
        results = run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        assert len(results) == 2
        assert "linear_svm" in results
        assert "naive_bayes" in results

        # Check that metrics files exist
        assert (results_dir / "linear_svm" / "test" / "test_metrics.json").exists()
        assert (results_dir / "naive_bayes" / "test" / "test_metrics.json").exists()
        assert (results_dir / "summary_metrics.csv").exists()


def test_pipeline_multilabel_e2e():
    """Test complete pipeline for multi-label classification."""
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    # Verify multi-label format
    assert is_multilabel(y_train)
    assert is_multilabel(y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        models_dir = tmp_path / "models"
        pred_dir = tmp_path / "predictions"
        results_dir = tmp_path / "results"

        # Step 1: Training
        models = {
            "linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False),
            "naive_bayes": NaiveBayesClassifier(max_features=100, alpha=0.5),
        }

        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=models_dir,
            save_models=True,
            verbose=False,
        )

        assert len(trained) == 2
        assert (models_dir / "linear_svm" / "model.pkl").exists()
        assert (models_dir / "naive_bayes" / "model.pkl").exists()

        # Step 2: Prediction
        model_paths = {
            "linear_svm": str(models_dir / "linear_svm" / "model.pkl"),
            "naive_bayes": str(models_dir / "naive_bayes" / "model.pkl"),
        }

        predictions = run_prediction(
            models_or_paths=model_paths,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=pred_dir,
            save_train_predictions=True,
            save_test_predictions=True,
            verbose=False,
        )

        assert len(predictions) == 2
        assert "linear_svm" in predictions
        assert "naive_bayes" in predictions

        # Verify predictions are multi-label format (list of lists)
        assert isinstance(predictions["linear_svm"]["test"], list)
        assert isinstance(predictions["linear_svm"]["test"][0], list)
        assert isinstance(predictions["naive_bayes"]["test"], list)
        assert isinstance(predictions["naive_bayes"]["test"][0], list)

        # Check CSV files exist
        assert (pred_dir / "linear_svm" / "test_predictions.csv").exists()
        assert (pred_dir / "naive_bayes" / "test_predictions.csv").exists()

        # Step 3: Evaluation
        results = run_evaluations(
            model_names=list(models.keys()),
            artefacts_root=pred_dir,
            output_dir=results_dir,
            verbose=False,
        )

        assert len(results) == 2
        assert "linear_svm" in results
        assert "naive_bayes" in results

        # Check that metrics files exist
        assert (results_dir / "linear_svm" / "test" / "test_metrics.json").exists()
        assert (results_dir / "naive_bayes" / "test" / "test_metrics.json").exists()
        assert (results_dir / "summary_metrics.csv").exists()


def test_pipeline_deploy_workflow():
    """Test deployment workflow: load trained model and make predictions."""
    # Train a model
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        models_dir = tmp_path / "models"

        # Train
        models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=models_dir,
            save_models=True,
            verbose=False,
        )

        # Simulate deployment: load model and make predictions
        import cloudpickle

        model_path = models_dir / "linear_svm" / "model.pkl"
        with open(model_path, "rb") as f:
            deployed_model = cloudpickle.load(f)

        # Make predictions on new data
        new_predictions = deployed_model.predict(X_test[:5])
        assert len(new_predictions) == 5
        assert all(isinstance(p, str) for p in new_predictions)

        # Verify predictions are valid
        assert all(p in trained["linear_svm"].classes_ for p in new_predictions)


def test_pipeline_multilabel_deploy_workflow():
    """Test deployment workflow for multi-label: load trained model and make predictions."""
    # Train a model
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        max_train_samples=50,
        max_test_samples=20,
        max_val_samples=10,
        max_classes=5,
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        models_dir = tmp_path / "models"

        # Train
        models = {"linear_svm": LinearSVMClassifier(max_features=100, C=0.1, calibrate=False)}
        trained = run_training(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=models_dir,
            save_models=True,
            verbose=False,
        )

        # Simulate deployment: load model and make predictions
        import cloudpickle

        model_path = models_dir / "linear_svm" / "model.pkl"
        with open(model_path, "rb") as f:
            deployed_model = cloudpickle.load(f)

        # Make predictions on new data
        new_predictions = deployed_model.predict(X_test[:5])
        assert len(new_predictions) == 5

        # Verify predictions are multi-label format (list of lists)
        assert isinstance(new_predictions, list)
        assert all(isinstance(p, list) for p in new_predictions)
