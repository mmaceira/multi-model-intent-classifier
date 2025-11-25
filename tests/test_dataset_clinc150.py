import pytest

try:
    from datasets import load_dataset  # noqa: F401
except ImportError:
    pytest.skip("datasets not installed; skipping CLINC150 tests", allow_module_level=True)

from intent_classifier.datasets.dataset import get_dataset


def test_clinc150_loader_basic():
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

    # Merge validation into training for compatibility with existing tests
    X_train = X_train + X_val
    y_train = y_train + y_val

    # Basic sanity checks
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Utterances should be non-empty strings
    assert isinstance(X_train[0], str)
    assert X_train[0].strip() != ""

    # Labels should be strings
    assert isinstance(y_train[0], str)
    assert y_train[0].strip() != ""

    # There should be multiple distinct intents
    assert len(set(y_train)) > 10


def test_tiny_training_smoke(tmp_path, monkeypatch):
    """End-to-end smoke test: train a simple classifier on a small CLINC150 subset."""
    from intent_classifier.algorithms.linear_svm import LinearSVMClassifier

    # Use a small subset for quick testing
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        max_train_samples=300,
        max_test_samples=100,
        max_classes=10,
        seed=123,
    )

    # Merge validation into training for compatibility with existing tests
    X_train = X_train + X_val
    y_train = y_train + y_val

    # Train a simple classifier
    clf = LinearSVMClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Basic assertions
    assert len(y_pred) == len(y_test)
    assert all(isinstance(pred, str) for pred in y_pred)
    assert all(pred in classes for pred in y_pred)
