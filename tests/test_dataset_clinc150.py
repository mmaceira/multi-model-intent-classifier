import pytest

try:
    from datasets import load_dataset  # noqa: F401
except ImportError:
    pytest.skip("datasets not installed; skipping CLINC150 tests", allow_module_level=True)

from src.datasets.dataset import get_dataset


def test_clinc150_loader_basic():
    X_train, y_train, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

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
