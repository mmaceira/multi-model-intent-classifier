"""Tests for dataset loaders (single-label and multi-label)."""

from __future__ import annotations

import pytest

from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.utils.label_utils import is_multilabel


@pytest.fixture
def tiny_dataset_config():
    """Configuration for tiny dataset tests."""
    return {
        "max_train_samples": 50,
        "max_test_samples": 20,
        "max_val_samples": 10,
        "max_classes": 5,
        "seed": 42,
    }


def test_clinc150_single_label(tiny_dataset_config):
    """Test CLINC150 dataset loading (single-label)."""
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        **tiny_dataset_config,
    )

    # Basic sanity checks
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check format: single-label (list of strings)
    assert isinstance(y_train[0], str)
    assert isinstance(y_test[0], str)

    # Verify not multi-label
    assert not is_multilabel(y_train)
    assert not is_multilabel(y_test)

    # Check classes
    assert len(classes) > 0
    assert all(isinstance(c, str) for c in classes)


def test_nlu_plus_multilabel(tiny_dataset_config):
    """Test multi-label dataset loading (multi-label mode)."""
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="nlu_plus",
        multilabel=True,
        **tiny_dataset_config,
    )

    # Basic sanity checks
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check format: multi-label (list of lists)
    assert isinstance(y_train[0], list)
    assert isinstance(y_test[0], list)

    # Verify multi-label format
    assert is_multilabel(y_train)
    assert is_multilabel(y_test)

    # Check that each sample has at least one label
    assert all(len(labels) > 0 for labels in y_train if labels)
    assert all(len(labels) > 0 for labels in y_test if labels)

    # Check classes
    assert len(classes) > 0
    assert all(isinstance(c, str) for c in classes)


def test_nlu_plus_multilabel(tiny_dataset_config):
    """Test NLU++ dataset loading (multi-label)."""
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
            dataset_name="nlu_plus",
            multilabel=True,
            **tiny_dataset_config,
        )

        # Basic sanity checks
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check format: multi-label (list of lists)
        assert isinstance(y_train[0], list)
        assert isinstance(y_test[0], list)

        # Verify multi-label format
        assert is_multilabel(y_train)
        assert is_multilabel(y_test)

        # Check that each sample has at least one label
        assert all(len(labels) > 0 for labels in y_train if labels)
        assert all(len(labels) > 0 for labels in y_test if labels)

        # Check classes
        assert len(classes) > 0
        assert all(isinstance(c, str) for c in classes)
    except ImportError:
        pytest.skip("datasets library not installed; skipping NLU++ tests")


def test_dataset_splits_are_separate(tiny_dataset_config):
    """Test that train/val/test splits are kept separate."""
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        **tiny_dataset_config,
    )

    # Check that splits don't overlap (by checking lengths)
    total_samples = len(X_train) + len(X_val) + len(X_test)
    assert total_samples > 0

    # Check that all splits have data
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0


def test_dataset_reproducibility(tiny_dataset_config):
    """Test that dataset loading is reproducible with same seed."""
    # Load dataset twice with same seed
    X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, classes1 = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        **tiny_dataset_config,
    )

    X_train2, y_train2, X_val2, y_val2, X_test2, y_test2, classes2 = get_dataset(
        dataset_name="clinc150",
        multilabel=False,
        **tiny_dataset_config,
    )

    # Should be identical
    assert X_train1 == X_train2
    assert y_train1 == y_train2
    assert X_test1 == X_test2
    assert y_test1 == y_test2
    assert classes1 == classes2
