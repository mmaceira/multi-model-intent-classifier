"""
Tiny end-to-end smoke test to ensure dataset + classifier wiring stays intact.
"""

from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
from intent_classifier.datasets.dataset import get_dataset


def test_tiny_training_smoke():
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_dataset(
        dataset_name="clinc150",
        max_train_samples=200,
        max_test_samples=60,
        max_classes=15,
        seed=123,
    )

    # Merge validation into training for compatibility with existing tests
    X_train = X_train + X_val
    y_train = y_train + y_val

    assert X_train and y_train and X_test and y_test

    clf = LinearSVMClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert len(y_pred) == len(y_test)
