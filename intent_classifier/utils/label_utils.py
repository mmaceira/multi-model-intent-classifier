"""Label format utilities for single-label and multi-label classification.

This module provides utilities for detecting and converting between single-label
and multi-label formats, enabling seamless support for both classification types
while maintaining backward compatibility.

Functions:
- is_multilabel: Detect if labels are in multi-label format
- to_multilabel_format: Convert single-label to multi-label format
- to_singlelabel_format: Convert multi-label to single-label format
- binarize_labels: Convert labels to sklearn MultiLabelBinarizer format
- multilabel_predictions_from_proba: Extract multi-label predictions from probabilities
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def is_multilabel(y: Sequence[Any]) -> bool:
    """Detect if labels are in multi-label format.

    Format detection logic:
    - Single-label: y = ["class1", "class2", "class1"]  (list of strings)
    - Multi-label: y = [["class1"], ["class2"], ["class1", "class2"]]  (list of lists)

    Multi-label is detected when:
    1. At least one sample is a sequence (list/tuple/array) containing multiple labels
    2. OR samples are already in list-of-lists format

    Args:
        y: Sequence of labels (can be single-label or multi-label format)

    Returns:
        True if multi-label format detected, False otherwise

    Examples:
        >>> is_multilabel(["class1", "class2", "class1"])
        False

        >>> is_multilabel([["class1"], ["class2"], ["class1", "class2"]])
        True

        >>> is_multilabel([np.array(["class1"]), np.array(["class2"])])
        True
    """
    if len(y) == 0:
        return False

    # Check first few samples to determine format (sample for efficiency)
    sample_size = min(10, len(y))
    for i in range(sample_size):
        sample = y[i]

        # Check if sample is a sequence (list, tuple, array) but not a string
        if isinstance(sample, (list, tuple, np.ndarray)):
            # For numpy arrays: check dimensions and length
            if isinstance(sample, np.ndarray):
                # Multi-dimensional array or array with multiple elements = multi-label
                if sample.ndim > 1 or len(sample) > 1:
                    return True
            # For lists/tuples: check if it contains multiple labels
            elif len(sample) > 1:
                return True

    return False


def to_multilabel_format(y: Sequence[Any], classes: Sequence[str] | None = None) -> list[list[str]]:
    """Convert labels to multi-label format (list of lists).

    Converts both single-label and multi-label inputs to consistent multi-label format:
    - Single-label input: ["class1", "class2"] -> [["class1"], ["class2"]]
    - Multi-label input: [["class1"], ["class2"]] -> [["class1"], ["class2"]] (no change)

    Args:
        y: Sequence of labels (can be single-label or multi-label format)
        classes: Optional list of all possible classes (for validation, currently unused)

    Returns:
        List of lists, where each inner list contains one or more labels.
        Each sample is guaranteed to be a list of strings.

    Examples:
        >>> to_multilabel_format(["class1", "class2"])
        [["class1"], ["class2"]]

        >>> to_multilabel_format([["class1"], ["class2"]])
        [["class1"], ["class2"]]
    """
    y_multilabel = []
    for label in y:
        if isinstance(label, (list, tuple)):
            # Already in multi-label format: convert to list of strings
            y_multilabel.append([str(label_item) for label_item in label])
        elif isinstance(label, np.ndarray):
            # NumPy array: check if it's indices or binary vector
            if label.ndim == 1 and label.dtype in (np.int_, np.int64, np.int32, np.int16, np.int8):
                # Looks like indices - this shouldn't happen in normal flow
                # Convert indices to strings (warning: this loses class name information)
                y_multilabel.append([str(int(label_item)) for label_item in label])
            elif label.ndim == 1:
                # 1D array of other type - treat as single label
                if label.size == 1:
                    y_multilabel.append([str(label.item())])
                else:
                    # Multiple elements - convert each to string
                    y_multilabel.append([str(label_item) for label_item in label])
            else:
                # Multi-dimensional array - flatten and convert
                y_multilabel.append([str(label_item) for label_item in label.flatten()])
        else:
            # Single label: wrap in list to make it multi-label format
            y_multilabel.append([str(label)])

    return y_multilabel


def to_singlelabel_format(y: Sequence[Sequence[str]], strategy: str = "first") -> list[str]:
    """Convert multi-label format to single-label format.

    Args:
        y: Sequence of label sequences (multi-label format)
        strategy: Strategy for selecting single label. Options:
            - "first": Use first label (default)
            - "random": Randomly select one label
            - "most_common": Select most common label across all samples

    Returns:
        List of single labels

    Examples:
        >>> to_singlelabel_format([["class1"], ["class2"], ["class1", "class2"]])
        ["class1", "class2", "class1"]

        >>> to_singlelabel_format([["class1", "class2"]], strategy="first")
        ["class1"]
    """
    import random

    if strategy == "first":
        return [labels[0] if labels else "" for labels in y]
    elif strategy == "random":
        random.seed(42)
        return [random.choice(labels) if labels else "" for labels in y]
    elif strategy == "most_common":
        # Count all labels
        from collections import Counter

        all_labels: list[str] = []
        for labels in y:
            all_labels.extend(labels)
        if not all_labels:
            return [""] * len(y)
        most_common = Counter(all_labels).most_common(1)[0][0]
        # For each sample, pick the most common label if present, else first
        result = []
        for labels in y:
            if most_common in labels:
                result.append(most_common)
            elif labels:
                result.append(labels[0])
            else:
                result.append("")
        return result
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def binarize_labels(
    y_multilabel: Sequence[Sequence[str]], classes: Sequence[str] | None = None
) -> tuple[np.ndarray, MultiLabelBinarizer]:
    """Convert multi-label format (list of lists) to binary matrix for sklearn.

    This function converts multi-label format to sklearn's expected binary matrix format:
    - Input: [["class1"], ["class2"], ["class1", "class2"]]  (list of lists)
    - Output: [[1, 0], [0, 1], [1, 1]]  (binary matrix, shape: n_samples x n_classes)

    Args:
        y_multilabel: Sequence of label sequences in multi-label format (list of lists)
        classes: Optional list of all classes. If None, inferred from y_multilabel.

    Returns:
        Tuple of (y_binary, binarizer) where:
        - y_binary: numpy array of shape (n_samples, n_classes) with 0/1 values
        - binarizer: Fitted MultiLabelBinarizer instance (can be used to convert back)

    Examples:
        >>> y_multilabel = [["class1"], ["class2"], ["class1", "class2"]]
        >>> y_binary, binarizer = binarize_labels(y_multilabel)
        >>> y_binary.shape
        (3, 2)
        >>> y_binary
        array([[1, 0], [0, 1], [1, 1]])
    """
    binarizer = MultiLabelBinarizer(classes=classes, sparse_output=False)
    y_binary = binarizer.fit_transform(y_multilabel)
    return y_binary, binarizer


def multilabel_predictions_from_proba(
    proba: np.ndarray,
    classes: Sequence[str],
    threshold: float = 0.5,
) -> list[list[str]]:
    """Extract multi-label predictions from probability matrix using threshold.

    Args:
        proba: Probability matrix of shape (n_samples, n_classes)
        classes: List of class labels corresponding to columns
        threshold: Probability threshold for including a label (default: 0.5)

    Returns:
        List of lists, where each inner list contains labels with prob >= threshold

    Examples:
        >>> proba = np.array([[0.8, 0.3], [0.4, 0.9]])
        >>> classes = ["class1", "class2"]
        >>> multilabel_predictions_from_proba(proba, classes, threshold=0.5)
        [["class1"], ["class2"]]
    """
    predictions = []
    for row in proba:
        # Get indices where probability >= threshold
        indices = np.where(row >= threshold)[0]
        # Convert to labels
        labels = [classes[i] for i in indices]
        predictions.append(labels)
    return predictions


def multilabel_predictions_from_binary(
    y_binary: np.ndarray, classes: Sequence[str]
) -> list[list[str]]:
    """Convert binary matrix (from sklearn) back to multi-label format (list of lists).

    This function converts sklearn's binary matrix format back to multi-label format:
    - Input: [[1, 0], [0, 1], [1, 1]]  (binary matrix, shape: n_samples x n_classes)
    - Output: [["class1"], ["class2"], ["class1", "class2"]]  (list of lists)

    Args:
        y_binary: Binary matrix of shape (n_samples, n_classes) with 0/1 values
        classes: List of class labels corresponding to columns in y_binary

    Returns:
        List of lists, where each inner list contains labels with value 1 in the binary matrix

    Examples:
        >>> y_binary = np.array([[1, 0], [0, 1], [1, 1]])
        >>> classes = ["class1", "class2"]
        >>> multilabel_predictions_from_binary(y_binary, classes)
        [["class1"], ["class2"], ["class1", "class2"]]
    """
    y_multilabel = []
    for row in y_binary:
        # Find indices where value is 1 (label is present)
        label_indices = np.where(row == 1)[0]
        # Convert indices to actual label strings
        labels = [classes[i] for i in label_indices]
        y_multilabel.append(labels)
    return y_multilabel
