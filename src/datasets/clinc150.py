"""
Dataset loader for the CLINC150 intent classification dataset.

Uses the HuggingFace `clinc_oos` dataset (subset "plus"):

- 150 in-scope intents across 10 domains

- Optional out-of-scope (OOS) examples, which we can either use or ignore.

This loader returns:

    X_train, X_test: list[str]   (utterances)

    y_train, y_test: list[str]   (intent labels as strings)

"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Tuple

from datasets import load_dataset


def load_clinc150(
    use_oos: bool = False,
    max_train_samples: int = None,
    max_test_samples: int = None,
    max_classes: int = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Load the CLINC150 dataset (clinc_oos, config 'plus').

    Parameters
    ----------
    use_oos:
        If True, include OOS (out-of-scope) examples as an extra class label.
        If False, drop OOS examples.
    max_train_samples:
        Maximum number of training samples to load. If None, loads all samples.
        Useful for quick testing with smaller datasets.
    max_test_samples:
        Maximum number of test samples to load. If None, loads all samples.
        Useful for quick testing with smaller datasets.
    max_classes:
        Maximum number of classes to include. If None, includes all classes.
        If specified, randomly selects max_classes classes and filters samples to only those classes.
        Useful for quick testing with fewer classes (e.g., 10 classes instead of 151).
    seed:
        Random seed for reproducibility when selecting classes and sampling.

    Returns
    -------
    X_train, y_train, X_test, y_test, classes
    """
    # 'plus' configuration has train/validation/test splits and oos examples
    ds = load_dataset("clinc_oos", "plus")

    def _extract(split_name: str):
        split = ds[split_name]
        # Get the intent names mapping from the dataset features
        intent_feature = split.features["intent"]
        intent_names = intent_feature.names if hasattr(intent_feature, "names") else None

        texts: List[str] = []
        labels: List[str] = []
        for row in split:
            text = row["text"]
            intent = row["intent"]

            # Convert numeric intent ID to intent name if available
            if intent_names is not None and isinstance(intent, int):
                intent_name = intent_names[intent]
            elif isinstance(intent, str):
                intent_name = intent
            else:
                # Fallback: convert to string
                intent_name = str(intent)

            # OOS indicator exists in 'intent' and/or 'domain'; we treat it explicitly.
            # In this dataset, OOS is usually represented by intent "oos".
            if (not use_oos) and intent_name == "oos":
                # Skip out-of-scope examples if we don't want them
                continue
            texts.append(text)
            labels.append(intent_name)
        return texts, labels

    X_train, y_train = _extract("train")
    X_valid, y_valid = _extract("validation")
    X_test, y_test = _extract("test")

    # Optionally, merge validation into train for simplicity
    X_train_extended = X_train + X_valid
    y_train_extended = y_train + y_valid

    # Get all unique classes from the full dataset
    all_classes = sorted(set(y_train_extended + y_test))

    # Limit number of classes if specified
    if max_classes is not None and len(all_classes) > max_classes:
        # Randomly select max_classes classes
        random.seed(seed)
        selected_classes = sorted(random.sample(all_classes, max_classes))

        # Filter samples to only include selected classes
        X_train_extended = [
            text
            for text, label in zip(X_train_extended, y_train_extended, strict=False)
            if label in selected_classes
        ]
        y_train_extended = [label for label in y_train_extended if label in selected_classes]
        X_test = [
            text for text, label in zip(X_test, y_test, strict=False) if label in selected_classes
        ]
        y_test = [label for label in y_test if label in selected_classes]

        classes = selected_classes
    else:
        classes = all_classes

    # Limit samples if specified (useful for quick testing)
    # Use stratified sampling to ensure all classes are represented
    if max_train_samples is not None and len(X_train_extended) > max_train_samples:
        X_train_extended, y_train_extended = _stratified_sample(
            X_train_extended, y_train_extended, max_train_samples, seed=seed
        )

    if max_test_samples is not None and len(X_test) > max_test_samples:
        X_test, y_test = _stratified_sample(X_test, y_test, max_test_samples, seed=seed)

    return X_train_extended, y_train_extended, X_test, y_test, classes


def _stratified_sample(
    texts: List[str], labels: List[str], max_samples: int, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Perform stratified sampling to ensure all classes are represented.

    This function:
    1. Groups samples by class
    2. Ensures at least 1 sample per class (if possible)
    3. Distributes remaining samples proportionally across classes
    4. Maintains class balance as much as possible

    Args:
        texts: List of text samples
        labels: List of corresponding labels
        max_samples: Maximum number of samples to return
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_texts, sampled_labels)
    """
    if max_samples >= len(texts):
        return texts, labels

    # Group samples by class
    class_to_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text, label in zip(texts, labels, strict=False):
        class_to_samples[label].append((text, label))

    num_classes = len(class_to_samples)
    class_names = list(class_to_samples.keys())

    # If we need fewer samples than classes, we can't represent all classes
    # In this case, just take samples from the first N classes
    if max_samples < num_classes:
        # Take at least 1 sample from each of the first max_samples classes
        sampled_texts = []
        sampled_labels = []
        for i, class_name in enumerate(class_names[:max_samples]):
            if class_to_samples[class_name]:
                text, label = class_to_samples[class_name][0]
                sampled_texts.append(text)
                sampled_labels.append(label)
        return sampled_texts, sampled_labels

    # Strategy: Ensure at least 1 sample per class, then distribute the rest
    # Calculate samples per class
    samples_per_class = max_samples // num_classes
    remainder = max_samples % num_classes

    sampled_texts = []
    sampled_labels = []

    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle each class's samples to randomize selection
    for class_name in class_names:
        class_samples = class_to_samples[class_name].copy()
        random.shuffle(class_samples)

        # Calculate how many samples to take from this class
        # Distribute remainder samples across first N classes
        num_to_take = samples_per_class
        if class_names.index(class_name) < remainder:
            num_to_take += 1

        # Take samples (but don't exceed what's available)
        num_to_take = min(num_to_take, len(class_samples))
        for i in range(num_to_take):
            text, label = class_samples[i]
            sampled_texts.append(text)
            sampled_labels.append(label)

    # Shuffle the final result to mix classes
    combined = list(zip(sampled_texts, sampled_labels, strict=False))
    random.shuffle(combined)
    sampled_texts, sampled_labels = zip(*combined, strict=False)

    return list(sampled_texts), list(sampled_labels)
