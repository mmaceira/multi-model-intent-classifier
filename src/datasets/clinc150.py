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

from typing import List, Tuple

from datasets import load_dataset


def load_clinc150(
    use_oos: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Load the CLINC150 dataset (clinc_oos, config 'plus').

    Parameters
    ----------
    use_oos:
        If True, include OOS (out-of-scope) examples as an extra class label.
        If False, drop OOS examples.

    Returns
    -------
    X_train, y_train, X_test, y_test, classes
    """
    # 'plus' configuration has train/validation/test splits and oos examples
    ds = load_dataset("clinc_oos", "plus")

    def _extract(split_name: str):
        split = ds[split_name]
        texts: List[str] = []
        labels: List[str] = []
        for row in split:
            text = row["text"]
            intent = row["intent"]
            # OOS indicator exists in 'intent' and/or 'domain'; we treat it explicitly.
            # In this dataset, OOS is usually represented by intent "oos".
            if (not use_oos) and intent == "oos":
                # Skip out-of-scope examples if we don't want them
                continue
            texts.append(text)
            labels.append(intent)
        return texts, labels

    X_train, y_train = _extract("train")
    X_valid, y_valid = _extract("validation")
    X_test, y_test = _extract("test")

    # Optionally, merge validation into train for simplicity
    X_train_extended = X_train + X_valid
    y_train_extended = y_train + y_valid

    # Get all unique classes from both train and test
    classes = sorted(set(y_train_extended + y_test))

    return X_train_extended, y_train_extended, X_test, y_test, classes
