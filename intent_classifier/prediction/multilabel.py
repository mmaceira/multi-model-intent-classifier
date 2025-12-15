"""Multi-label prediction runner implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intent_classifier.prediction.base import BasePredictionRunner


class MultiLabelPredictionRunner(BasePredictionRunner):
    """Prediction runner for multi-label classification tasks.

    This class handles prediction workflows specific to multi-label classification,
    where each sample can have multiple labels.
    """

    def _get_classes_from_estimator(self, estimator: Any) -> list[str] | None:
        """Extract class names from estimator.

        Supports multiple estimator types:
        - Standard sklearn: _classes or classes_
        - RAG models: labels attribute (direct or via adapter)

        Args:
            estimator: The model estimator

        Returns:
            List of class names as strings, or None if not available
        """
        classes = None

        # Priority: _classes (names) > classes_ (may be indices for some models)
        if hasattr(estimator, "_classes"):
            classes = estimator._classes
        elif hasattr(estimator, "classes_"):
            classes = estimator.classes_
        # For RAG models: check labels attribute (direct or via adapter)
        elif hasattr(estimator, "labels"):
            classes = estimator.labels
        elif hasattr(estimator, "rag") and hasattr(estimator.rag, "labels"):
            # RAG models wrapped in adapter
            classes = estimator.rag.labels
        elif hasattr(estimator, "rag_clf") and hasattr(estimator.rag_clf, "labels"):
            # Alternative adapter attribute name
            classes = estimator.rag_clf.labels

        # Convert to list of strings
        if classes is not None:
            if isinstance(classes, np.ndarray):
                classes = classes.tolist()
            elif not isinstance(classes, (list, tuple)):
                classes = list(classes)

            # For multilabel RAG models: labels may be string representations of lists
            # (e.g., "['Alta', 'circuit']") but predictions are individual tags
            # Extract individual tags from string representations of lists
            all_tags: set[str] = set()
            for c in classes:
                c_str = str(c)
                # Check if it's a string representation of a list (starts with
                # '[' and contains quotes)
                if c_str.startswith("[") and ("'" in c_str or '"' in c_str):
                    try:
                        # Try to parse as a list (using ast.literal_eval for safety)
                        import ast

                        parsed = ast.literal_eval(c_str)
                        if isinstance(parsed, list):
                            # Extract individual tags from the list
                            all_tags.update(str(tag) for tag in parsed)
                        else:
                            # Not a list, use as-is
                            all_tags.add(c_str)
                    except (ValueError, SyntaxError):
                        # Can't parse, use as-is
                        all_tags.add(c_str)
                else:
                    # Not a list representation, use as-is
                    all_tags.add(c_str)

            classes = sorted(all_tags)

        return classes

    def _is_empty_prediction(self, pred: Any) -> bool:
        """Check if a prediction is empty/None/NaN.

        Args:
            pred: Prediction value to check

        Returns:
            True if prediction is empty/None/NaN, False otherwise
        """
        if pred is None:
            return True
        if isinstance(pred, float) and np.isnan(pred):
            return True
        if not isinstance(pred, np.ndarray) and isinstance(pred, (str, int, float)):
            try:
                if pd.isna(pred):
                    return True
            except (ValueError, TypeError):
                pass
        if isinstance(pred, (list, tuple)) and len(pred) == 0:
            return True
        if isinstance(pred, np.ndarray):
            if pred.size == 0 or np.all(np.isnan(pred)):
                return True
        return False

    def _normalize_prediction_to_list(self, pred: Any) -> list[str]:
        """Convert prediction to list of strings.

        Args:
            pred: Prediction value (can be list, tuple, array, or single value)

        Returns:
            List of strings
        """
        if isinstance(pred, (list, tuple)):
            return [str(p) for p in pred]
        elif isinstance(pred, np.ndarray):
            return [str(p) for p in pred.flatten()]
        else:
            return [str(pred)]

    def _get_fallback_prediction(
        self,
        i: int,
        classes: list[str] | None,
        probabilities: Any,
    ) -> list[str]:
        """Get fallback prediction when original is invalid.

        Args:
            i: Sample index
            classes: List of valid class names
            probabilities: Probability matrix (can be array or list of arrays for multi-label)

        Returns:
            Fallback prediction (always a list with at least one label)
        """
        if probabilities is not None and classes is not None and len(classes) > 0:
            try:
                # Handle multi-label probabilities (list of arrays from MultiOutputClassifier)
                if isinstance(probabilities, list) and len(probabilities) > 0:
                    if i < len(probabilities[0]):
                        # Get probability of positive class for each label
                        label_probs = np.array(
                            [
                                proba[i, 1] if proba.shape[1] > 1 else proba[i, 0]
                                for proba in probabilities
                            ]
                        )
                        top_idx = int(np.argmax(label_probs))
                        if 0 <= top_idx < len(classes):
                            return [str(classes[top_idx])]
                elif isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
                    if i < len(probabilities) and probabilities.shape[1] == len(classes):
                        top_idx = int(np.argmax(probabilities[i]))
                        if 0 <= top_idx < len(classes):
                            return [str(classes[top_idx])]
            except (IndexError, ValueError, AttributeError) as e:
                if self.verbose:
                    print(
                        f"Warning: Error accessing probabilities for sample {i}: {e}. "
                        f"Using first class as fallback.",
                        flush=True,
                    )

        if classes is not None and len(classes) > 0:
            return [str(classes[0])]

        # Last resort: empty list (shouldn't happen if model is properly trained)
        if self.verbose:
            print(
                f"Warning: No fallback available for sample {i}. Returning empty list.",
                flush=True,
            )
        return []

    def ensure_valid_predictions(
        self,
        predictions: Sequence[Any],
        estimator: Any,
        X: Sequence[str],
    ) -> Sequence[Any]:
        """Ensure predictions are never None, NaN, or empty. Always return valid class labels.

        Validates that all predictions are:
        1. Not None, NaN, or empty
        2. Valid class names from the dataset
        3. Always returns at least one label per sample (as a list)

        If a prediction is invalid, uses probabilities to select the top-1 class as fallback.

        Args:
            predictions: Sequence of predictions (may contain None/NaN/empty/invalid)
            estimator: The model estimator (for getting classes and probabilities)
            X: Input texts (for getting probabilities if needed)

        Returns:
            Sequence of valid predictions (list of lists, all labels in valid classes)
        """
        if predictions is None:
            predictions = []

        # Get classes and probabilities once
        classes = self._get_classes_from_estimator(estimator)
        probabilities = None

        if hasattr(estimator, "predict_proba") and classes is not None and len(classes) > 0:
            try:
                probabilities = estimator.predict_proba(X)
            except Exception:
                probabilities = None

        valid_predictions = []
        valid_classes_set = set(classes) if classes else None

        for i, pred in enumerate(predictions):
            # Check if prediction is empty
            if self._is_empty_prediction(pred):
                # Use fallback
                fallback = self._get_fallback_prediction(i, classes, probabilities)
                valid_predictions.append(fallback)
                if self.verbose and len(fallback) == 0:
                    print(
                        f"Warning: Empty prediction at index {i}, no fallback available",
                        flush=True,
                    )
            else:
                # Prediction exists, normalize to list and validate
                pred_list = self._normalize_prediction_to_list(pred)

                if valid_classes_set is not None:
                    # Filter out invalid classes
                    valid_labels = [label for label in pred_list if label in valid_classes_set]

                    if len(valid_labels) == 0:
                        # All labels invalid - use fallback
                        if self.verbose:
                            num_classes = len(valid_classes_set)
                            sample_classes = sorted(valid_classes_set)[:3]
                            sample_str = ", ".join(str(c) for c in sample_classes)
                            suffix = f" (and {num_classes - 3} more)" if num_classes > 3 else ""
                            print(
                                f"Warning: All labels invalid at index {i} "
                                f"(predicted: {pred_list}), not in {num_classes} "
                                f"dataset classes [{sample_str}{suffix}]. "
                                f"Using fallback.",
                                flush=True,
                            )
                        fallback = self._get_fallback_prediction(i, classes, probabilities)
                        valid_predictions.append(fallback)
                    else:
                        # At least one valid label, keep valid ones
                        valid_predictions.append(valid_labels)
                else:
                    # No classes available for validation, keep prediction as-is
                    if self.verbose:
                        print(
                            f"Warning: No classes available for validation at index {i}. "
                            f"Keeping prediction {pred_list} as-is.",
                            flush=True,
                        )
                    valid_predictions.append(pred_list)

        # Return as list (multi-label format)
        return valid_predictions

    def normalize_label_for_saving(self, label: Any) -> str:
        """Normalize a multi-label to a string for saving. Never returns None or NaN.

        Args:
            label: Can be list, tuple, numpy array, string, None, or NaN

        Returns:
            String representation (empty string for None/NaN/empty)
        """
        # Handle None explicitly
        if label is None:
            return ""

        # Handle NaN (float) - check before pd.isna to avoid array issues
        if isinstance(label, float) and np.isnan(label):
            return ""

        # Handle NaN (pandas) - only for scalar values (not arrays)
        if not isinstance(label, np.ndarray) and not isinstance(label, (list, tuple)):
            try:
                if pd.isna(label):
                    return ""
            except (ValueError, TypeError):
                # pd.isna can fail on some types, skip it
                pass

        # Handle empty strings
        if isinstance(label, str) and label == "":
            return ""

        # Handle lists/tuples
        if isinstance(label, (list, tuple)):
            if len(label) == 0:
                return ""
            # Strip, filter empty/None/NaN, sort for consistency, and join
            cleaned_labels = sorted(
                {
                    str(item).strip()
                    for item in label
                    if item is not None
                    and not (isinstance(item, float) and np.isnan(item))
                    and str(item).strip()
                }
            )
            # Join with comma, ensuring no trailing comma
            result = ",".join(cleaned_labels)
            return result.rstrip(",")

        # Handle numpy arrays
        if isinstance(label, np.ndarray):
            if label.size == 0:
                return ""
            # If it's a binary array (multilabel format), we need classes to
            # convert it. But at this point we don't have access to classes, so
            # this shouldn't happen. If we receive a binary array here, it's
            # likely a bug - convert indices to strings as fallback
            if label.ndim == 1:
                # 1D array: treat as indices or binary vector
                if label.dtype in (np.int_, np.int64, np.int32):
                    # Looks like indices - convert to strings (this shouldn't happen in normal flow)
                    return ",".join(str(int(idx)) for idx in label if idx >= 0)
                else:
                    # Binary vector - can't convert without classes, return empty
                    return ""
            else:
                # Multi-dimensional array - shouldn't happen, return empty
                return ""

        # Handle string representations of arrays (shouldn't happen, but handle gracefully)
        if isinstance(label, str) and label.startswith("[") and " " in label:
            return ""

        # Default: convert to string
        result = str(label)
        # Ensure we don't save "None" or "nan" as strings
        if result.lower() in ("none", "nan", "<na>"):
            return ""
        return result

    def persist_predictions(
        self,
        estimator: Any,
        X: Sequence[str],
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        output_dir: Path,
        prefix: str,
        model_name: str,
    ) -> None:
        """Persist multi-label predictions and probabilities for a dataset.

        Args:
            estimator: Fitted scikit-learn estimator
            X: Input text data to generate predictions for
            y_true: Ground truth labels for the input data
            y_pred: Predicted labels for the input data (cached)
            output_dir: Directory path to save the predictions
            prefix: Prefix for output files (e.g., 'train' or 'test')
            model_name: Name of the model being used
        """
        # Multi-label: convert list of lists to list of strings
        y_true_str = [self.normalize_label_for_saving(label) for label in y_true]
        y_pred_str = [self.normalize_label_for_saving(label) for label in y_pred]

        # Include text column for analysis
        df = pd.DataFrame(
            {
                "text": [str(x) for x in X],
                "y_true": y_true_str,
                "y_pred": y_pred_str,
            },
            dtype=str,
        )

        # Ensure no NaN values in the DataFrame (replace with empty strings)
        df = df.fillna("").astype(str)
        # Replace any "nan" strings (from numpy/pandas) with empty strings
        df = df.replace("nan", "", regex=False)
        df = df.replace("None", "", regex=False)

        # Save with explicit handling to prevent pandas from converting empty
        # strings to NaN. Note: Since we ensure all predictions have at least
        # one label, empty strings shouldn't occur
        output_path = str(output_dir / f"{prefix}_predictions.csv")
        df.to_csv(
            output_path,
            index=False,
            na_rep="",
        )

        # Save probabilities if available
        if hasattr(estimator, "predict_proba"):
            try:
                y_prob = estimator.predict_proba(X)
                np.save(output_dir / f"{prefix}_prob.npy", y_prob)
            except Exception as e:
                if self.verbose:
                    print(
                        f"[{model_name}] Warning: Could not save {prefix} probabilities: {e}",
                        flush=True,
                    )
