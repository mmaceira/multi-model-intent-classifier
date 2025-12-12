"""Single-label prediction runner implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intent_classifier.prediction.base import BasePredictionRunner


class SingleLabelPredictionRunner(BasePredictionRunner):
    """Prediction runner for single-label classification tasks.

    This class handles prediction workflows specific to single-label classification,
    where each sample has exactly one label.
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
            # Ensure all are strings
            classes = [str(c) for c in classes]

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
        if isinstance(pred, str) and pred == "":
            return True
        if isinstance(pred, np.ndarray):
            if pred.size == 0 or np.all(np.isnan(pred)):
                return True
        return False

    def _get_fallback_prediction(
        self,
        i: int,
        classes: list[str] | None,
        probabilities: np.ndarray | None,
    ) -> str:
        """Get fallback prediction when original is invalid.

        Args:
            i: Sample index
            classes: List of valid class names
            probabilities: Probability matrix (n_samples, n_classes)

        Returns:
            Fallback prediction (always a string)
        """
        if probabilities is not None and classes is not None and len(classes) > 0:
            if i < len(probabilities) and probabilities.shape[1] == len(classes):
                top_idx = int(np.argmax(probabilities[i]))
                if 0 <= top_idx < len(classes):
                    return str(classes[top_idx])

        if classes is not None and len(classes) > 0:
            return str(classes[0])

        # Last resort: empty string (shouldn't happen if model is properly trained)
        if self.verbose:
            print(
                f"Warning: No fallback available for sample {i}. Returning empty string.",
                flush=True,
            )
        return ""

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
        3. Always returns at least one label per sample

        If a prediction is invalid, uses probabilities to select the top-1 class as fallback.

        Args:
            predictions: Sequence of predictions (may contain None/NaN/empty/invalid)
            estimator: The model estimator (for getting classes and probabilities)
            X: Input texts (for getting probabilities if needed)

        Returns:
            Sequence of valid predictions (strings, all in valid classes)
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
                if self.verbose and fallback == "":
                    print(
                        f"Warning: Empty prediction at index {i}, no fallback available",
                        flush=True,
                    )
            else:
                # Prediction exists, validate it's a valid class
                pred_str = str(pred)

                if valid_classes_set is not None:
                    if pred_str not in valid_classes_set:
                        # Invalid class - use fallback
                        if self.verbose:
                            num_classes = len(valid_classes_set)
                            sample_classes = sorted(valid_classes_set)[:5]
                            sample_str = ", ".join(str(c) for c in sample_classes)
                            suffix = f" (and {num_classes - 5} more)" if num_classes > 5 else ""
                            print(
                                f"Warning: Invalid class '{pred_str}' at "
                                f"index {i}, not in {num_classes} dataset "
                                f"classes [{sample_str}{suffix}]. "
                                f"Using fallback.",
                                flush=True,
                            )
                        fallback = self._get_fallback_prediction(i, classes, probabilities)
                        valid_predictions.append(fallback)
                    else:
                        # Valid class, keep it
                        valid_predictions.append(pred_str)
                else:
                    # No classes available for validation, keep prediction as-is
                    if self.verbose:
                        print(
                            f"Warning: No classes available for validation at index {i}. "
                            f"Keeping prediction '{pred_str}' as-is.",
                            flush=True,
                        )
                    valid_predictions.append(pred_str)

        # Return in same format as input
        if isinstance(predictions, np.ndarray):
            return np.array(valid_predictions)  # type: ignore[no-any-return]
        return valid_predictions

    def normalize_label_for_saving(self, label: Any) -> str:
        """Normalize a single-label to a string for saving. Never returns None or NaN.

        Args:
            label: Can be string, None, or NaN

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

        # Handle lists/tuples (shouldn't happen in single-label, but handle gracefully)
        if isinstance(label, (list, tuple)):
            if len(label) == 0:
                return ""
            # Take first element if list (fallback)
            label = label[0]

        # Handle numpy arrays (shouldn't happen in single-label, but handle gracefully)
        if isinstance(label, np.ndarray):
            if label.size == 0:
                return ""
            # Take first element if array (fallback)
            label = label.flat[0]

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
        """Persist single-label predictions and probabilities for a dataset.

        Args:
            estimator: Fitted scikit-learn estimator
            X: Input text data to generate predictions for
            y_true: Ground truth labels for the input data
            y_pred: Predicted labels for the input data (cached)
            output_dir: Directory path to save the predictions
            prefix: Prefix for output files (e.g., 'train' or 'test')
            model_name: Name of the model being used
        """
        # Single-label: normalize but keep as single values
        y_true_normalized = [self.normalize_label_for_saving(label) for label in y_true]
        y_pred_normalized = [self.normalize_label_for_saving(label) for label in y_pred]
        df = pd.DataFrame({"y_true": y_true_normalized, "y_pred": y_pred_normalized}, dtype=str)

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
