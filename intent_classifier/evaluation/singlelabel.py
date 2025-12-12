"""Single-label evaluation runner implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from intent_classifier.evaluation.base import BaseEvaluationRunner
from intent_classifier.evaluation.metrics_singlelabel import (
    compute_singlelabel_metrics,
    get_singlelabel_overfitting_metrics,
    get_singlelabel_summary_metrics,
)


class SingleLabelEvaluationRunner(BaseEvaluationRunner):
    """Evaluation runner for single-label classification tasks.

    This class handles evaluation workflows specific to single-label classification,
    where each sample has exactly one label.
    """

    def parse_labels_from_csv(self, labels: pd.Series) -> List[str]:
        """Parse single-label from CSV format.

        Args:
            labels: Series of labels from CSV (single strings)

        Returns:
            List of single label strings
        """
        return [str(label) if pd.notna(label) and label != "" else "" for label in labels]

    def compute_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        split_name: str,
        output_dir: Path,
    ) -> Optional[Dict[str, float]]:
        """Compute single-label metrics for a split.

        Args:
            y_true: Ground truth labels (list of strings)
            y_pred: Predicted labels (list of strings)
            split_name: Name of the split (e.g., 'train', 'test')
            output_dir: Directory to save metrics

        Returns:
            Dictionary of metric names to values, or None if computation failed
        """
        return compute_singlelabel_metrics(
            y_true=y_true,
            y_pred=y_pred,
            split_name=split_name,
            output_dir=output_dir,
            logger=self.logger,
        )

    def get_overfitting_metrics(self) -> List[str]:
        """Get list of metrics to use for overfitting analysis in single-label tasks.

        Returns:
            List of metric names (without 'train_' or 'test_' prefix)
        """
        return get_singlelabel_overfitting_metrics()

    def get_summary_metrics(self) -> List[str]:
        """Get list of metrics to include in summary tables for single-label tasks.

        Returns:
            List of metric names (without 'test_' prefix)
        """
        return get_singlelabel_summary_metrics()

    def should_generate_visualizations(self) -> bool:
        """Single-label supports visualizations.

        Returns:
            True
        """
        return True

    def _extract_classes(self, test_df: pd.DataFrame) -> List[str]:
        """Extract unique classes from test DataFrame (single-label).

        Args:
            test_df: Test DataFrame with y_true column

        Returns:
            List of unique class names
        """
        return sorted(test_df["y_true"].unique().tolist())
