"""Multi-label evaluation runner implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from intent_classifier.evaluation.base import BaseEvaluationRunner
from intent_classifier.evaluation.metrics_multilabel import (
    compute_multilabel_metrics,
    get_multilabel_overfitting_metrics,
    get_multilabel_summary_metrics,
)


class MultiLabelEvaluationRunner(BaseEvaluationRunner):
    """Evaluation runner for multi-label classification tasks.

    This class handles evaluation workflows specific to multi-label classification,
    where each sample can have multiple labels.
    """

    def parse_labels_from_csv(self, labels: pd.Series) -> List[List[str]]:
        """Parse multi-label from CSV format (comma-separated strings).

        Args:
            labels: Series of labels from CSV (comma-separated strings)

        Returns:
            List of lists of label strings
        """
        result = []
        for label in labels:
            if pd.isna(label) or label == "":
                result.append([])
            else:
                tags = [tag.strip() for tag in str(label).split(",") if tag.strip()]
                result.append(tags if tags else [])
        return result

    def compute_metrics(
        self,
        y_true: List[List[str]],
        y_pred: List[List[str]],
        split_name: str,
        output_dir: Path,
    ) -> Optional[Dict[str, float]]:
        """Compute multi-label metrics for a split.

        Args:
            y_true: Ground truth labels (list of lists of strings)
            y_pred: Predicted labels (list of lists of strings)
            split_name: Name of the split (e.g., 'train', 'test')
            output_dir: Directory to save metrics

        Returns:
            Dictionary of metric names to values, or None if computation failed
        """
        return compute_multilabel_metrics(
            y_true=y_true,
            y_pred=y_pred,
            split_name=split_name,
            output_dir=output_dir,
            logger=self.logger,
        )

    def get_overfitting_metrics(self) -> List[str]:
        """Get list of metrics to use for overfitting analysis in multi-label tasks.

        Returns:
            List of metric names (without 'train_' or 'test_' prefix)
        """
        return get_multilabel_overfitting_metrics()

    def get_summary_metrics(self) -> List[str]:
        """Get list of metrics to include in summary tables for multi-label tasks.

        Returns:
            List of metric names (without 'test_' prefix)
        """
        return get_multilabel_summary_metrics()

    def should_generate_visualizations(self) -> bool:
        """Multi-label does not support standard visualizations.

        Returns:
            False
        """
        return False

    def _extract_classes(self, test_df: pd.DataFrame) -> List[str]:
        """Extract unique classes from test DataFrame (multi-label).

        Args:
            test_df: Test DataFrame with y_true column (comma-separated strings)

        Returns:
            List of unique class names (all tags)
        """
        all_tags = set()
        for label_str in test_df["y_true"].values:
            if pd.notna(label_str) and label_str != "":
                tags = [tag.strip() for tag in str(label_str).split(",") if tag.strip()]
                all_tags.update(tags)
        return sorted(all_tags)
