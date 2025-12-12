"""Base evaluation runner with shared functionality for single-label and multi-label."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from intent_classifier.evaluation.utils import load_all_prediction_files, setup_logging
from intent_classifier.utils.file_ops import ensure_dir


class BaseEvaluationRunner(ABC):
    """Base class for evaluation runners with shared functionality.

    This class provides common functionality for both single-label and multi-label
    evaluation workflows, including data loading and result aggregation.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the evaluation runner.

        Args:
            verbose: Whether to print progress and warning messages
        """
        self.verbose = verbose
        self.logger = setup_logging(verbose)

    @abstractmethod
    def parse_labels_from_csv(self, labels: pd.Series) -> List[Any]:
        """Parse labels from CSV format (comma-separated strings) to native format.

        Args:
            labels: Series of labels from CSV (may be comma-separated strings)

        Returns:
            List of labels in native format (list of strings for multi-label, strings for single-label)
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        split_name: str,
        output_dir: Path,
    ) -> Optional[Dict[str, float]]:
        """Compute metrics for a split.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            split_name: Name of the split (e.g., 'train', 'test')
            output_dir: Directory to save metrics

        Returns:
            Dictionary of metric names to values, or None if computation failed
        """
        pass

    @abstractmethod
    def get_overfitting_metrics(self) -> List[str]:
        """Get list of metrics to use for overfitting analysis.

        Returns:
            List of metric names (without 'train_' or 'test_' prefix)
        """
        pass

    @abstractmethod
    def get_summary_metrics(self) -> List[str]:
        """Get list of metrics to include in summary tables.

        Returns:
            List of metric names (without 'test_' prefix)
        """
        pass

    @abstractmethod
    def should_generate_visualizations(self) -> bool:
        """Whether to generate visualizations for this label type.

        Returns:
            True if visualizations should be generated, False otherwise
        """
        pass

    def run_evaluations(
        self,
        model_names: List[str] | Dict[str, Any] | None,
        *,
        artefacts_root: str | Path = "artefacts",
        output_dir: str | Path = "results",
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics from persisted predictions and render rich reports.

        Args:
            model_names: List of model names to evaluate, dict of models (keys will be used),
                        or None/empty list to evaluate all models with predictions
            artefacts_root: Root directory containing prediction files
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary mapping model names to their metrics
        """
        results: Dict[str, Dict[str, Any]] = {}
        artefacts_root, output_dir = Path(artefacts_root), ensure_dir(output_dir)

        predictions_dict = load_all_prediction_files(artefacts_root)
        if not predictions_dict:
            self.logger.error("No prediction files found.")
            return results

        first_model = next(iter(predictions_dict))
        if "test" not in predictions_dict[first_model]:
            self.logger.error("No test predictions found for model %s", first_model)
            return results

        # Extract classes from test set
        test_df = predictions_dict[first_model]["test"]
        classes = self._extract_classes(test_df)

        # If model_names is None or empty, use all models that have predictions
        if not model_names:
            model_names = list(predictions_dict.keys())
        # If model_names is a dict, convert to list of keys
        elif isinstance(model_names, dict):
            model_names = list(model_names.keys())

        # Per-model processing
        for name, model_predictions in predictions_dict.items():
            if name not in model_names:
                continue

            model_out_dir = ensure_dir(output_dir / name)
            self.logger.info("Processing model %s", name)

            model_results: Dict[str, float] = {}
            for split_name, df in model_predictions.items():
                split_out_dir = ensure_dir(model_out_dir / split_name)
                y_true_raw, y_pred_raw = df["y_true"].values, df["y_pred"].values

                # Parse labels from CSV format
                y_true = self.parse_labels_from_csv(pd.Series(y_true_raw))
                y_pred = self.parse_labels_from_csv(pd.Series(y_pred_raw))

                # Compute metrics
                metrics = self.compute_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    split_name=split_name,
                    output_dir=split_out_dir,
                )
                if metrics:
                    model_results.update(metrics)
                else:
                    continue

                # Generate visualizations (if supported)
                if self.should_generate_visualizations() and len(df) > 0:
                    self._generate_visualizations(name, split_name, df, split_out_dir)

                # Generate error analysis (if supported and applicable)
                if (
                    split_name == "test"
                    and "text" in df.columns
                    and self.should_generate_visualizations()
                ):
                    self._generate_error_analysis(df, split_out_dir)

            # Overfitting indicators
            if {"train", "test"}.issubset(model_predictions):
                metrics = self.get_overfitting_metrics()
                for metric in metrics:
                    train_val = model_results.get(f"train_{metric}")
                    test_val = model_results.get(f"test_{metric}")
                    if train_val is not None and test_val is not None:
                        model_results[f"{metric}_diff"] = train_val - test_val

            results[name] = model_results

        # Summary & cross-model visualizations
        self._generate_summary(results, predictions_dict, output_dir, model_names)

        return results

    def _extract_classes(self, test_df: pd.DataFrame) -> List[str]:
        """Extract unique classes from test DataFrame.

        Args:
            test_df: Test DataFrame with y_true column

        Returns:
            List of unique class names
        """
        # Default implementation - can be overridden
        return sorted(test_df["y_true"].unique().tolist())

    def _generate_visualizations(
        self,
        model_name: str,
        split_name: str,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Generate visualizations for a split.

        Args:
            model_name: Name of the model
            split_name: Name of the split
            df: DataFrame with predictions
            output_dir: Directory to save visualizations
        """
        from intent_classifier.evaluation.visualization import (
            generate_detailed_error_report,
            plot_confusion_matrix,
            plot_label_distribution,
            plot_precision_recall_curves,
            visualize_error_distribution,
        )

        # Create a cleaned DataFrame for visualizations
        df_clean = df.copy()
        mask = pd.notna(df_clean["y_true"]) & pd.notna(df_clean["y_pred"])
        df_clean = df_clean[mask].copy()
        df_clean["y_true"] = df_clean["y_true"].astype(str)
        df_clean["y_pred"] = df_clean["y_pred"].astype(str)

        if len(df_clean) > 0:
            single_model_predictions = {model_name: {split_name: df_clean}}
            try:
                plot_label_distribution(single_model_predictions, output_dir)
                plot_confusion_matrix(single_model_predictions, output_dir)
                plot_precision_recall_curves(single_model_predictions, output_dir)
                visualize_error_distribution(single_model_predictions, output_dir)
                generate_detailed_error_report(
                    single_model_predictions, output_dir, only_split=split_name
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to generate visualizations for {model_name}/{split_name}: {e}"
                )

    def _generate_error_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate error analysis for test split.

        Args:
            df: DataFrame with test predictions
            output_dir: Directory to save error analysis
        """
        # Import here to avoid circular dependencies
        from intent_classifier.evaluation.evaluation import (
            analyze_rag_documents,
            analyze_top_errors,
        )
        from intent_classifier.evaluation.visualization import (
            plot_top_error_types,
            plot_top_misclassifications,
        )

        plot_top_misclassifications(df, output_dir / "top_misclassifications.csv")
        analyze_top_errors(df, output_dir / "test_top_20_errors.csv")
        plot_top_error_types(df, output_dir / "top_10_error_types.png", n=10)
        if "retrieved_docs" in df.columns:
            analyze_rag_documents(df, output_dir / "test_rag_analysis.csv")

    def _generate_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        predictions_dict: Dict[str, Dict[str, pd.DataFrame]],
        output_dir: Path,
        model_names: List[str],
    ) -> None:
        """Generate summary tables and cross-model visualizations.

        Args:
            results: Dictionary of model results
            predictions_dict: Dictionary of predictions
            output_dir: Directory to save summaries
            model_names: List of model names
        """
        import pandas as pd

        # Import here to avoid circular dependencies
        from intent_classifier.evaluation.metrics import analyze_text_features
        from intent_classifier.evaluation.utils import (
            analyse_error_patterns,
            consistently_misclassified,
        )
        from intent_classifier.evaluation.visualization import plot_model_comparisons

        summary_df = pd.DataFrame(results).T
        metrics_to_check = self.get_summary_metrics()

        for metric in metrics_to_check:
            if metric in summary_df.columns:
                # For hamming_loss, lower is better, so invert the comparison
                if "hamming_loss" in metric:
                    summary_df[f"{metric}_best"] = summary_df[metric] == summary_df[metric].min()
                else:
                    summary_df[f"{metric}_best"] = summary_df[metric] == summary_df[metric].max()
        summary_df.to_csv(output_dir / "summary_metrics.csv")

        if len(model_names) > 1:
            plot_model_comparisons(predictions_dict, output_dir)

        # Global error analysis
        error_patterns = analyse_error_patterns(predictions_dict)
        error_patterns.to_csv(output_dir / "common_error_patterns.csv", index=False)
        misclass_examples = consistently_misclassified(
            predictions_dict, min_models=len(predictions_dict)
        )
        if not misclass_examples.empty:
            misclass_examples.to_csv(output_dir / "consistently_misclassified.csv", index=False)

        first_model = next(iter(predictions_dict))
        if "text" in predictions_dict[first_model]["test"].columns:
            analyze_text_features(predictions_dict).to_csv(
                output_dir / "text_features_analysis.csv", index=False
            )
