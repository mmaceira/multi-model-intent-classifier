"""Base evaluation runner with shared functionality for single-label and multi-label."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from intent_classifier.evaluation.utils import load_all_prediction_files, setup_logging
from intent_classifier.utils.file_ops import ensure_dir, sanitize_model_name


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
    def parse_labels_from_csv(self, labels: pd.Series) -> list[Any]:
        """Parse labels from CSV format (comma-separated strings) to native format.

        Args:
            labels: Series of labels from CSV (may be comma-separated strings)

        Returns:
            List of labels in native format (list of strings for multi-label,
            strings for single-label)
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def compute_metrics(
        self,
        y_true: list[Any],
        y_pred: list[Any],
        split_name: str,
        output_dir: Path,
    ) -> dict[str, float] | None:
        """Compute metrics for a split.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            split_name: Name of the split (e.g., 'train', 'test')
            output_dir: Directory to save metrics

        Returns:
            Dictionary of metric names to values, or None if computation failed
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def get_overfitting_metrics(self) -> list[str]:
        """Get list of metrics to use for overfitting analysis.

        Returns:
            List of metric names (without 'train_' or 'test_' prefix)
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def get_summary_metrics(self) -> list[str]:
        """Get list of metrics to include in summary tables.

        Returns:
            List of metric names (without 'test_' prefix)
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def should_generate_visualizations(self) -> bool:
        """Whether to generate visualizations for this label type.

        Returns:
            True if visualizations should be generated, False otherwise
        """
        pass  # pylint: disable=unnecessary-pass

    def run_evaluations(
        self,
        model_names: list[str] | dict[str, Any] | None,
        *,
        artefacts_root: str | Path = "artefacts",
        output_dir: str | Path = "results",
    ) -> dict[str, dict[str, Any]]:
        """Run evaluations and save results.

        Args:
            model_names: List of model names to evaluate, dict of models (keys will be used),
                        or None/empty list to evaluate all models with predictions
            artefacts_root: Root directory containing prediction files
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary mapping model names to their metrics
        """
        """Compute metrics from persisted predictions and render rich reports.

        Args:
            model_names: List of model names to evaluate, dict of models (keys will be used),
                        or None/empty list to evaluate all models with predictions
            artefacts_root: Root directory containing prediction files
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary mapping model names to their metrics
        """
        results: dict[str, dict[str, Any]] = {}
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
        self._extract_classes(test_df)

        # ------------------------------------------------------------------
        # Normalise requested model names to prediction directory keys
        # ------------------------------------------------------------------
        # predictions_dict is keyed by *sanitised* model names (see
        # intent_classifier.utils.file_ops.sanitize_model_name), whereas
        # callers may provide human‑readable display names. To ensure we
        # evaluate the correct models (including those whose names contain
        # path separators such as "Qwen/Ollama"), we build a mapping from
        # prediction‑directory key -> display name.
        # ------------------------------------------------------------------

        display_name_by_pred_key: dict[str, str] = {}

        if not model_names:
            # No filter specified: evaluate all models and keep their
            # existing keys as display names.
            for pred_key in predictions_dict.keys():
                display_name_by_pred_key[pred_key] = pred_key
            normalised_model_names: list[str] | None = None

        else:
            # If a dict was provided, use its keys (e.g. mapping of names to objects)
            if isinstance(model_names, dict):
                normalised_model_names = list(model_names.keys())
            else:
                normalised_model_names = list(model_names)

            # Build helper map from sanitised name -> original display name
            safe_to_display: dict[str, str] = {}
            for name in normalised_model_names:
                safe = sanitize_model_name(name)
                # Prefer the first occurrence if there are collisions
                if safe not in safe_to_display:
                    safe_to_display[safe] = name
                # Also allow exact (unsanitised) matches
                if name not in safe_to_display:
                    safe_to_display[name] = name

            # For each predictions directory, decide whether it should be
            # evaluated and which display name to use in the results.
            for pred_key in predictions_dict.keys():
                if pred_key in safe_to_display:
                    display_name_by_pred_key[pred_key] = safe_to_display[pred_key]

        # As a safety net, ensure that *all* models with predictions are
        # considered, even if their names do not exactly match the ones
        # passed in model_names (e.g. due to unexpected sanitisation).
        # Any such models fall back to using their prediction directory name
        # as display label so they are visible in summaries instead of being
        # silently skipped.
        for pred_key in predictions_dict.keys():
            display_name_by_pred_key.setdefault(pred_key, pred_key)

        selected_display_names = sorted(set(display_name_by_pred_key.values()))

        # Per-model processing
        for pred_key, model_predictions in predictions_dict.items():
            # Skip models that were not requested
            if pred_key not in display_name_by_pred_key:
                continue

            display_name = display_name_by_pred_key[pred_key]

            # Keep filesystem layout based on prediction key (already sanitised)
            model_out_dir = ensure_dir(output_dir / pred_key)
            self.logger.info("Processing model %s", display_name)

            model_results: dict[str, float] = {}
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
            if {"train", "test"}.issubset(set(model_predictions.keys())):
                overfitting_metrics = self.get_overfitting_metrics()
                for metric in overfitting_metrics:
                    train_val = model_results.get(f"train_{metric}")
                    test_val = model_results.get(f"test_{metric}")
                    if train_val is not None and test_val is not None:
                        model_results[f"{metric}_diff"] = train_val - test_val

            # Store results under the *display* name so that downstream
            # tables use human‑readable model identifiers.
            results[display_name] = model_results

        # Summary & cross-model visualizations
        self._generate_summary(
            results,
            predictions_dict,
            output_dir,
            selected_display_names,
            artefacts_root,
        )

        return results

    def _extract_classes(self, test_df: pd.DataFrame) -> list[str]:
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
        results: dict[str, dict[str, Any]],
        predictions_dict: dict[str, dict[str, pd.DataFrame]],
        output_dir: Path,
        model_names: list[str],
        artefacts_root: Path | None = None,
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

        # Additional analysis for multilabel tasks
        from intent_classifier.evaluation.metrics import (
            analyze_coverage_per_label,
            analyze_label_count_distribution,
            analyze_optimal_thresholds,
            analyze_per_bucket_metrics,
            compute_label_cooccurrence,
            compute_multilabel_pr_summary,
            run_multilabel_threshold_sweep,
            write_hardest_examples,
        )
        from intent_classifier.utils.config_loader import load_config_with_metadata
        from intent_classifier.utils.paths import get_repo_root

        # Check if this is multilabel by looking at first prediction
        test_df = predictions_dict[first_model]["test"]
        if "y_true" in test_df.columns:
            sample_y_true = str(test_df["y_true"].iloc[0])
            is_multilabel = "," in sample_y_true or (
                isinstance(sample_y_true, str) and len(sample_y_true.split(",")) > 1
            )

            if is_multilabel:
                # Read analysis configuration (if available) to gate heavier steps.
                try:
                    cfg = load_config_with_metadata()["config"]
                    analysis_cfg = cfg.get("analysis", {})
                    threshold_sweep_enabled = bool(analysis_cfg.get("threshold_sweep", False))
                    store_text_enabled = bool(analysis_cfg.get("store_text", False))
                except Exception:
                    threshold_sweep_enabled = False
                    store_text_enabled = False

                # Per-bucket metrics (by number of labels)
                per_bucket_df = analyze_per_bucket_metrics(predictions_dict)
                if not per_bucket_df.empty:
                    per_bucket_df.to_csv(output_dir / "per_label_count_metrics.csv", index=False)

                # Coverage analysis per label
                coverage_df = analyze_coverage_per_label(predictions_dict)
                if not coverage_df.empty:
                    coverage_df.to_csv(output_dir / "prediction_coverage.csv", index=False)

                # Label count distribution
                dist_df = analyze_label_count_distribution(predictions_dict)
                if not dist_df.empty:
                    dist_df.to_csv(output_dir / "label_count_distribution.csv", index=False)

                # Optimal thresholds analysis (if probability files exist)
                if artefacts_root is not None:
                    thresholds_df = analyze_optimal_thresholds(
                        predictions_dict, Path(artefacts_root)
                    )
                    if not thresholds_df.empty:
                        thresholds_df.to_csv(output_dir / "optimal_thresholds.csv", index=False)

                    # PR summary curves + JSON (multilabel only, probabilities required).
                    # This is intentionally defensive: if anything fails (missing files,
                    # shape mismatches, etc.), the rest of the evaluation still succeeds.
                    pr_summary_df = compute_multilabel_pr_summary(
                        predictions_dict=predictions_dict,
                        artefacts_root=Path(artefacts_root),
                        eval_root=output_dir,
                    )
                    if not pr_summary_df.empty:
                        pr_summary_df.to_csv(output_dir / "multilabel_pr_summary.csv", index=False)

                    # Optional global threshold sweep, gated by configuration.
                    if threshold_sweep_enabled:
                        sweep_df = run_multilabel_threshold_sweep(
                            predictions_dict=predictions_dict,
                            artefacts_root=Path(artefacts_root),
                            eval_root=output_dir,
                        )
                        if not sweep_df.empty:
                            sweep_df.to_csv(
                                output_dir / "multilabel_threshold_sweep.csv", index=False
                            )

                    # Label co-occurrence statistics for the dataset.
                    cooccurrence_df = compute_label_cooccurrence(predictions_dict)
                    if not cooccurrence_df.empty:
                        try:
                            paths_cfg = cfg.get("paths", {})
                            dataset_rel = paths_cfg.get("dataset_dir")
                            if isinstance(dataset_rel, str) and dataset_rel:
                                dataset_dir = get_repo_root() / dataset_rel
                                dataset_dir.mkdir(parents=True, exist_ok=True)
                                try:
                                    cooccurrence_df.to_parquet(
                                        dataset_dir / "label_cooccurrence.parquet",
                                        index=False,
                                    )
                                except Exception:
                                    cooccurrence_df.to_csv(
                                        dataset_dir / "label_cooccurrence.csv",
                                        index=False,
                                    )
                            else:
                                cooccurrence_df.to_csv(
                                    output_dir / "label_cooccurrence.csv", index=False
                                )
                        except Exception:
                            cooccurrence_df.to_csv(
                                output_dir / "label_cooccurrence.csv", index=False
                            )

                    # Hardest examples per model (error_analysis/hardest_examples.*).
                    write_hardest_examples(
                        predictions_dict=predictions_dict,
                        artefacts_root=Path(artefacts_root),
                        eval_root=output_dir,
                        store_text=store_text_enabled,
                    )

                    # PR summary curves + JSON (multilabel only, probabilities required).
                    # This is intentionally defensive: if anything fails (missing files,
                    # shape mismatches, etc.), the rest of the evaluation still succeeds.
                    pr_summary_df = compute_multilabel_pr_summary(
                        predictions_dict=predictions_dict,
                        artefacts_root=Path(artefacts_root),
                        eval_root=output_dir,
                    )
                    if not pr_summary_df.empty:
                        pr_summary_df.to_csv(output_dir / "multilabel_pr_summary.csv", index=False)
