"""Main evaluation module that orchestrates the evaluation workflow.

This module provides the main entry point for running evaluations on text
classification models. It coordinates the various evaluation components and
produces comprehensive, publication‑quality reports and visualisations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

# -----------------------------------------------------------------------------
# Metric‑table visualisation helpers
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: F401 – kept for future extensions
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize  # noqa: F401 – kept for future use

from intent_classifier.utils.file_ops import ensure_dir
from intent_classifier.utils.method_logger import get_logger

from .metrics import analyze_text_features, compute_metrics  # noqa: F401 – API surface
from .utils import (
    load_all_prediction_files,
    setup_logging,
)
from .visualization import (
    plot_roc_curves,  # noqa: F401 – exported elsewhere
)

__all__ = [
    "analyze_top_errors",
    "analyze_rag_documents",
    "run_evaluations",
    "save_metric_table",
    "display_detailed_results",
]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_HIGHLIGHT_COLOR = "#d7f0fa"  # unified styling colour for highlighted cells


def _best_per_axis_mask(
    df: pd.DataFrame,
    axis: Literal["row", "col"],
    mode: Literal["max", "min"] = "max",
    absolute: bool = False,
) -> pd.DataFrame:
    """Return a *boolean* ``DataFrame`` of shape ``df`` where the *best* values
    (``max`` or ``min``) along the chosen axis are *True*.

    Parameters
    ----------
    df : pd.DataFrame
        The metric table to analyse.
    axis : {"row", "col"}
        Whether to look for the best value in each *row* or each *column*.
    mode : {"max", "min"}, default="max"
        Define if the *largest* (``max``) or *smallest* (``min``) value is considered best.
    absolute : bool, default=False
        Compare on the absolute values of *df*. Useful when e.g. the *smallest
        absolute error* is desired regardless of sign.
    """
    data = df.abs() if absolute else df
    if axis == "col":
        best = data.eq(data.max() if mode == "max" else data.min())
    else:  # axis == "row"
        extrema = data.max(axis=1) if mode == "max" else data.min(axis=1)
        best = data.eq(extrema, axis=0)
    return best


# -----------------------------------------------------------------------------
# Public API – table renderer
# -----------------------------------------------------------------------------


def save_metric_table(
    df: pd.DataFrame,
    path: Path | str,
    title: str,
    *,
    number_format: str = "{:.3f}",
    highlight: Literal["row", "col"] | None = "col",
    highlight_mode: Literal["max", "min"] = "max",
    highlight_abs: bool = False,
    highlight_color: str = _HIGHLIGHT_COLOR,
    **kwargs: Any,
) -> None:
    """Render *df* as a high‑resolution PNG using **dataframe_image**.

    By default, the *best* value per **column** is shown with a coloured background
    and **bold** font. Change ``highlight`` to ``"row"`` to switch to a
    row‑wise comparison. Both the colour and the definition of *best*
    (``max``/``min``) are configurable.

    Note: If dataframe_image is not available, this function will log a warning
    and skip image generation.
    """
    if df.empty:
        raise ValueError("Provided DataFrame is empty – nothing to plot.")

    # Lazy import of dataframe_image to avoid ModuleNotFoundError if missing
    try:
        import dataframe_image as dfi
    except ImportError:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "dataframe_image not available - skipping metric table image generation. "
            "Install with: uv pip install dataframe_image"
        )
        return

    # 1. Basic formatting ------------------------------------------------------
    styler = df.style.format(number_format)

    # 2. Conditional styling for best values ----------------------------------
    if highlight is not None:
        mask = _best_per_axis_mask(df, highlight, highlight_mode, highlight_abs)

        def _style_best(data: pd.DataFrame) -> pd.DataFrame:
            """Return a style DataFrame for *Styler.apply*."""
            styled_arr = np.where(
                mask, f"font-weight: bold; background-color: {highlight_color}", ""
            )
            return pd.DataFrame(styled_arr, index=data.index, columns=data.columns)

        styler = styler.apply(_style_best, axis=None)

    # 3. Final touches ---------------------------------------------------------
    styler = styler.set_caption(title)

    # 4. Export ----------------------------------------------------------------
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Using the *matplotlib* backend (requires *kaleido*) to avoid the need for
    # a running Chrome instance on headless servers.
    dfi.export(styler, str(path), table_conversion="matplotlib")


# -----------------------------------------------------------------------------
# Error‑analysis helpers – unchanged public surface
# -----------------------------------------------------------------------------


def analyze_top_errors(
    predictions_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Analyze and save the top *N* (y_true, y_pred) error pairs."""
    error_df = predictions_df[predictions_df["y_true"] != predictions_df["y_pred"]].copy()
    error_counts = (
        error_df.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    error_counts.to_csv(output_path, index=False)


def analyze_rag_documents(
    predictions_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 5,
) -> None:
    """Light‑weight RAG relevance analysis saved to CSV."""
    if {"retrieved_docs", "query"}.difference(predictions_df.columns):
        logger = setup_logging(False)
        logger.warning("Missing RAG document or query information – skipping.")
        return

    analysis_results: list[dict[str, Any]] = []
    for idx, row in predictions_df.iterrows():
        query, docs = row["query"], row["retrieved_docs"]
        for pos, doc in enumerate(docs[:top_n], start=1):
            analysis_results.append(
                {
                    "query_id": idx,
                    "query": query,
                    "true_label": row["y_true"],
                    "pred_label": row["y_pred"],
                    "document_position": pos,
                    "document_length": len(doc.split()),
                    "document_sentences": len(doc.split(".")),
                    "contains_true_label": row["y_true"].lower() in doc.lower(),
                    "contains_pred_label": row["y_pred"].lower() in doc.lower(),
                    "document_text": doc,
                }
            )
    pd.DataFrame(analysis_results).to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Main evaluation orchestration – mostly unchanged
# -----------------------------------------------------------------------------


def run_evaluations(
    model_names: list[str] | dict[str, Any] | None,
    *,
    artefacts_root: str | Path = "artefacts",
    eval_dir: str | Path,
    compare_dir: str | Path,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute metrics from persisted predictions and render rich reports.

    This function automatically detects whether the task is single-label or multi-label
    and uses the appropriate evaluation implementation.

    Args:
        model_names: List of model names to evaluate, dict of models (keys will be used),
                    or None/empty list to evaluate all models with predictions
        artefacts_root: Root directory containing prediction files
        eval_dir: Directory for per-model evaluation results (eval/<model_id>/)
        compare_dir: Directory for cross-model comparison summaries (compare/)
        verbose: Whether to print progress and warning messages

    Returns:
        Dictionary mapping model names to their metrics
    """
    # Disable method logging during evaluation
    get_logger().disable()

    from intent_classifier.evaluation.base import BaseEvaluationRunner
    from intent_classifier.evaluation.multilabel import MultiLabelEvaluationRunner
    from intent_classifier.evaluation.singlelabel import SingleLabelEvaluationRunner

    artefacts_root = Path(artefacts_root)
    eval_dir = ensure_dir(eval_dir)
    compare_dir = ensure_dir(compare_dir)

    predictions_dict = load_all_prediction_files(artefacts_root)
    if not predictions_dict:
        logger = setup_logging(verbose)
        logger.error("No prediction files found.")
        return {}

    first_model = next(iter(predictions_dict))
    if "test" not in predictions_dict[first_model]:
        logger = setup_logging(verbose)
        logger.error("No test predictions found for model %s", first_model)
        return {}

    # If model_names is a dict, convert to list of keys (model names)
    # If None or empty, use all models that have predictions
    if isinstance(model_names, dict):
        model_names = list(model_names.keys())
    elif model_names is None or len(model_names) == 0:
        model_names = list(predictions_dict.keys())

    # Detect multi-label format from test predictions
    test_df = predictions_dict[first_model]["test"]
    y_true_test = test_df["y_true"].values

    # Check if labels contain commas (multi-label format in CSV)
    is_multi_format = False
    if len(y_true_test) > 0:
        sample_label = str(y_true_test[0])
        # Multi-label format: comma-separated labels (e.g., "tag1,tag2")
        # Single-label format: single label (e.g., "tag1")
        # Check if there are commas and it's not just an empty string
        if "," in sample_label and sample_label.strip():
            is_multi_format = True
        # Also check if any label has multiple tags
        for label in y_true_test[: min(10, len(y_true_test))]:
            label_str = str(label)
            if pd.notna(label) and label_str and "," in label_str:
                is_multi_format = True
                break

    # Select appropriate runner based on label type
    runner: BaseEvaluationRunner
    if is_multi_format:
        runner = MultiLabelEvaluationRunner(verbose=verbose)
    else:
        runner = SingleLabelEvaluationRunner(verbose=verbose)

    # Delegate to the appropriate runner
    return runner.run_evaluations(
        model_names=model_names,
        artefacts_root=artefacts_root,
        eval_dir=eval_dir,
        compare_dir=compare_dir,
    )


# -----------------------------------------------------------------------------
# User‑facing display helper – unchanged public surface but uses new renderer
# -----------------------------------------------------------------------------


def display_detailed_results(
    results: dict[str, dict[str, Any]],
    model_order: list[str] | None = None,
    output_dir: str | Path = "results",
) -> None:
    """Convenience helper to pretty‑print and persist the key result tables."""
    output_dir = ensure_dir(output_dir)

    # Test‑set metrics -----------------------------------------------------------
    test_metrics = pd.DataFrame(
        {m: {k: v for k, v in d.items() if k.startswith("test_")} for m, d in results.items()}
    ).T
    if model_order:
        test_metrics = test_metrics.reindex([m for m in model_order if m in test_metrics.index])

    print("\n=== Summary of Test Metrics ===")
    print(test_metrics)
    test_metrics.to_csv(output_dir / "test_metrics_summary.csv")
    save_metric_table(test_metrics, output_dir / "test_metrics_summary.png", "Test Metrics")

    # Train‑set metrics ----------------------------------------------------------
    train_cols: Sequence[str] = [
        c for c in next(iter(results.values())).keys() if c.startswith("train_")
    ]
    if train_cols:
        train_metrics = pd.DataFrame(
            {m: {k: v for k, v in d.items() if k.startswith("train_")} for m, d in results.items()}
        ).T
        if model_order:
            train_metrics = train_metrics.reindex(
                [m for m in model_order if m in train_metrics.index]
            )

        print("\n=== Summary of Train Metrics ===")
        print(train_metrics)
        train_metrics.to_csv(output_dir / "train_metrics_summary.csv")
        save_metric_table(train_metrics, output_dir / "train_metrics_summary.png", "Train Metrics")

        # Overfitting indicators --------------------------------------------------
        diff_cols: Sequence[str] = [
            c for c in next(iter(results.values())).keys() if c.endswith("_diff")
        ]
        if diff_cols:
            diff_metrics = pd.DataFrame(
                {m: {k: v for k, v in d.items() if k.endswith("_diff")} for m, d in results.items()}
            ).T
            if model_order:
                diff_metrics = diff_metrics.reindex(
                    [m for m in model_order if m in diff_metrics.index]
                )

            print("\n=== Train/Test Differences (Overfitting Analysis) ===")
            print(diff_metrics)
            diff_metrics.to_csv(output_dir / "overfitting_analysis.csv")
            save_metric_table(
                diff_metrics,
                output_dir / "overfitting_analysis.png",
                "Overfitting Analysis (Train/Test Differences)",
                highlight=None,
            )

            guide = (
                "Interpretation guide:\n"
                "- Positive values → model may be overfitting (better on train).\n"
                "- ≈0 → good generalisation.\n"
                "- Negative values → underfitting or potential data leakage.\n"
            )
            print("\n" + guide)
            (output_dir / "interpretation_guide.txt").write_text(guide)
