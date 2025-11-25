"""Main evaluation module that orchestrates the evaluation workflow.

This module provides the main entry point for running evaluations on text
classification models. It coordinates the various evaluation components and
produces comprehensive, publication‑quality reports and visualisations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import dataframe_image as dfi

# -----------------------------------------------------------------------------
# Metric‑table visualisation helpers
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: F401 – kept for future extensions
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import label_binarize  # noqa: F401 – kept for future use

from .metrics import analyze_text_features, compute_metrics  # noqa: F401 – API surface
from .utils import (
    analyse_error_patterns,
    consistently_misclassified,
    ensure_dir,
    load_all_prediction_files,
    setup_logging,
)
from .visualization import (
    generate_detailed_error_report,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_model_comparisons,
    plot_precision_recall_curves,
    plot_roc_curves,  # noqa: F401 – exported elsewhere
    plot_top_error_types,
    plot_top_misclassifications,
    visualize_error_distribution,
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
    highlight: Optional[Literal["row", "col"]] = "col",
    highlight_mode: Literal["max", "min"] = "max",
    highlight_abs: bool = False,
    highlight_color: str = _HIGHLIGHT_COLOR,
    **kwargs,
) -> None:
    """Render *df* as a high‑resolution PNG using **dataframe_image**.

    By default, the *best* value per **column** is shown with a coloured background
    and **bold** font. Change ``highlight`` to ``"row"`` to switch to a
    row‑wise comparison. Both the colour and the definition of *best*
    (``max``/``min``) are configurable.
    """
    if df.empty:
        raise ValueError("Provided DataFrame is empty – nothing to plot.")

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

    analysis_results: List[Dict[str, Any]] = []
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
    model_names: List[str],
    *,
    artefacts_root: str | Path = "artefacts",
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics from persisted predictions and render rich reports."""
    logger = setup_logging(verbose)
    logger.info("Starting evaluation process → output dir: %s", output_dir)

    results: Dict[str, Dict[str, Any]] = {}
    artefacts_root, output_dir = Path(artefacts_root), ensure_dir(output_dir)

    predictions_dict = load_all_prediction_files(artefacts_root)
    if not predictions_dict:
        logger.error("No prediction files found.")
        return results

    first_model = next(iter(predictions_dict))
    if "test" not in predictions_dict[first_model]:
        logger.error("No test predictions found for model %s", first_model)
        return results

    classes = sorted(predictions_dict[first_model]["test"]["y_true"].unique())
    is_multiclass = len(classes) > 2  # noqa: F841 – may be used downstream

    # ------------------------------------------------------------------
    # Per‑model processing
    # ------------------------------------------------------------------
    for name, model_predictions in predictions_dict.items():
        if name not in model_names:
            continue

        model_out_dir = ensure_dir(output_dir / name)
        logger.info("Processing model %s", name)

        model_results: Dict[str, float] = {}
        for split_name, df in model_predictions.items():
            split_out_dir = ensure_dir(model_out_dir / split_name)
            y_true, y_pred = df["y_true"].values, df["y_pred"].values

            # Core metrics ---------------------------------------------------
            model_results.update(
                {
                    f"{split_name}_accuracy": accuracy_score(y_true, y_pred),
                    f"{split_name}_macro_f1": f1_score(y_true, y_pred, average="macro"),
                    f"{split_name}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                }
            )

            # Classification report ----------------------------------------
            pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T.to_csv(
                split_out_dir / f"{split_name}_report.csv"
            )

            # Visualisations -------------------------------------------------
            single_model_predictions = {name: {split_name: df}}
            plot_label_distribution(single_model_predictions, split_out_dir)
            plot_confusion_matrix(single_model_predictions, split_out_dir)
            plot_precision_recall_curves(single_model_predictions, split_out_dir)
            visualize_error_distribution(single_model_predictions, split_out_dir)
            generate_detailed_error_report(
                single_model_predictions, split_out_dir, only_split=split_name
            )

            if split_name == "test" and "text" in df.columns:
                plot_top_misclassifications(df, split_out_dir / "top_misclassifications.csv")
                analyze_top_errors(df, split_out_dir / f"{split_name}_top_20_errors.csv")
                plot_top_error_types(df, split_out_dir / "top_10_error_types.png", n=10)
                if "retrieved_docs" in df.columns:
                    analyze_rag_documents(df, split_out_dir / f"{split_name}_rag_analysis.csv")

        # Overfitting indicators ----------------------------------------------
        if {"train", "test"}.issubset(model_predictions):
            for metric in ["accuracy", "macro_f1", "weighted_f1"]:
                model_results[f"{metric}_diff"] = model_results.get(
                    f"train_{metric}"
                ) - model_results.get(f"test_{metric}")

        results[name] = model_results

    # ------------------------------------------------------------------
    # Summary & cross‑model visualisations
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(results).T
    for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1"]:
        if metric in summary_df.columns:
            summary_df[f"{metric}_best"] = summary_df[metric] == summary_df[metric].max()
    summary_df.to_csv(output_dir / "summary_metrics.csv")

    if len(model_names) > 1:
        plot_model_comparisons(predictions_dict, output_dir)

    # Global error analysis ----------------------------------------------------
    error_patterns = analyse_error_patterns(predictions_dict)
    error_patterns.to_csv(output_dir / "common_error_patterns.csv", index=False)
    misclass_examples = consistently_misclassified(
        predictions_dict, min_models=len(predictions_dict)
    )
    if not misclass_examples.empty:
        misclass_examples.to_csv(output_dir / "consistently_misclassified.csv", index=False)

    if "text" in predictions_dict[first_model]["test"].columns:
        analyze_text_features(predictions_dict).to_csv(
            output_dir / "text_features_analysis.csv", index=False
        )

    return results


# -----------------------------------------------------------------------------
# User‑facing display helper – unchanged public surface but uses new renderer
# -----------------------------------------------------------------------------


def display_detailed_results(
    results: Dict[str, Dict[str, Any]],
    model_order: Optional[List[str]] = None,
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
