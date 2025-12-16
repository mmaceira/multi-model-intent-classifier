"""Shared utilities for multilabel dataset analysis scripts.

This module contains helpers that are reused by dataset-specific analysis
scripts (e.g. CLINC-style CSV and other multilabel CSV exports).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

from intent_classifier.utils.label_utils import binarize_labels


def parse_labels_from_csv(label_series: pd.Series) -> list[list[str]]:
    """Parse comma-separated labels from a pandas Series."""
    result: list[list[str]] = []
    for label_str in label_series:
        if pd.isna(label_str) or label_str == "":
            result.append([])
        else:
            labels = [label.strip() for label in str(label_str).split(",") if label.strip()]
            result.append(labels)
    return result


def run_multilabel_analysis(
    *,
    raw_csv_path: Path,
    output_dir: Path,
    predictions_dir: Path,
    best_model_name: str,
    category_column: str,
    tags_column: str,
    normalize_labels_fn: Callable[[Iterable[str]], Sequence[str]],
    analyze_normalization_fn: Callable[[Iterable[Iterable[str]]], dict[str, Counter]],
    print_normalization_report_fn: Callable[[Counter, Counter, Counter, Counter], None],
) -> None:
    """Run end-to-end analysis for a multilabel CSV dataset.

    The dataset-specific scripts are responsible for providing the paths and
    normalization functions; this helper focuses on the shared analysis logic.
    """
    from intent_classifier.utils.paths import get_repo_root

    repo_root = get_repo_root()
    raw_csv_path = repo_root / raw_csv_path
    output_dir = repo_root / output_dir
    predictions_dir = repo_root / predictions_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Dataset Analysis")
    print("=" * 80)
    print(f"Loading raw CSV from: {raw_csv_path}")
    print()

    # Load raw CSV
    df = pd.read_csv(raw_csv_path)

    # Parse and normalize tags
    df["tag_list_raw"] = df[tags_column].apply(
        lambda x: [t.strip() for t in str(x).split(",") if t.strip()]
    )
    df["tag_list"] = df["tag_list_raw"].apply(normalize_labels_fn)

    # Print normalization report
    print("\n" + "=" * 80)
    print("NORMALIZATION REPORT")
    print("=" * 80)
    normalization_report = analyze_normalization_fn(df["tag_list_raw"])
    print_normalization_report_fn(
        normalization_report["raw_label_counts"],
        normalization_report["normalized_label_counts"],
        normalization_report["merge_counts"],
        normalization_report["skipped_time_labels"],
    )

    # 1. Generate label frequency CSV
    print("\n" + "=" * 80)
    print("GENERATING LABEL FREQUENCY STATISTICS")
    print("=" * 80)

    all_labels: list[str] = []
    label_to_categories: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        for label in row["tag_list"]:
            all_labels.append(label)
            if label not in label_to_categories:
                label_to_categories[label] = set()
            label_to_categories[label].add(row[category_column])

    label_counts = Counter(all_labels)
    label_frequency_df = pd.DataFrame(
        [
            {
                "label": label,
                "frequency": count,
                "n_categories": len(label_to_categories[label]),
            }
            for label, count in label_counts.most_common()
        ]
    )
    label_frequency_df.to_csv(output_dir / "label_frequency.csv", index=False)
    print(f"✓ Saved label frequency to: {output_dir / 'label_frequency.csv'}")
    print(f"  Total labels: {len(label_frequency_df)}")
    print(
        f"  Labels with frequency < 5: "
        f"{len(label_frequency_df[label_frequency_df['frequency'] < 5])}"
    )

    # 2. Generate label co-occurrence CSV
    print("\nGenerating co-occurrence statistics...")
    cooccurrence_counts: Counter = Counter()
    for tag_list in df["tag_list"]:
        if len(tag_list) > 1:
            for label_a, label_b in combinations(sorted(tag_list), 2):
                cooccurrence_counts[(label_a, label_b)] += 1

    cooccurrence_df = pd.DataFrame(
        [
            {
                "label_a": label_a,
                "label_b": label_b,
                "cooccurrence_count": count,
            }
            for (label_a, label_b), count in cooccurrence_counts.most_common()
        ]
    )
    cooccurrence_df.to_csv(output_dir / "label_cooccurrence.csv", index=False)
    print(f"✓ Saved co-occurrence to: {output_dir / 'label_cooccurrence.csv'}")
    print(f"  Total co-occurrence pairs: {len(cooccurrence_df)}")
    print("  Top 10 co-occurring pairs:")
    for _, row in cooccurrence_df.head(10).iterrows():
        print(f"    {row['label_a']} + {row['label_b']}: {row['cooccurrence_count']}")

    # 3. Generate label quality report
    print("\nGenerating label quality report...")
    label_quality_data: list[dict[str, object]] = []
    for label in label_counts.keys():
        # Get top co-occurring labels
        top_cooccurring = cooccurrence_df[
            (cooccurrence_df["label_a"] == label) | (cooccurrence_df["label_b"] == label)
        ].head(5)
        top_labels: list[str] = []
        for _, row in top_cooccurring.iterrows():
            other_label = row["label_b"] if row["label_a"] == label else row["label_a"]
            top_labels.append(f"{other_label}({int(row['cooccurrence_count'])})")
        top_cooccurring_str = "; ".join(top_labels) if top_labels else ""

        label_quality_data.append(
            {
                "label": label,
                "frequency": label_counts[label],
                "n_categories": len(label_to_categories[label]),
                "top_cooccurring_labels": top_cooccurring_str,
            }
        )

    label_quality_df = pd.DataFrame(label_quality_data)
    label_quality_df = label_quality_df.sort_values("frequency", ascending=False)
    label_quality_df.to_csv(output_dir / "label_quality_report.csv", index=False)
    print(f"✓ Saved label quality report to: {output_dir / 'label_quality_report.csv'}")

    # Print summary of low-frequency labels
    low_freq_labels = label_quality_df[label_quality_df["frequency"] < 5]
    if len(low_freq_labels) > 0:
        print(f"\n⚠ Labels with frequency < 5 ({len(low_freq_labels)} labels):")
        for _, row in low_freq_labels.head(10).iterrows():
            print(f"  {row['label']}: {row['frequency']} occurrences")

    # 4. Generate metrics by category
    print("\n" + "=" * 80)
    print("GENERATING METRICS BY CATEGORY")
    print("=" * 80)

    # Load predictions
    pred_file = predictions_dir / best_model_name / "test_predictions.csv"
    if not pred_file.exists():
        print(f"⚠ Predictions file not found: {pred_file}")
        print("  Skipping metrics by category and label count analysis.")
        print("  Run the prediction pipeline first to generate predictions.")
        return

    print(f"Loading predictions from: {pred_file}")
    pred_df = pd.read_csv(pred_file)

    # Parse labels
    y_true = parse_labels_from_csv(pred_df["y_true"])
    y_pred = parse_labels_from_csv(pred_df["y_pred"])

    # Merge with original data to get category
    # Note: predictions might not have IDs, so we'll match by index
    if "id" in pred_df.columns:
        merged = pred_df.merge(
            df[["id", category_column, "tag_list"]],
            on="id",
            how="left",
        )
    else:
        # If no ID column, match by index (assuming same order)
        merged = pred_df.copy()
        merged[category_column] = df[category_column].values[: len(merged)]
        merged["tag_list"] = df["tag_list"].values[: len(merged)]

    # Compute metrics per category
    category_metrics: list[dict[str, object]] = []
    for category in merged[category_column].unique():
        if pd.isna(category):
            continue

        category_mask = merged[category_column] == category
        category_y_true = [y_true[i] for i in range(len(y_true)) if category_mask.iloc[i]]
        category_y_pred = [y_pred[i] for i in range(len(y_pred)) if category_mask.iloc[i]]

        if not category_y_true:
            continue

        # Get all classes
        all_classes = sorted(
            set(tag for labels in category_y_true if labels for tag in labels)
            | set(tag for labels in category_y_pred if labels for tag in labels)
        )

        if not all_classes:
            continue

        # Convert to binary
        y_true_binary, _ = binarize_labels(category_y_true, classes=all_classes)
        y_pred_binary, _ = binarize_labels(category_y_pred, classes=all_classes)

        # Compute metrics
        micro_f1 = f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
        hamming = hamming_loss(y_true_binary, y_pred_binary)
        exact_match = accuracy_score(y_true_binary, y_pred_binary)
        mean_n_true = float(np.mean([len(labels) for labels in category_y_true]))
        mean_n_pred = float(np.mean([len(labels) for labels in category_y_pred]))

        category_metrics.append(
            {
                "category": category,
                "n_samples": len(category_y_true),
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "hamming_loss": hamming,
                "exact_match": exact_match,
                "mean_n_true": mean_n_true,
                "mean_n_pred": mean_n_pred,
            }
        )

    category_metrics_df = pd.DataFrame(category_metrics)
    category_metrics_df = category_metrics_df.sort_values("micro_f1", ascending=False)
    category_metrics_df.to_csv(output_dir / "metrics_by_category.csv", index=False)
    print(f"✓ Saved metrics by category to: {output_dir / 'metrics_by_category.csv'}")

    # Print summary
    print("\nMetrics by category:")
    for _, row in category_metrics_df.iterrows():
        print(
            f"  {row['category']}: micro_f1={row['micro_f1']:.3f}, "
            f"hamming={row['hamming_loss']:.3f}, exact_match={row['exact_match']:.3f}, "
            f"n_samples={int(row['n_samples'])}"
        )

    # 5. Generate metrics by label count bucket
    print("\n" + "=" * 80)
    print("GENERATING METRICS BY LABEL COUNT BUCKET")
    print("=" * 80)

    # Create buckets based on true label count
    bucket_metrics: list[dict[str, object]] = []
    for k in (1, 2, 3):
        bucket_mask = np.array([len(labels) == k for labels in y_true])
        if bucket_mask.sum() == 0:
            continue

        bucket_y_true = [y_true[i] for i in range(len(y_true)) if bucket_mask[i]]
        bucket_y_pred = [y_pred[i] for i in range(len(y_pred)) if bucket_mask[i]]

        # Get all classes
        all_classes = sorted(
            set(tag for labels in bucket_y_true if labels for tag in labels)
            | set(tag for labels in bucket_y_pred if labels for tag in labels)
        )

        if not all_classes:
            continue

        # Convert to binary
        y_true_binary, _ = binarize_labels(bucket_y_true, classes=all_classes)
        y_pred_binary, _ = binarize_labels(bucket_y_pred, classes=all_classes)

        # Compute metrics
        micro_f1 = f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
        hamming = hamming_loss(y_true_binary, y_pred_binary)
        exact_match = accuracy_score(y_true_binary, y_pred_binary)
        mean_n_true = float(np.mean([len(labels) for labels in bucket_y_true]))
        mean_n_pred = float(np.mean([len(labels) for labels in bucket_y_pred]))

        bucket_metrics.append(
            {
                "label_count": k,
                "n_samples": len(bucket_y_true),
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "hamming_loss": hamming,
                "exact_match": exact_match,
                "mean_n_true": mean_n_true,
                "mean_n_pred": mean_n_pred,
            }
        )

    # k >= 4 bucket
    bucket_mask = np.array([len(labels) >= 4 for labels in y_true])
    if bucket_mask.sum() > 0:
        bucket_y_true = [y_true[i] for i in range(len(y_true)) if bucket_mask[i]]
        bucket_y_pred = [y_pred[i] for i in range(len(y_pred)) if bucket_mask[i]]

        all_classes = sorted(
            set(tag for labels in bucket_y_true if labels for tag in labels)
            | set(tag for labels in bucket_y_pred if labels for tag in labels)
        )

        if all_classes:
            y_true_binary, _ = binarize_labels(bucket_y_true, classes=all_classes)
            y_pred_binary, _ = binarize_labels(bucket_y_pred, classes=all_classes)

            micro_f1 = f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0)
            macro_f1 = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
            hamming = hamming_loss(y_true_binary, y_pred_binary)
            exact_match = accuracy_score(y_true_binary, y_pred_binary)
            mean_n_true = float(np.mean([len(labels) for labels in bucket_y_true]))
            mean_n_pred = float(np.mean([len(labels) for labels in bucket_y_pred]))

            bucket_metrics.append(
                {
                    "label_count": ">=4",
                    "n_samples": len(bucket_y_true),
                    "micro_f1": micro_f1,
                    "macro_f1": macro_f1,
                    "hamming_loss": hamming,
                    "exact_match": exact_match,
                    "mean_n_true": mean_n_true,
                    "mean_n_pred": mean_n_pred,
                }
            )

    bucket_metrics_df = pd.DataFrame(bucket_metrics)
    bucket_metrics_df.to_csv(output_dir / "metrics_by_label_count.csv", index=False)
    print(f"✓ Saved metrics by label count to: {output_dir / 'metrics_by_label_count.csv'}")

    # Print summary table
    print("\nMetrics by label count:")
    header = (
        f"{'Label Count':<12} {'N Samples':<12} {'Micro F1':<12} "
        f"{'Hamming':<12} {'Exact Match':<12}"
    )
    print(header)
    print("-" * 60)
    for _, row in bucket_metrics_df.iterrows():
        print(
            f"{str(row['label_count']):<12} {int(row['n_samples']):<12} "
            f"{row['micro_f1']:<12.3f} {row['hamming_loss']:<12.3f} {row['exact_match']:<12.3f}"
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)
