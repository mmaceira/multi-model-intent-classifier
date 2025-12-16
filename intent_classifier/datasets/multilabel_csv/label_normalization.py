"""
Label normalization module for CSV-based multilabel datasets.

This module provides functions to normalize labels by:
- Merging similar labels
- Skipping time-like labels (years, months)
- Never dropping labels due to low frequency
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

# Label merge map: raw label -> canonical label
LABEL_MERGE_MAP: dict[str, str] = {
    # Merge similar labels here
    "Assegurances": "asseguranca",
    "assegurances": "asseguranca",
    "integracio_software": "integracio",
    "integracio software": "integracio",
    # Add more mappings as needed
}

# Time-like labels to skip (years, months)
TIME_LABELS: set[str] = {
    "2025",
    "desembre",
    "novembre",
    "octubre",
    "setembre",
    "agost",
    "juliol",
    "juny",
    "maig",
    "abril",
    "març",
    "febrer",
    "gener",
}


def normalize_labels(raw_labels: Iterable[str]) -> list[str]:
    """
    Normalize a list of raw labels:
    - strip whitespace
    - lower-case where needed
    - merge similar labels using LABEL_MERGE_MAP
    - drop year/month labels in TIME_LABELS
    - NEVER drop labels just because they are rare.

    Args:
        raw_labels: Iterable of raw label strings

    Returns:
        List of normalized labels (deduplicated, stable order)
    """
    cleaned: list[str] = []
    for label in raw_labels:
        label_clean = label.strip()
        if not label_clean:
            continue

        # Normalization: lower-case
        normalized = label_clean.lower()

        # Explicit merge map (overrides lower-casing if needed)
        if normalized in LABEL_MERGE_MAP:
            normalized = LABEL_MERGE_MAP[normalized]

        # Skip time-like labels (year/month)
        if normalized in TIME_LABELS:
            # Will be logged by analyze_normalization
            continue

        cleaned.append(normalized)

    # Deduplicate while keeping stable order
    seen = set()
    result: list[str] = []
    for label_item in cleaned:
        if label_item not in seen:
            seen.add(label_item)
            result.append(label_item)

    return result


def analyze_normalization(
    all_raw_label_lists: Iterable[Iterable[str]],
) -> dict[str, Counter]:
    """
    Analyze normalization across all raw label lists.

    Args:
        all_raw_label_lists: Iterable of raw label lists (each is a list of strings)

    Returns:
        Dictionary with keys:
        - raw_label_counts: Counter of raw labels
        - normalized_label_counts: Counter of normalized labels
        - merge_counts: Counter of (raw_label, normalized_label) pairs
        - skipped_time_labels: Counter of skipped time labels
    """
    raw_label_counts: Counter = Counter()
    normalized_label_counts: Counter = Counter()
    merge_counts: Counter = Counter()  # (raw, normalized) pairs
    skipped_time_labels: Counter = Counter()

    for raw_label_list in all_raw_label_lists:
        # Count raw labels
        for raw_label in raw_label_list:
            raw_label_counts[raw_label.strip()] += 1

        # Normalize and count
        normalized_list = normalize_labels(raw_label_list)
        for normalized_label in normalized_list:
            normalized_label_counts[normalized_label] += 1

        # Track merges and skips
        for raw_label in raw_label_list:
            raw_label_clean = raw_label.strip()
            if not raw_label_clean:
                continue

            normalized = raw_label_clean.lower()

            # Check if merged
            if normalized in LABEL_MERGE_MAP:
                canonical = LABEL_MERGE_MAP[normalized]
                merge_counts[(raw_label_clean, canonical)] += 1
            elif normalized in TIME_LABELS:
                skipped_time_labels[normalized] += 1

    return {
        "raw_label_counts": raw_label_counts,
        "normalized_label_counts": normalized_label_counts,
        "merge_counts": merge_counts,
        "skipped_time_labels": skipped_time_labels,
    }


def print_normalization_report(
    raw_label_counts: Counter,
    normalized_label_counts: Counter,
    merge_counts: Counter,
    skipped_time_labels: Counter,
) -> None:
    """
    Print a normalization report to stdout.

    Args:
        raw_label_counts: Counter of raw labels
        normalized_label_counts: Counter of normalized labels
        merge_counts: Counter of (raw, normalized) merge pairs
        skipped_time_labels: Counter of skipped time labels
    """
    print("=" * 60)
    print("Label Normalization Report")
    print("=" * 60)
    print(f"Total distinct raw labels: {len(raw_label_counts)}")
    print(f"Total distinct normalized labels: {len(normalized_label_counts)}")
    print(f"Total raw label occurrences: {sum(raw_label_counts.values())}")
    print(f"Total normalized label occurrences: {sum(normalized_label_counts.values())}")
    print()

    if merge_counts:
        print("Merges (raw -> normalized):")
        # Group by normalized label
        merge_by_target: dict[str, list[tuple[str, int]]] = {}
        for (raw, norm), count in merge_counts.items():
            if norm not in merge_by_target:
                merge_by_target[norm] = []
            merge_by_target[norm].append((raw, count))

        for norm in sorted(merge_by_target.keys()):
            for raw, count in sorted(merge_by_target[norm], key=lambda x: (-x[1], x[0])):
                print(f"  {raw} -> {norm} (count: {count})")
        print()

    if skipped_time_labels:
        print("Skipped time-like labels (years/months):")
        for label, count in skipped_time_labels.most_common():
            print(f"  {label}: {count} occurrences")
        print()

    print("Top 20 normalized labels by frequency:")
    for label, count in normalized_label_counts.most_common(20):
        print(f"  {label}: {count}")
    print("=" * 60)
