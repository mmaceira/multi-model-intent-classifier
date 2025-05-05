"""Utility helpers shared by qualitative + quantitative error analysis.

Having this single source of truth simplifies maintenance and ensures that
both notebooks and automated scripts compute the exact same numbers.
"""

from pathlib import Path
from typing import List
import pandas as pd

def load_all_prediction_files(folder: Path) -> pd.DataFrame:
    """Load all CSVs inside *folder* into a concatenated DataFrame."""
    dfs = []
    for csv in folder.glob("*.csv"):
        dfs.append(pd.read_csv(csv))
    if not dfs:
        raise FileNotFoundError(f"No prediction CSVs found under {folder}")
    return pd.concat(dfs, ignore_index=True)

def analyze_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple stats (length, #digits…) per document."""
    import numpy as np
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["digit_ratio"] = df["text"].str.count(r"\d") / df["char_len"]
    return df.describe()

def visualize_error_distribution(df: pd.DataFrame, out: Path) -> None:
    """Simple bar plot of correctly vs incorrectly classified samples."""
    import matplotlib.pyplot as plt
    counts = df["is_error"].value_counts().sort_index()
    counts.plot.bar()
    plt.xlabel("Classification correct?")
    plt.ylabel("#documents")
    plt.tight_layout()
    plt.savefig(out, dpi=150)

def generate_detailed_error_report(df: pd.DataFrame, out: Path) -> None:
    """Write every misclassified sample to *out* as MarkDown."""
    errors = df[df["is_error"]]
    lines: List[str] = [f"# Error report – {len(errors)} samples\n"]
    for _, row in errors.iterrows():
        lines.append(f"## id={row['id']} – true={row['y_true']} / pred={row['y_pred']}\n")
        lines.append(row["text"])
        lines.append("\n\n---\n")
    Path(out).write_text("\n".join(lines), encoding="utf-8")
