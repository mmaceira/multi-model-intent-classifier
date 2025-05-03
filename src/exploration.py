"""exploration.py – Quick EDA helpers for Reuters corpus."""
from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "class_frequency",
    "length_distribution"
]

def class_frequency(labels: List[str], plot: bool = False, save_path: str | None = "results/class_distribution.png", top_n: int | None = None) -> pd.DataFrame:
    """Return DataFrame with counts sorted descending. Always plot and/or save the class distribution. Optionally limit to top_n classes."""
    freq = Counter(labels)
    df = pd.DataFrame.from_dict(freq, orient='index', columns=['count'])
    df = df.sort_values('count', ascending=False)
    if top_n:
        df = df.head(top_n)
    plt.figure(figsize=(8,4) if top_n else (6,3))
    plt.bar(df.index, df['count'])
    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.title('Class distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return df

def length_distribution(texts: List[str], save_path: str | None = "results/length_distribution.png") -> Tuple[float, float]:
    """Plot and return mean, median article length (words). Always show and optionally save the plot to a file."""
    lengths = [len(t.split()) for t in texts]
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(lengths, bins=40)
    ax.set_xlabel('Words per article')
    ax.set_ylabel('Frequency')
    ax.set_title('Article length distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return mean_len, median_len
