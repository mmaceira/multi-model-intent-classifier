"""\
Dataset module.

Classes:
- None

Functions:
- _extract_year
- load_data
- load_data_temporal

Created: 2025-05-03
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import reuters
from collections import Counter
import re as _re
import warnings as _warnings

def _extract_year(text: str) -> int | None:
    m = _re.search(r"<DATE>[^<]*?(\d{2})-(\w{3})-(\d{2,4})", text)
    if not m:
        return None
    _, _, year = m.groups()
    year = int(year)
    if year < 100:  # Reuters stores 2‑digit years
        year += 1900
    return year

def load_data(n_classes: Optional[int] = None):
    """Return train/test splits restricted to *n_classes* most frequent labels."""
    nltk.download("reuters", quiet=True)
    train_ids = [fid for fid in reuters.fileids() if fid.startswith("train")]
    test_ids  = [fid for fid in reuters.fileids() if fid.startswith("test")]

    y_train_all = [reuters.categories(fid)[0] for fid in train_ids]
    top_labels = (
        [lab for lab, _ in Counter(y_train_all).most_common(n_classes)]
        if n_classes else sorted(set(y_train_all))
    )
    def _filtered(ids): return [fid for fid in ids if reuters.categories(fid)[0] in top_labels]

    train_ids, test_ids = map(_filtered, [train_ids, test_ids])

    X_train = [reuters.raw(fid) for fid in train_ids]
    y_train = [reuters.categories(fid)[0] for fid in train_ids]
    X_test  = [reuters.raw(fid) for fid in test_ids]
    y_test  = [reuters.categories(fid)[0] for fid in test_ids]
    return X_train, y_train, X_test, y_test, top_labels

def load_data_temporal(cutoff_year: int = 1996, n_classes: Optional[int] = None):
    """Temporal hold‑out split: train on docs before *cutoff_year*, test on or after."""
    nltk.download("reuters", quiet=True)
    year_map = {fid: _extract_year(reuters.raw(fid)) for fid in reuters.fileids()}
    train_ids = [fid for fid, y in year_map.items() if y is not None and y < cutoff_year]
    test_ids  = [fid for fid, y in year_map.items() if y is not None and y >= cutoff_year]
    if not test_ids:
        _warnings.warn("No test documents found – falling back to default split.")
        return load_data(n_classes=n_classes)

    y_train_all = [reuters.categories(fid)[0] for fid in train_ids]
    if n_classes:
        top_labels = [lab for lab, _ in Counter(y_train_all).most_common(n_classes)]
    else:
        top_labels = sorted(set(y_train_all))
    def _filtered(ids): return [fid for fid in ids if reuters.categories(fid)[0] in top_labels]

    train_ids, test_ids = map(_filtered, [train_ids, test_ids])

    X_train = [reuters.raw(fid) for fid in train_ids]
    y_train = [reuters.categories(fid)[0] for fid in train_ids]
    X_test  = [reuters.raw(fid) for fid in test_ids]
    y_test  = [reuters.categories(fid)[0] for fid in test_ids]
    return X_train, y_train, X_test, y_test, top_labels
