
"""dataset.py – Load and filter the Reuters‑21578 corpus."""
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import reuters
from collections import Counter

def load_data(n_classes: Optional[int] = None) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Return train/test splits restricted to the *n_classes* most frequent labels."""
    nltk.download("reuters", quiet=True)
    train_ids = [fid for fid in reuters.fileids() if fid.startswith("train")]
    test_ids  = [fid for fid in reuters.fileids() if fid.startswith("test")]

    y_train_all = [reuters.categories(fid)[0] for fid in train_ids]
    if n_classes is not None and n_classes > 0:
        top_labels = [lab for lab, _ in Counter(y_train_all).most_common(n_classes)]
    else:
        top_labels = sorted(set(y_train_all))

    def filtered(ids):
        return [fid for fid in ids if reuters.categories(fid)[0] in top_labels]

    train_ids = filtered(train_ids)
    test_ids  = filtered(test_ids)

    X_train = [reuters.raw(fid) for fid in train_ids]
    y_train = [reuters.categories(fid)[0] for fid in train_ids]
    X_test  = [reuters.raw(fid) for fid in test_ids]
    y_test  = [reuters.categories(fid)[0] for fid in test_ids]

    return X_train, y_train, X_test, y_test, top_labels


def load_data_temporal(
    cutoff_year: int = 1996,
    n_classes: Optional[int] = None
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Temporal hold‑out split: train on docs before *cutoff_year*, test on *cutoff_year* or later."""
    import re, datetime as _dt, warnings
    nltk.download("reuters", quiet=True)
    def _extract_year(text: str) -> int | None:
        m = re.search(r"<DATE>[^<]*?(\d{2})-(\w{3})-(\d{2,4})", text)
        if not m:
            return None
        day, mon, year = m.groups()
        year = int(year)
        # Reuters SGML stores 2‑digit years like 87 for 1987
        if year < 100:
            year += 1900
        return year
    year_map = {}
    for fid in reuters.fileids():
        y = _extract_year(reuters.raw(fid))
        year_map[fid] = y
    # split ids
    train_ids = [fid for fid, y in year_map.items() if y is not None and y < cutoff_year]
    test_ids  = [fid for fid, y in year_map.items() if y is not None and y >= cutoff_year]
    if not test_ids:
        warnings.warn("No test documents found for cutoff_year – falling back to default split.")
        return load_data(n_classes=n_classes)
    # filter by classes
    y_train_all = [reuters.categories(fid)[0] for fid in train_ids]
    if n_classes is not None and n_classes > 0:
        from collections import Counter
        top_labels = [lab for lab, _ in Counter(y_train_all).most_common(n_classes)]
    else:
        top_labels = sorted(set(y_train_all))

    def filtered(ids):
        return [fid for fid in ids if reuters.categories(fid)[0] in top_labels]

    train_ids = filtered(train_ids)
    test_ids  = filtered(test_ids)

    X_train = [reuters.raw(fid) for fid in train_ids]
    y_train = [reuters.categories(fid)[0] for fid in train_ids]
    X_test  = [reuters.raw(fid) for fid in test_ids]
    y_test  = [reuters.categories(fid)[0] for fid in test_ids]

    return X_train, y_train, X_test, y_test, top_labels
