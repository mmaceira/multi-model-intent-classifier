
from __future__ import annotations
from typing import Sequence, Tuple
from collections import Counter
import pandas as pd

def vocabulary_drift(train_texts: Sequence[str], test_texts: Sequence[str], *, top_k: int = 2000) -> pd.DataFrame:
    """Compute token frequency drift between train and test splits."""
    def _tokenize(s: str):
        return s.lower().split()
    train_counts = Counter(token for doc in train_texts for token in _tokenize(doc))
    test_counts = Counter(token for doc in test_texts for token in _tokenize(doc))
    
    # restrict to top_k in either split
    vocab = set([w for w,_ in train_counts.most_common(top_k)] + [w for w,_ in test_counts.most_common(top_k)])
    rows=[]
    for w in vocab:
        tr = train_counts.get(w,0)
        te = test_counts.get(w,0)
        total = tr + te
        if total == 0:
            continue
        rows.append({"token": w, "train_freq": tr/total, "test_freq": te/total, "abs_diff": abs(tr/total - te/total)})
    df = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)
    return df
