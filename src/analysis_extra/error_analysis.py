from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Sequence

def top_misclassifications(
    estimator: Any,
    X_test: Sequence[str],
    y_test: Sequence[str|int],
    *, 
    top_n: int = 20
) -> pd.DataFrame:
    """Return a DataFrame with the *top_n* most confident mis‑predictions.
    
    Parameters
    ----------
    estimator
        Fitted classifier with `predict` and optionally `predict_proba`.
    X_test, y_test
        Test split.
    top_n
        Number of mis‑predicted instances to return.
    """
    # Convert X_test to a list if it's not already one
    X_test_list = list(X_test)
    
    y_pred = estimator.predict(X_test)
    wrong_idx = np.where(np.array(y_pred) != np.array(y_test))[0]
    
    # Handle empty wrong_idx
    if len(wrong_idx) == 0:
        return pd.DataFrame(columns=["index", "text", "y_true", "y_pred", "confidence"])
    
    # Get misclassified samples
    X_test_wrong = [X_test_list[i] for i in wrong_idx]
    
    conf = None
    if hasattr(estimator, "predict_proba"):
        probas = estimator.predict_proba(X_test_wrong)
        conf = probas.max(axis=1)
    else:
        # fall back to decision function magnitude if present
        if hasattr(estimator, "decision_function"):
            decf = estimator.decision_function(X_test_wrong)
            if decf.ndim == 1:
                conf = np.abs(decf)
            else:
                conf = decf.max(axis=1)
    if conf is None:
        conf = [float("nan")] * len(wrong_idx)
    
    df = pd.DataFrame(
        {
            "index": wrong_idx,
            "text": X_test_wrong,
            "y_true": [y_test[i] for i in wrong_idx],
            "y_pred": [y_pred[i] for i in wrong_idx],
            "confidence": conf,
        }
    )
    df_sorted = df.sort_values("confidence", ascending=False).head(top_n)
    return df_sorted.reset_index(drop=True)
