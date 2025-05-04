
from __future__ import annotations
from typing import Sequence, Any
import shap
import numpy as np
import pandas as pd

def shap_logreg_explanations(
    estimator: Any,
    vectorizer: Any,
    feature_names: Sequence[str] | None = None,
    max_display: int = 20,
):
    """Return a SHAP summary DataFrame for a linear/logistic model.

    Parameters
    ----------
    estimator
        Fitted linear classifier with `coef_`.
    vectorizer
        Fitted vectorizer (`TfidfVectorizer`, `CountVectorizer`, …) to obtain feature names.
    feature_names
        Optional explicit feature names if vectorizer not provided.
    """
    if feature_names is None:
        feature_names = np.array(vectorizer.get_feature_names_out())
    explainer = shap.LinearExplainer(estimator, feature_names=feature_names, seed=42)
    shap_values = explainer.shap_values(vectorizer.transform(vectorizer.get_feature_names_out()))
    # Aggregate absolute Shap values
    abs_mean = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "abs_shap": abs_mean})
    return df.sort_values("abs_shap", ascending=False).head(max_display)
