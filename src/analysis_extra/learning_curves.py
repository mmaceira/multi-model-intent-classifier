
from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from typing import Sequence, Any, Tuple

def plot_learning_curves(
    estimator: Any,
    X: Sequence[str],
    y: Sequence[str|int],
    *,
    cv: int = 5,
    train_sizes: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    scoring: str = "accuracy",
):
    """Render a learning curve plot for *estimator*."""
    import numpy as np
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure()
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.plot(train_sizes_abs, train_mean, label="Training", marker="o")
    plt.plot(train_sizes_abs, val_mean, label="Validation", marker="o")
    plt.title(f"Learning curves ({scoring})")
    plt.xlabel("Training examples")
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    return plt.gca()
