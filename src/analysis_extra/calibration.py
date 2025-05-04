
from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from typing import Sequence, Any

def plot_calibration_curve(estimator: Any, X: Sequence[str], y: Sequence[str|int], *, n_bins: int = 10):
    """Plot probability calibration curve (binary or OvR for multi‑class)."""
    disp = CalibrationDisplay.from_estimator(estimator, X, y, n_bins=n_bins)
    plt.title("Calibration curve")
    plt.tight_layout()
    return disp
