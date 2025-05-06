"""
Evaluation package for text classification models.

This package provides a unified evaluation workflow for text classification models.
It reads model artifacts (predictions, probabilities) and computes comprehensive
metrics and visualizations without touching the models directly, ensuring
deterministic and reproducible evaluation.
"""

from .evaluation import run_evaluations, display_detailed_results
from .visualization import (
    plot_label_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_model_comparisons,
    plot_top_misclassifications,
    visualize_error_distribution,
    generate_detailed_error_report,
    plot_top_error_types,
)

__all__ = [
    'run_evaluations',
    'display_detailed_results',
    'plot_label_distribution',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_model_comparisons',
    'plot_top_misclassifications',
    'visualize_error_distribution',
    'generate_detailed_error_report',
    'plot_top_error_types',
] 