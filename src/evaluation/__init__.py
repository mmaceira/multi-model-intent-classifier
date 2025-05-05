"""
Evaluation package for text classification models.

This package provides a unified evaluation workflow for text classification models.
It reads model artifacts (predictions, probabilities) and computes comprehensive
metrics and visualizations without touching the models directly, ensuring
deterministic and reproducible evaluation.
"""

from .evaluation import run_evaluations

__all__ = ['run_evaluations'] 