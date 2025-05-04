"""
Extra analysis utilities: error introspection, learning curves, embedding visualisations,
probability calibration, vocabulary drift and SHAP explanations.

Each helper is intentionally *lightweight* and free of heavy dependencies
unless strictly required.
"""
from .error_analysis import top_misclassifications
from .learning_curves import plot_learning_curves
from .embedding_viz import plot_tsne
from .calibration import plot_calibration_curve
from .vocab_drift import vocabulary_drift
# from .shap_explain import shap_logreg_explanations  # Commented out due to NumPy 2.0+ compatibility issue
from .model_analyzer import analyze_all_models
