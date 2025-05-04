"""Enhanced evaluation module for text classification models.

This module provides an enhanced version of run_evaluations with improved logging
while maintaining the same interface and functionality as the original.
"""

import os
import logging
from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import the original functions to maintain compatibility
from src.utils.model_storage import load_model
from src.evaluation import _persist_report, _persist_confusion

def run_evaluations_enhanced(model_or_path: Union[Any, str],
                    X_test: List[str], y_test: np.ndarray,
                    X_train: Optional[List[str]] = None,
                    y_train: Optional[np.ndarray] = None,
                    output_dir: str = "results",
                    model_name: str = "unknown") -> Dict[str, Any]:
    """Run comprehensive evaluation of a text classification model with enhanced logging.
    
    This function performs a complete evaluation of a text classifier,
    including computing metrics, generating visualizations, and analyzing
    errors. It saves all results to the specified output directory.
    
    Args:
        model_or_path: *Either* a fitted model **or** a path to a
            ``.joblib`` file saved by `run_trainings`
        X_test: List of test documents
        y_test: Array of test labels
        X_train, y_train: (optional) supply to compute train metrics
        output_dir: Directory to save results (default: 'results')
        model_name: Name of the model for logging purposes
        
    Returns:
        Dictionary containing evaluation results
        
    Example:
        >>> results = run_evaluations_enhanced(model, X_test, y_test, model_name="LinearSVM")
        >>> print(f"Macro F1: {results['macro_f1']:.3f}")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"🔍 Evaluating model: {model_name}")
    
    # Load the model if a string path was supplied
    if isinstance(model_or_path, str):
        logging.info(f"  • Loading model from {model_or_path}")
        model = load_model(model_or_path)
    else:
        model = model_or_path

    # Evaluate both splits
    splits = {"test": (X_test, y_test)}
    if X_train is not None and y_train is not None:
        splits["train"] = (X_train, y_train)

    # Prepare the results dictionary with top-level metrics
    results = {}
    
    for split, (X_split, y_split) in splits.items():
        logging.info(f"  • Evaluating on {split} set ({len(X_split)} samples)...")
        
        # Make predictions
        y_pred = model.predict(X_split)
        
        # Calculate metrics
        accuracy = accuracy_score(y_split, y_pred)
        logging.info(f"    ✓ {split.capitalize()} accuracy: {accuracy:.4f}")
        
        # Calculate classification report
        report_dict = classification_report(y_split, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_split, y_pred)
        
        # Extract scores
        macro_f1 = report_dict['macro avg']['f1-score']
        weighted_f1 = report_dict['weighted avg']['f1-score']
        
        logging.info(f"    ✓ {split.capitalize()} macro F1: {macro_f1:.4f}, weighted F1: {weighted_f1:.4f}")
        
        # Save results
        report_path = os.path.join(output_dir, f"{split}_report.csv")
        confusion_path = os.path.join(output_dir, f"{split}_confusion.png")
        
        _persist_report(report_df, report_path)
        _persist_confusion(cm, confusion_path)
        logging.info(f"    ✓ Saved {split} reports to {output_dir}")
        
        # Store metrics in the split-specific results
        results[split] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report_dict
        }
    
    # For compatibility with the notebook, add top-level metrics using test split
    if 'test' in results:
        results['accuracy'] = results['test']['accuracy']
        results['macro_f1'] = results['test']['macro_f1'] 
        results['weighted_f1'] = results['test']['weighted_f1']
        results['f1_macro'] = results['test']['macro_f1']  # For backward compatibility
    
    logging.info(f"  ✓ Evaluation of {model_name} completed")
    
    return results

# Make it easier to use by making this the default import
run_evaluations = run_evaluations_enhanced 