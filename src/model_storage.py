"""\
Model storage module for saving and loading trained models.

This module provides functions for persisting trained models to disk
and loading them back into memory. It uses joblib for efficient
serialization of scikit-learn models.

Functions:
- save_model: Save a trained model to disk
- load_model: Load a saved model from disk

Created: 2025-05-03
"""

import os
import joblib
from typing import Any

def save_model(model: Any, path: str) -> None:
    """Save a trained model to disk.
    
    This function saves a trained model to the specified path using
    joblib for efficient serialization. It creates the directory if
    it doesn't exist.
    
    Args:
        model: The trained model to save
        path: Path where the model should be saved
        
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
        >>> save_model(model, 'models/logreg.joblib')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, path)

def load_model(path: str) -> Any:
    """Load a saved model from disk.
    
    This function loads a saved model from the specified path using
    joblib. It raises a FileNotFoundError if the model file doesn't
    exist.
    
    Args:
        path: Path to the saved model
        
    Returns:
        The loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        
    Example:
        >>> model = load_model('models/logreg.joblib')
        >>> y_pred = model.predict(X_test)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    return joblib.load(path) 