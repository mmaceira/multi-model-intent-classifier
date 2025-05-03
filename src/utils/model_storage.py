"""Utility functions for storing and loading trained models."""
import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

def save_model(
    model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    n_classes: int,
    base_dir: str = "models"
) -> None:
    """
    Save a trained model and its metrics in an organized directory structure.
    
    Args:
        model: The trained model to save
        model_name: Name of the model (will be used as directory name)
        metrics: Dictionary of model metrics
        n_classes: Number of classes the model was trained on
        base_dir: Base directory for storing models
    """
    # Create directory structure: base_dir/num_classes_N/model_name/
    n_classes_str = f"{n_classes:04d}"
    model_dir = Path(base_dir) / f"num_classes_{n_classes_str}" / model_name.replace(' ', '_')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'model.joblib'
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Save metrics
    metrics_path = model_dir / 'metrics.joblib'
    joblib.dump(metrics, metrics_path)
    print(f"Saved metrics to {metrics_path}")

def load_model(
    model_name: str,
    n_classes: int,
    base_dir: str = "models"
) -> tuple[Any, Dict[str, Any]]:
    """
    Load a saved model and its metrics.
    
    Args:
        model_name: Name of the model to load
        n_classes: Number of classes the model was trained on
        base_dir: Base directory where models are stored
        
    Returns:
        Tuple of (model, metrics)
    """
    n_classes_str = f"{n_classes:04d}"
    model_dir = Path(base_dir) / f"num_classes_{n_classes_str}" / model_name.replace(' ', '_')
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load model
    model_path = model_dir / 'model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    
    # Load metrics
    metrics_path = model_dir / 'metrics.joblib'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    metrics = joblib.load(metrics_path)
    
    return model, metrics

def list_available_models(
    n_classes: Optional[int] = None,
    base_dir: str = "models"
) -> Dict[int, list[str]]:
    """
    List all available models in the storage directory.
    
    Args:
        n_classes: Optional filter for specific number of classes
        base_dir: Base directory where models are stored
        
    Returns:
        Dictionary mapping number of classes to list of available model names
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}
        
    available_models = {}
    for num_classes_dir in base_path.glob("num_classes_*"):
        try:
            n = int(num_classes_dir.name.split('_')[-1])
        except ValueError:
            continue
            
        if n_classes is not None and n != n_classes:
            continue
            
        model_names = [d.name.replace('_', ' ') for d in num_classes_dir.iterdir() if d.is_dir()]
        if model_names:
            available_models[n] = model_names
            
    return available_models 