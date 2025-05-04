"""Utility to lazily load models for the FastAPI service.

Assumes joblib‑serialised estimators stored under:
    MODELS_DIR environment variable or default locate 'models/' at repo root.

The estimator must expose `.predict([text]) -> [label]` and, optionally,
`.predict_proba([text])`.
"""
import os
from functools import lru_cache
from pathlib import Path
import joblib

# Use MODELS_DIR env var if provided, else default to '../models'
env_dir = os.getenv('MODELS_DIR')
if env_dir:
    MODELS_DIR = Path(env_dir)
else:
    MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'models'

@lru_cache(maxsize=32)
def get_model(model_name: str):
    """Load and memoise the model with the given name."""
    # First, try direct joblib file
    model_path = MODELS_DIR / f'{model_name}.joblib'
    
    # If the direct path doesn't exist, check if it's a directory containing model.joblib
    if not model_path.exists():
        # Try to find a matching directory (case insensitive)
        for dir_path in MODELS_DIR.iterdir():
            if dir_path.is_dir() and dir_path.name.lower() == model_name.lower():
                model_path = dir_path / 'model.joblib'
                if model_path.exists():
                    return joblib.load(model_path)
    
    if not model_path.exists():
        available_models = []
        for p in MODELS_DIR.glob("*.joblib"):
            available_models.append(p.stem)
        for d in MODELS_DIR.iterdir():
            if d.is_dir() and (d / "model.joblib").exists():
                available_models.append(d.name)
                
        error_msg = f'Model artifact not found for "{model_name}". '
        if available_models:
            error_msg += f'Available models: {", ".join(sorted(available_models))}'
        else:
            error_msg += f'No models found in {MODELS_DIR}'
            
        raise FileNotFoundError(error_msg)
    
    return joblib.load(model_path)
