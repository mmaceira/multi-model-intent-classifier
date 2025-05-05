from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .model_loader import get_model, MODELS_DIR, MODELS_INFO
from pathlib import Path

app = FastAPI(
    title="Text Classification API",
    description="Send any text and have it classified by a chosen model.",
    version="0.1.0",
)

class PredictRequest(BaseModel):
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: Optional[float] = None

class ModelInfo(BaseModel):
    name: str
    type: str
    path: str

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    models = []
    
    # Add direct .joblib files
    for p in Path(MODELS_DIR).glob("*.joblib"):
        models.append(ModelInfo(
            name=p.stem,
            type="file",
            path=str(p.relative_to(MODELS_DIR))
        ))
    
    # Add directories containing model.joblib
    for dir_path in Path(MODELS_DIR).iterdir():
        if dir_path.is_dir() and (dir_path / "model.joblib").exists():
            models.append(ModelInfo(
                name=dir_path.name,
                type="directory",
                path=f"{dir_path.name}/model.joblib"
            ))
    
    return sorted(models, key=lambda x: x.name.lower())

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Use model_id if provided, otherwise fall back to model_name
    model_identifier = req.model_id or req.model_name
    if not model_identifier:
        raise HTTPException(status_code=400, detail="Either model_id or model_name must be provided")
    
    # Get model info
    model_info = None
    for mid, info in MODELS_INFO.items():
        if mid == model_identifier or info['dir'] == model_identifier:
            model_info = info
            break
    
    if model_info is None:
        raise HTTPException(status_code=400, detail=f"Unknown model type for {model_identifier}")
    
    model = get_model(model_identifier)
    
    try:
        if model_info['type'] == 'rag':
            # Handle RAG models
            if not model['model']:
                raise HTTPException(status_code=400, detail="RAG model not properly initialized")
            
            label = model['model'].predict([req.text])[0]
            conf = None
            if hasattr(model['model'], "predict_proba"):
                try:
                    conf = float(max(model['model'].predict_proba([req.text])[0]))
                except Exception:
                    conf = None
        else:
            # Handle regular classifiers
            label = model.predict([req.text])[0]
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    conf = float(max(model.predict_proba([req.text])[0]))
                except Exception:
                    conf = None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictResponse(label=label, confidence=conf)
