from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .model_loader import get_model, MODELS_DIR
from pathlib import Path

app = FastAPI(
    title="Text Classification API",
    description="Send any text and have it classified by a chosen model.",
    version="0.1.0",
)

class PredictRequest(BaseModel):
    model_name: str
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: Optional[float] = None

class ModelInfo(BaseModel):
    name: str
    type: str  # 'file' or 'directory'
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
    model = get_model(req.model_name)
    try:
        label = model.predict([req.text])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    conf = None
    if hasattr(model, "predict_proba"):
        try:
            conf = float(max(model.predict_proba([req.text])[0]))
        except Exception:
            conf = None

    return PredictResponse(label=label, confidence=conf)
