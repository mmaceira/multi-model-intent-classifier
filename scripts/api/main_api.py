import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Import unified model loader
from src.utils.model_loader import load_persisted_model
from src.utils.paths import get_embeddings_dir, get_models_dir
from src.utils.seed import set_global_seed

# Load environment variables from .env file
load_dotenv()

# Set up logging (entry point - this is where we configure logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set global seed for reproducibility
set_global_seed(int(os.getenv("SEED", 42)))

# Initialize FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="Production-ready API for intent classification using multiple model backends.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (simple in-memory implementation)
# For production, use Redis-based rate limiting
_rate_limit_store: dict[str, list[datetime]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    # Skip rate limiting for health/ready endpoints
    if request.url.path in ["/health", "/ready", "/metrics"]:
        return await call_next(request)

    # Get client identifier (IP address or API key)
    client_id = request.client.host if request.client else "unknown"
    api_key = request.headers.get("X-API-Key")
    if api_key:
        client_id = f"api_key:{api_key}"

    # Clean old entries
    now = datetime.now()
    _rate_limit_store[client_id] = [
        ts for ts in _rate_limit_store[client_id] if now - ts < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]

    # Check rate limit
    if len(_rate_limit_store[client_id]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."},
        )

    # Add current request
    _rate_limit_store[client_id].append(now)

    # Add request ID for tracing
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# API Key authentication (optional)
API_KEY = os.getenv("API_KEY")  # Set API_KEY env var to enable auth


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API key authentication middleware."""
    # Skip auth for health/ready/metrics/docs endpoints
    if request.url.path in ["/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    if API_KEY:
        api_key = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization", ""
        ).replace("Bearer ", "")
        if api_key != API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
            )

    return await call_next(request)


# Prometheus metrics (if available)
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed. Metrics endpoint disabled.")


# Request/Response models
class PredictRequest(BaseModel):
    model_name: Optional[str] = Field(None, description="Model name (deprecated, use model_id)")
    model_id: Optional[str] = Field(None, description="Model identifier")
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and clean input text."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Optional[dict[str, float]] = Field(
        None, description="All class probabilities (if available)"
    )
    request_id: Optional[str] = Field(None, description="Request ID for tracing")


class ModelInfo(BaseModel):
    name: str
    type: str
    path: str
    description: Optional[str] = None


class ModelDetailResponse(BaseModel):
    model_id: str
    name: str
    type: str
    embedding_backend: Optional[str] = None
    training_timestamp: Optional[str] = None
    vectorizer_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    ready: bool
    models_loaded: int


# Model cache (LRU-like, but simple dict for now)
_model_cache: dict[str, Any] = {}
MODELS_DIR = get_models_dir()
EMBEDDINGS_DIR = get_embeddings_dir()

# Model info mapping
MODELS_INFO = {
    "naive_bayes": {"name": "Naive Bayes", "dir": "Naive Bayes", "type": "classifier"},
    "linear_svm": {"name": "Linear SVM", "dir": "Linear SVM", "type": "classifier"},
    "tfidf_svm": {"name": "TF-IDF + SVM", "dir": "TF-IDF bigrams + SVM", "type": "classifier"},
    "minilm_logreg": {"name": "MiniLM + LogReg", "dir": "MiniLM + LogReg", "type": "classifier"},
    "rag_centroid": {"name": "RAG CentroidNN", "dir": "RAG-CentroidNN", "type": "rag"},
    "rag_kmajority": {"name": "RAG k-Majority", "dir": "RAG-kMajority", "type": "rag"},
    "rag_llm_local": {
        "name": "RAG LLM (Local)",
        "dir": "RAG-LLM (local-embeddings)",
        "type": "rag",
    },
    "rag_llm_openai": {
        "name": "RAG LLM (OpenAI)",
        "dir": "RAG-LLM (OpenAI-embeddings)",
        "type": "rag",
    },
}


def _get_model(model_identifier: str):
    """Get model from cache or load it."""
    if model_identifier not in _model_cache:
        logger.info(f"Loading model: {model_identifier}")
        _model_cache[model_identifier] = load_persisted_model(
            model_identifier, models_dir=MODELS_DIR, embeddings_dir=EMBEDDINGS_DIR
        )
    return _model_cache[model_identifier]


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness check endpoint."""
    models_loaded = len(_model_cache)
    return ReadyResponse(ready=True, models_loaded=models_loaded)


@app.get("/v1/models", response_model=List[ModelInfo])
def list_models():
    """List all available models."""
    models = []

    # Add direct model files
    for ext in [".joblib", ".pkl"]:
        for p in MODELS_DIR.glob(f"*{ext}"):
            models.append(
                ModelInfo(
                    name=p.stem,
                    type="file",
                    path=str(p.relative_to(MODELS_DIR)),
                    description=f"Model file: {p.name}",
                )
            )

    # Add directories containing model files
    for dir_path in MODELS_DIR.iterdir():
        if dir_path.is_dir():
            for filename in ["model.joblib", "model.pkl"]:
                if (dir_path / filename).exists():
                    models.append(
                        ModelInfo(
                            name=dir_path.name,
                            type="directory",
                            path=f"{dir_path.name}/{filename}",
                            description=f"Model directory: {dir_path.name}",
                        )
                    )
                    break

    return sorted(models, key=lambda x: x.name.lower())


@app.get("/v1/models/{model_id}", response_model=ModelDetailResponse)
def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    model_info = MODELS_INFO.get(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Try to load model to get more info
    try:
        model = _get_model(model_id)
        # Extract info from model if possible
        embedding_backend = None
        if isinstance(model, dict) and "model" in model:
            # RAG model
            embedding_backend = "openai" if "openai" in model_id else "sbert"
        elif hasattr(model, "vectorizer"):
            # Check vectorizer type
            vec = model.vectorizer if hasattr(model, "vectorizer") else None
            if vec and hasattr(vec, "__class__"):
                embedding_backend = vec.__class__.__name__

        return ModelDetailResponse(
            model_id=model_id,
            name=model_info["name"],
            type=model_info["type"],
            embedding_backend=embedding_backend,
            training_timestamp=None,  # Could be extracted from model metadata
            vectorizer_id=None,  # Could be extracted from model metadata
        )
    except Exception as e:
        logger.warning(f"Could not load model info for {model_id}: {e}")
        return ModelDetailResponse(
            model_id=model_id,
            name=model_info["name"],
            type=model_info["type"],
        )


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    """Classify text using a specified model."""
    # Use model_id if provided, otherwise fall back to model_name
    model_identifier = req.model_id or req.model_name
    if not model_identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either model_id or model_name must be provided",
        )

    request_id = getattr(request.state, "request_id", None)

    try:
        model = _get_model(model_identifier)

        # Handle RAG models
        if isinstance(model, dict) and "model" in model:
            if not model["model"]:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="RAG model not properly initialized",
                )

            clf = model["model"]
            label = clf.predict([req.text])[0]
            conf = None
            probabilities = None

            if hasattr(clf, "predict_proba"):
                try:
                    probas = clf.predict_proba([req.text])[0]
                    conf = float(max(probas))
                    # Get class names if available
                    if hasattr(clf, "classes_"):
                        probabilities = {
                            str(cls): float(prob)
                            for cls, prob in zip(clf.classes_, probas, strict=False)
                        }
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

        else:
            # Handle regular classifiers
            label = model.predict([req.text])[0]
            conf = None
            probabilities = None

            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba([req.text])[0]
                    conf = float(max(probas))
                    # Get class names if available
                    if hasattr(model, "classes_"):
                        probabilities = {
                            str(cls): float(prob)
                            for cls, prob in zip(model.classes_, probas, strict=False)
                        }
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

        return PredictResponse(
            label=str(label),
            confidence=conf,
            probabilities=probabilities,
            request_id=request_id,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_identifier} not found: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Prediction failed for {model_identifier}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


# Backward compatibility endpoints
@app.get("/models", response_model=List[ModelInfo])
def list_models_legacy():
    """Legacy endpoint for listing models."""
    return list_models()


@app.post("/predict", response_model=PredictResponse)
async def predict_legacy(req: PredictRequest, request: Request):
    """Legacy endpoint for predictions."""
    return await predict(req, request)


def main():
    """Main entry point for the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    logger.info(f"Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")
    logger.info(f"API key auth: {'enabled' if API_KEY else 'disabled'}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
