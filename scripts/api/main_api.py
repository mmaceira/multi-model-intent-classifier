import logging
import os
import uuid
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Import unified model loader
from intent_classifier.utils.model_loader import load_persisted_model
from intent_classifier.utils.paths import get_embeddings_dir, get_models_dir
from intent_classifier.utils.seed import set_global_seed

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
    description="HTTP API for multi-backend intent classification.",
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

    # Get client identifier (IP address)
    client_id = request.client.host if request.client else "unknown"

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


# Prometheus metrics (if available)
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed. Metrics endpoint disabled.")


# Request/Response models
class PredictRequest(BaseModel):  # pylint: disable=missing-class-docstring
    model_id: str = Field(..., description="Model identifier")
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and clean input text."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class PredictResponse(BaseModel):  # pylint: disable=missing-class-docstring
    label: str = Field(
        ..., description="Predicted class label (or '__ABSTAIN__' if confidence too low)"
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: dict[str, float] | None = Field(
        None, description="All class probabilities (if available)"
    )
    abstained: bool = Field(
        default=False, description="Whether the model abstained from prediction"
    )
    request_id: str | None = Field(None, description="Request ID for tracing")


class ModelInfo(BaseModel):  # pylint: disable=missing-class-docstring
    name: str
    type: str
    path: str
    description: str | None = None


class ModelDetailResponse(BaseModel):  # pylint: disable=missing-class-docstring
    model_id: str
    name: str
    type: str
    embedding_backend: str | None = None
    training_timestamp: str | None = None
    vectorizer_id: str | None = None


class HealthResponse(BaseModel):  # pylint: disable=missing-class-docstring
    status: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):  # pylint: disable=missing-class-docstring
    ready: bool
    models_loaded: int


# Model cache with LRU eviction
# Maximum number of models to keep in cache (default: 10)
MAX_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "10"))
_model_cache: OrderedDict[str, Any] = OrderedDict()


# Determine experiment name from config or environment variable
def _get_experiment_name() -> str | None:
    """Get experiment name from CONFIG_FILE or EXPERIMENT_NAME env var."""
    experiment_name = os.getenv("EXPERIMENT_NAME")
    if experiment_name:
        return experiment_name

    # Try to read from config file
    config_file = os.getenv("CONFIG_FILE")
    if not config_file:
        return None

    try:
        from intent_classifier.utils.config_loader import get_run_name_from_config

        run_name = get_run_name_from_config(config_file)
        if run_name:
            from intent_classifier.utils.paths import get_config_path

            config_path = get_config_path(config_file)
            logger.info(f"Loaded experiment name '{run_name}' from config file: {config_path}")
        else:
            logger.warning(
                f"Config file {config_file} does not contain 'general.run_name'. "
                f"Using default models directory."
            )
        return run_name
    except Exception as e:
        logger.warning(
            f"Failed to read config file '{config_file}': {e}. "
            f"Using default models directory. "
            f"Set EXPERIMENT_NAME or ensure CONFIG_FILE points to a valid config file."
        )
        return None


# Get models and embeddings directories
# If EXPERIMENT_NAME or CONFIG_FILE is set, use that experiment's directory
# Otherwise, default to root models/ directory
experiment_name = _get_experiment_name()
MODELS_DIR = get_models_dir(experiment_name) if experiment_name else get_models_dir()
EMBEDDINGS_DIR = get_embeddings_dir(experiment_name) if experiment_name else get_embeddings_dir()

logger.info(f"Using models directory: {MODELS_DIR}")
logger.info(f"Using embeddings directory: {EMBEDDINGS_DIR}")

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
    """Get model from cache or load it with LRU eviction."""
    if model_identifier not in _model_cache:
        logger.info(f"Loading model: {model_identifier}")
        # Evict oldest entry if cache is full
        if len(_model_cache) >= MAX_CACHE_SIZE:
            oldest_key, _ = _model_cache.popitem(last=False)
            logger.info(f"Evicting model from cache: {oldest_key}")

        _model_cache[model_identifier] = load_persisted_model(
            model_identifier, models_dir=MODELS_DIR, embeddings_dir=EMBEDDINGS_DIR
        )
    else:
        # Move to end (most recently used)
        _model_cache.move_to_end(model_identifier)

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


@app.get("/v1/models", response_model=list[ModelInfo])
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
    if not req.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_id must be provided",
        )
    model_identifier = req.model_id

    request_id = getattr(request.state, "request_id", None)

    # Get minimum confidence threshold for abstention
    min_conf = float(os.getenv("MIN_CONFIDENCE", "0.6"))

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
            abstained = False

            if hasattr(clf, "predict_proba"):
                try:
                    probas = clf.predict_proba([req.text])[0]
                    pred_idx = int(probas.argmax())
                    conf = float(probas[pred_idx])
                    # Get class names if available
                    if hasattr(clf, "classes_"):
                        label_map = {i: str(cls) for i, cls in enumerate(clf.classes_)}
                        probabilities = {
                            str(cls): float(prob)
                            for cls, prob in zip(clf.classes_, probas, strict=False)
                        }
                        # Check abstention threshold
                        if conf < min_conf:
                            label = "__ABSTAIN__"
                            abstained = True
                        else:
                            label = label_map[pred_idx]
                    else:
                        # Fallback if no classes_ attribute
                        if conf < min_conf:
                            label = "__ABSTAIN__"
                            abstained = True
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

        else:
            # Handle regular classifiers
            label = model.predict([req.text])[0]
            conf = None
            probabilities = None
            abstained = False

            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba([req.text])[0]
                    pred_idx = int(probas.argmax())
                    conf = float(probas[pred_idx])
                    # Get class names if available
                    if hasattr(model, "classes_"):
                        label_map = {i: str(cls) for i, cls in enumerate(model.classes_)}
                        probabilities = {
                            str(cls): float(prob)
                            for cls, prob in zip(model.classes_, probas, strict=False)
                        }
                        # Check abstention threshold
                        if conf < min_conf:
                            label = "__ABSTAIN__"
                            abstained = True
                        else:
                            label = label_map[pred_idx]
                    else:
                        # Fallback if no classes_ attribute
                        if conf < min_conf:
                            label = "__ABSTAIN__"
                            abstained = True
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

        return PredictResponse(
            label=str(label),
            confidence=conf,
            probabilities=probabilities,
            abstained=abstained,
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
            detail=f"Prediction failed: {e!s}",
        ) from e


def cli():
    """CLI entry point for the API server (for console script)."""
    import argparse

    parser = argparse.ArgumentParser(description="Start the intent classification API server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info").lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    logger.info(f"Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )


def main():
    """Main entry point for the API server (backward compatibility)."""
    cli()


if __name__ == "__main__":
    main()
