# API Server

## Overview

HTTP API for serving trained intent classification models using FastAPI.
The `api-serve` entry point loads models from `output/` and exposes health, readiness,
metrics (if enabled), and prediction endpoints.

## Quickstart

```bash
# Install API dependencies once
uv sync --extra api

# Start server (single-label CLINC150 tiny)
CONFIG_FILE=config/dataset/clinc150/tiny.yaml \
  uv run api-serve --host 0.0.0.0 --port 8000

# Start server (multi-label NLU+ tiny)
CONFIG_FILE=config/dataset/nlu_plus/tiny.yaml \
  uv run api-serve --host 0.0.0.0 --port 8000
```

The server:

- Loads models from `output/{label_type}/{dataset_name}/{config_name}/models/`
- Exposes OpenAPI docs at `http://localhost:8000/docs`
- Enables CORS and simple in-memory rate limiting by default

## Endpoints

- `GET /health` – basic health check
- `GET /ready` – readiness check (includes number of loaded models)
- `GET /v1/models` – list available models
- `GET /v1/models/{model_id}` – model details
- `POST /v1/predict` – classify a single text with a selected model

### Example Requests

```bash
# Health
curl http://localhost:8000/health

# Ready
curl http://localhost:8000/ready

# List models
curl http://localhost:8000/v1/models

# Predict with a specific model
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Linear SVM", "text": "what is my account balance?"}'
```

## Configuration

Key environment variables:

- `CONFIG_FILE` – dataset/config YAML used to locate outputs
- `SEED` – random seed (default: 42)
- `CORS_ORIGINS` – comma-separated list of allowed origins (default: `*`)
- `RATE_LIMIT_REQUESTS` – max requests per client per window (default: `100`)
- `RATE_LIMIT_WINDOW` – rate limit window in seconds (default: `60`)

For LLM-backed models (RAG-LLM or other providers), also see:

- `llm_providers.md` – provider setup and `config/llm_config.yaml`
- `hyperparameter_tuning.md` – how tuned models are selected for serving
