# FastAPI Service

## Overview

FastAPI text classification service.

## Quickstart

```bash
# Install dependencies
uv sync --extra api

# Start server
CONFIG_FILE=config/dataset/clinc150/tiny.yaml uv run api-serve --host 0.0.0.0 --port 8000
```

## Endpoints

- Swagger UI: http://127.0.0.1:8000/docs
- List models: GET http://127.0.0.1:8000/models
- Predict: POST http://127.0.0.1:8000/v1/predict

## Usage

### List Models

```bash
curl -X GET http://127.0.0.1:8000/models
```

### Make Prediction

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "linear_svm", "text": "your text here"}'
```

**Response:**
```json
{
  "label": "intent_name",
  "confidence": 0.95,
  "probabilities": {...},
  "abstained": false
}
```

**Abstention**: If confidence is below threshold (default: 0.6), returns `"__ABSTAIN__"` label.

## Configuration

Environment variables:
- `MIN_CONFIDENCE` (default: 0.6) - Minimum confidence threshold
- `RATE_LIMIT_REQUESTS` (default: 100) - Max requests per window
- `RATE_LIMIT_WINDOW` (default: 60) - Time window in seconds
- `CORS_ORIGINS` - Allowed CORS origins

## Requirements

- Model files in `MODELS_DIR` (from config)
- Python 3.12+
- OpenAI API key only if using OpenAI embeddings/models
