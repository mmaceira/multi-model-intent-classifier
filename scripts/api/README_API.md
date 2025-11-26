# FastAPI Text Classification Service

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync --extra api

# 2. Test the API
uv run python scripts/api/manual_test_api.py

# 3. Run the server
uv run uvicorn scripts.api.main_api:app --reload
```

## 📋 API Endpoints

- Swagger UI: http://127.0.0.1:8000/docs
- List models: GET http://127.0.0.1:8000/models
- Predict: POST http://127.0.0.1:8000/predict

## 🔧 Usage Examples

### List Available Models
```bash
curl -X GET http://127.0.0.1:8000/models
```

### Make a Prediction
```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "linear_svm", "text": "A barrel of crude futures climbed to $75 following OPEC meeting."}'
```

**Response format:**
```json
{
  "label": "transfer_money",
  "confidence": 0.95,
  "probabilities": {
    "transfer_money": 0.95,
    "check_balance": 0.03,
    "greeting": 0.02
  },
  "abstained": false,
  "request_id": "uuid-here"
}
```

**Abstention:** If the model's confidence is below the threshold (default: 0.6, configurable via `MIN_CONFIDENCE` env var), the response will have:
- `label: "__ABSTAIN__"`
- `abstained: true`
- `confidence: <threshold>`

This is useful for production systems where you want to reject low-confidence predictions.

### Test the API
```bash
# Run the manual test script
python scripts/api/manual_test_api.py

# Or test manually
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "rag_kmajority", "text": "The Federal Reserve announced interest rates will remain unchanged this quarter."}'
```

## 📝 Model Organization

Models can be organized in two ways:

1. Direct `.joblib` files:
```
models/
    linear_svm.joblib
    rag_knn.joblib
```

2. Subdirectories with `model.joblib`:
```
models/
    Naive Bayes/
        model.joblib
    Random Forest/
        model.joblib
```

## ⚙️ Configuration

### Environment Variables

- `MIN_CONFIDENCE` (default: 0.6): Minimum confidence threshold for predictions. If a prediction's confidence is below this threshold, the API will return `"__ABSTAIN__"` as the label.
- `API_KEY`: Optional API key for authentication. If set, all requests must include `X-API-Key` header.
- `RATE_LIMIT_REQUESTS` (default: 100): Maximum number of requests per window.
- `RATE_LIMIT_WINDOW` (default: 60): Time window in seconds for rate limiting.
- `CORS_ORIGINS`: Comma-separated list of allowed CORS origins (default: "*").
- `MLFLOW_TRACKING_URI`: Optional MLflow tracking URI for experiment logging.

### Example with Abstention Threshold

```bash
# Set minimum confidence to 0.7 (more conservative)
export MIN_CONFIDENCE=0.7
uvicorn scripts.api.main_api:app --reload
```

## ⚠️ Requirements

1. Model files in the specified `MODELS_DIR`
2. Python 3.12+
3. **OpenAI API key is required only if you use OpenAI embeddings or OpenAI-backed RAG models**
   - Naive Bayes, Linear SVM, and other non-RAG models don't need it
   - RAG models using local embeddings (SBERT) don't need it
   - RAG-LLM with `ollama/...` models don't need it
   - Only RAG models using OpenAI embeddings require the API key

## 🔍 Troubleshooting

- If you get a "Model not found" error, check the `/models` endpoint to see available models
- For JSON formatting issues, ensure proper escaping of special characters
- For RAG models, verify your OpenAI API key is set in the `.env` file
