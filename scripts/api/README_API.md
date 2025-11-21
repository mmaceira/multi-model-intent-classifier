# FastAPI Text Classification Service

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the API
python scripts/api/manual_test_api.py

# 3. Run the server
MODELS_DIR=output/experiment_with_03_classes/models uvicorn scripts.api.main_api:app --reload
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
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "linear_svm", "text": "A barrel of crude futures climbed to $75 following OPEC meeting."}'
```

### Test the API
```bash
# Run the manual test script
python scripts/api/manual_test_api.py

# Or test manually
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "rag_knn", "text": "The Federal Reserve announced interest rates will remain unchanged this quarter."}'
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

## ⚠️ Requirements

1. OpenAI API key in `.env` file
2. Model files in the specified `MODELS_DIR`
3. Python 3.8+

## 🔍 Troubleshooting

- If you get a "Model not found" error, check the `/models` endpoint to see available models
- For JSON formatting issues, ensure proper escaping of special characters
- For RAG models, verify your OpenAI API key is set in the `.env` file
