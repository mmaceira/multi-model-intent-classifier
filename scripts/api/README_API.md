## FastAPI text‑classification service

### Install
```bash
pip install -r requirements.txt   # ensure fastapi & uvicorn are present
```

### Prepare models
Serialise each trained estimator with `joblib`:

```
import joblib
joblib.dump(clf, 'models/linear_svm.joblib')
```

Place them in a folder, e.g.:

```
experiment_with_07_classes/models/
    linear_svm.joblib
    rag_knn.joblib
    ...
```

Models can also be organized in subdirectories:

```
experiment_with_07_classes/models/
    linear_svm.joblib
    rag_knn.joblib
    Naive Bayes/
        model.joblib
    Random Forest/
        model.joblib
    ...
```

The API automatically detects both `.joblib` files directly in the models directory and `model.joblib` files inside subdirectories.

### Run server
You can point the service to any models directory by setting the `MODELS_DIR`
environment variable at launch:

```bash
MODELS_DIR=experiment_with_07_classes/models uvicorn src.api.main:app --reload
```

If `MODELS_DIR` is not set, it defaults to the `models/` folder at the repo root.

* Swagger UI: http://127.0.0.1:8000/docs
* List models: GET http://127.0.0.1:8000/models
* Predict: POST http://127.0.0.1:8000/predict
  ```json
  { "model_name": "linear_svm", "text": "A barrel of crude futures..." }
  ```

### Command Line Examples

#### List available models
```bash
curl -X GET http://127.0.0.1:8000/models
```

#### Make a prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "linear_svm", "text": "A barrel of crude futures climbed to $75 following OPEC meeting."}'
```

#### Making a prediction with a different model
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "rag_knn", "text": "The Federal Reserve announced interest rates will remain unchanged this quarter."}'
```

#### Important note about JSON formatting
When passing model names or text that contain special characters, ensure proper JSON escaping. For example, if a model name contains spaces, do not escape with backslashes in the JSON payload:

```bash
# CORRECT:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Naive bayes", "text": "A barrel of crude futures climbed to $75 following OPEC meeting."}'

# INCORRECT - will cause JSON parsing error:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Naive\ bayes", "text": "A barrel of crude futures climbed to $75 following OPEC meeting."}'
```

#### Using models in nested directories
If your model is stored in a subdirectory (e.g., `Naive Bayes/model.joblib`), you can reference it directly by the directory name:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Naive Bayes", "text": "A barrel of crude futures climbed to $75 following OPEC meeting."}'
```

Note: The API will automatically locate and load the `model.joblib` file inside the specified directory.

#### Listing available models
The `/models` endpoint now returns detailed information about available models:

```bash
curl -X GET http://127.0.0.1:8000/models
```

Example response:
```json
[
  {
    "name": "linear_svm",
    "type": "file",
    "path": "linear_svm.joblib"
  },
  {
    "name": "Naive Bayes",
    "type": "directory",
    "path": "Naive Bayes/model.joblib"
  }
]
```

#### Troubleshooting
If you encounter a "Model not found" error, the error message will now list all available models to help you identify the correct model name to use.
