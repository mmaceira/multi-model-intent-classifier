import json
from pathlib import Path

import requests

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


def get_available_models():
    """Get list of available models from the output directory."""
    models_dir = Path("output/experiment_with_03_classes/models")
    available_models = {}

    for model_id, info in MODELS_INFO.items():
        model_dir = models_dir / info["dir"]
        if model_dir.exists():
            # Check for model files
            for ext in ("model.joblib", "model.pkl", "classifier.joblib", "classifier.pkl"):
                if (model_dir / ext).exists():
                    available_models[model_id] = info
                    break

    return available_models


def run_model_test(model_id, text):
    """Send a test request to a specific model."""
    url = "http://127.0.0.1:8000/v1/predict"
    headers = {"Content-Type": "application/json"}
    data = {
        "model_id": model_id,  # Changed from model_name to model_id
        "text": text,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    # Test text
    test_text = "A barrel of crude futures climbed to $75 following OPEC meeting."

    # Get all available models
    models = get_available_models()
    print(f"Found {len(models)} models to test:")

    # Test each model
    for model_id, info in models.items():
        print(f"\nTesting model: {info['name']} ({model_id})")
        result = run_model_test(model_id, test_text)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("Response:")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
