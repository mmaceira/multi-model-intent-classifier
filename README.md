# Reuters News Topic Classification & RAG Framework

This repository provides a **production-ready NLP pipeline** for topic classification and semantic retrieval over the Reuters-21578 news corpus. It supports both classic and modern approaches, including:

- Bag-of-words and transformer-based classifiers
- Retrieval-Augmented Generation (RAG) with fast vector search
- LLM-augmented classification (OpenAI-compatible)
- HTTP API for real-time inference
- Robust evaluation and EDA utilities

---

## Features

- **Classic Text Classification**: Naive Bayes, Linear SVM, and MiniLM+Logistic Regression baselines
- **RAG Classifiers**: kNN-majority, centroid-NN, and LLM-augmented topic classification
- **Fast Vector Search**: FAISS-backed semantic retrieval
- **API Service**: FastAPI microservice for HTTP classification
- **Evaluation Suite**: Macro-F1, accuracy, confusion matrix, ROC, error analysis, and significance testing
- **Exploratory Data Analysis**: Class distribution, length statistics, and imbalance visualization
- **Semantic Search**: Simple cosine similarity search over embeddings
- **Extensible & Modular**: Easy to add new models, retrievers, or datasets

---

## Project Structure

```
├── src/
│   ├── rag/              # RAG classifiers, vector store, retrieval, adapters
│   ├── api/              # FastAPI microservice
│   ├── datasets/         # Reuters dataset loading utilities
│   ├── algorithms/       # Classic classifiers (Naive Bayes, SVM, Transformer+LogReg)
│   ├── evaluation.py     # Evaluation and reporting utilities
│   ├── exploration.py    # EDA helpers
│   ├── model.py          # Base TextClassifier class
│   ├── search.py         # Semantic search helper
│   └── ...
├── artifacts/            # Vector index and metadata (auto-generated)
├── results/              # Evaluation outputs (reports, confusion, ROC, errors)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd nltk-reuters
   ```
2. **Create a Python 3.10+ virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Build the Vector Index

Build a FAISS vector index and metadata for RAG classifiers:
```bash
python -m src.rag.build_index
```
- Default: 10 classes, cutoff year 1996, MiniLM embeddings
- Artifacts: `artifacts/index.faiss`, `artifacts/meta.jsonl`
- For custom options:
  ```bash
  python -m src.rag.build_index --help
  ```

### 2. Run the API Service

Start the FastAPI microservice:
```bash
uvicorn src.api.main:app --reload --port 8007
```

### 3. Classify via HTTP API

Send a classification request:
```bash
curl -X POST http://localhost:8007/classify \
     -H "Content-Type: application/json" \
     -d '{"doc": "Sample article text", "model": "km", "top_k": 5}'
```
- `model`: "km" (kNN-majority), "centroid" (centroid-NN), or "llm" (LLM-augmented)
- `top_k`: Number of context docs to retrieve (for RAG/LLM)

### 4. Python API (Batch or Interactive)

```python
from src.rag import load_kmajority, load_centroid, load_llm

clf = load_llm(top_k=5, model="gpt-4o-mini")
labels = clf.predict(["Sample news article text."])
```

---

## Classic Baselines

Train and evaluate classic models (Naive Bayes, SVM, Transformer+LogReg):

```python
from src.datasets.dataset import load_data
from src.algorithms.naive_bayes import NaiveBayesClassifier
from src.algorithms.linear_svm import LinearSVMClassifier
from src.algorithms.transformer_logreg import TransformerLogReg
from src.evaluation import run_evaluations

X_train, y_train, X_test, y_test, label_names = load_data(n_classes=10)
models = {
    "Naive Bayes": NaiveBayesClassifier(),
    "Linear SVM": LinearSVMClassifier(),
    "MiniLM+LogReg": TransformerLogReg()
}
results, _ = run_evaluations(models, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, label_names=label_names)
```

---

## Evaluation & Reporting

- **Metrics:** Macro-F1, accuracy, confusion matrix, ROC curves
- **Artifacts:** CSV reports, PNG confusion matrices, ROC, error lists
- **Significance Testing:** Bootstrap test between top-2 models
- **EDA:** Class frequency, length distribution, imbalance plots

All outputs are saved in the `results/` directory.

---

## Configuration

- **Indexing:** `src/rag/build_index.py` supports custom embedding models, class count, cutoff year, and input CSVs
- **API:** Configure port, model, and top_k via HTTP request
- **LLM:** Requires `OPENAI_API_KEY` in environment for LLM-augmented classification

---

## Extending & Customization

- **Add new classifiers:** Implement a new class inheriting from `TextClassifier` or `RagClassifierBase`
- **Plug in new retrievers or embedding models** in `src/rag/vector_store.py` and `src/rag/retrieval.py`
- **Custom datasets:** Adapt `src/datasets/dataset.py` or provide CSVs to `build_index.py`
- **API endpoints:** Extend `src/api/main.py` for new routes or batch processing

---

## Troubleshooting

- **Missing artifacts:** Run `python -m src.rag.build_index` before using RAG or API
- **LLM errors:** Ensure `OPENAI_API_KEY` is set and you have API access
- **Dependency issues:** Check Python version (3.10+ recommended) and reinstall with `pip install -r requirements.txt`
- **Evaluation errors:** Ensure all models implement `fit` and `predict` as per `TextClassifier` interface

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Reuters-21578 corpus via NLTK
- FAISS, Sentence-Transformers, OpenAI, scikit-learn
- Inspired by recent advances in RAG and hybrid IR/NLP pipelines

