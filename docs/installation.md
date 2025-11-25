# Installation

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Virtual environment (recommended)
- Internet connection (for initial dataset download from HuggingFace)

## Installation Methods

### Using uv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/mmaceira/multi-model-intent-classifier.git
cd multi-model-intent-classifier

# 2. Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install dependencies (uv will automatically create a virtual environment)
uv sync

# 4. Activate the virtual environment (if needed)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 5. Set up Ollama (required for RAG-LLM models)
# Make sure Ollama is installed and running:
#   ollama serve
#   ollama pull llama3.1:8b

# 6. (Optional) Set up OpenAI API key if you want to use OpenAI models
# export OPENAI_API_KEY="your-api-key-here"
# Note: OpenAI models are disabled by default - the pipeline works with Ollama only
```

### Using pip (Alternative)

```bash
# 1. Clone the repository
git clone https://github.com/mmaceira/multi-model-intent-classifier.git
cd multi-model-intent-classifier

# 2. Create and activate virtual environment
python -m venv venv/multi-model-intent-classifier
source venv/multi-model-intent-classifier/bin/activate  # On Windows: venv\multi-model-intent-classifier\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Set up Ollama (required for RAG-LLM models)
# Make sure Ollama is installed and running:
#   ollama serve
#   ollama pull llama3.1:8b

# 5. (Optional) Set up OpenAI API key if you want to use OpenAI models
# export OPENAI_API_KEY="your-api-key-here"
# Note: OpenAI models are disabled by default - the pipeline works with Ollama only
```

## External Dependencies

The project has several optional dependencies for different features:

| Feature | Required Packages | Notes |
|---------|------------------|-------|
| **Base training & evaluation** | `scikit-learn`, `numpy`, `pandas`, `datasets` | Core dependencies for all models |
| **RAG + FAISS** | `faiss-cpu` | Required for RAG models (CentroidNN, k-Majority, LLM) |
| **OpenAI / LLM mode** | `litellm`, `openai` | Requires `OPENAI_API_KEY` environment variable |
| **API server** | `uvicorn`, `fastapi` | For running the REST API (`scripts/api/main_api.py`) |
| **SBERT embeddings** | `sentence-transformers` | Default embedding backend |
| **Testing** | `pytest`, `datasets` | Required for running test suite |
| **MLflow tracking** | `mlflow>=2.0.0` | Optional - install with `pip install -e ".[mlflow]"` or `uv sync --extra mlflow` |
| **API server** | `fastapi`, `uvicorn` | Install with `pip install -e ".[api]"` or `uv sync --extra api` |
| **Visualization** | `matplotlib`, `seaborn` | Install with `pip install -e ".[viz]"` or `uv sync --extra viz` |
| **Hyperparameter tuning** | `ray[tune]` | Install with `pip install -e ".[tune]"` or `uv sync --extra tune` |

**Note**: All dependencies are listed in `pyproject.toml`. The base installation includes most dependencies. For OpenAI features, ensure `OPENAI_API_KEY` is set in your environment or `.env` file.

## Development Setup

1. Install development dependencies (using uv):
   ```bash
   uv sync --extra dev
   ```
   Or using pip:
   ```bash
   pip install -e ".[dev]"
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Verification

After installation, verify everything works:

```bash
# Run the complete pipeline
python scripts/pipeline/run_all.py

# Or run a quick test
pytest tests/test_dataset_clinc150.py
```
