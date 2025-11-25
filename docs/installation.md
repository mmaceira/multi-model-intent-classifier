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
# For full pipeline with all features (recommended):
uv sync --extra all

# For minimal installation (core only):
uv sync

# For just the pipeline (includes matplotlib, seaborn, dataframe_image):
uv sync --extra pipeline

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
# For full pipeline with all features (recommended):
pip install -e ".[all]"

# For minimal installation (core only):
pip install -e .

# For just the pipeline (includes matplotlib, seaborn, dataframe_image):
pip install -e ".[pipeline]"

# 4. Set up Ollama (required for RAG-LLM models)
# Make sure Ollama is installed and running:
#   ollama serve
#   ollama pull llama3.1:8b

# 5. (Optional) Set up OpenAI API key if you want to use OpenAI models
# export OPENAI_API_KEY="your-api-key-here"
# Note: OpenAI models are disabled by default - the pipeline works with Ollama only
```

## Optional Dependencies

The project has several optional dependency groups for different features:

| Extra | Includes | Install Command |
|-------|----------|----------------|
| **all** | Everything (recommended for full pipeline) | `uv sync --extra all` or `pip install -e ".[all]"` |
| **pipeline** | Core pipeline dependencies (matplotlib, seaborn, dataframe_image) | `uv sync --extra pipeline` or `pip install -e ".[pipeline]"` |
| **api** | FastAPI server dependencies | `uv sync --extra api` or `pip install -e ".[api]"` |
| **demo** | Gradio demos and visualization tools | `uv sync --extra demo` or `pip install -e ".[demo]"` |
| **viz** | Visualization tools (matplotlib, seaborn, umap) | `uv sync --extra viz` or `pip install -e ".[viz]"` |
| **tune** | Ray Tune for hyperparameter optimization | `uv sync --extra tune` or `pip install -e ".[tune]"` |
| **mlflow** | MLflow tracking | `uv sync --extra mlflow` or `pip install -e ".[mlflow]"` |
| **dev** | Development tools (pytest, ruff, black, pre-commit) | `uv sync --extra dev` or `pip install -e ".[dev]"` |

**Note**: The `pipeline` extra is required to run the full training pipeline (scripts in `scripts/pipeline/`). The `all` extra includes everything and is recommended for most users.

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
