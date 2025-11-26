# Installation

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Internet connection (for initial dataset download from HuggingFace)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mmaceira/multi-model-intent-classifier.git
cd multi-model-intent-classifier

# 2. Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install dependencies (uv will automatically create a virtual environment)
uv sync --extra all
```

## Optional Dependencies

The project has several optional dependency groups for different features:

| Extra | Includes | Install Command |
|-------|----------|----------------|
| **all** | Everything (recommended for full pipeline) | `uv sync --extra all` |
| **api** | FastAPI server dependencies | `uv sync --extra api` |
| **ui** | Gradio UI dependencies | `uv sync --extra ui` |
| **dev** | Development tools (pytest, ruff, black, pre-commit) | `uv sync --extra dev` |

**Note**: The `pipeline` extra is required to run the full training pipeline (scripts in `scripts/pipeline/`). The `all` extra includes everything and is recommended for most users.

**Note**: All dependencies are listed in `pyproject.toml`. The base installation includes most dependencies. For OpenAI features, ensure `OPENAI_API_KEY` is set in your environment or `.env` file.

## Development Setup

1. Install development dependencies:
   ```bash
   uv sync --extra dev
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
