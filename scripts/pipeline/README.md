# Training Pipeline

## Overview

Python scripts for the complete training and evaluation pipeline.

## Commands

### Full Pipeline

```bash
uv run python scripts/pipeline/run_all.py --config config/experiments/clinc150/tiny.yaml
```

### Individual Steps

```bash
uv run python scripts/pipeline/00_data_loading.py
uv run python scripts/pipeline/01_exploratory_analysis.py
uv run python scripts/pipeline/02_build_embeddings.py
uv run python scripts/pipeline/03_model_training.py
uv run python scripts/pipeline/04_model_prediction.py
uv run python scripts/pipeline/05_model_evaluation.py
```

## Steps

### 00_data_loading.py
- Load and validate dataset
- Outputs: Processed dataset, statistics

### 01_exploratory_analysis.py
- Analyze dataset characteristics
- Outputs: Visualizations, vocabulary analysis

### 02_build_embeddings.py
- Generate embeddings (SBERT default, OpenAI optional)
- Outputs: FAISS indices and metadata

### 03_model_training.py
- Train multiple classification models
- Outputs: Trained model files

### 04_model_prediction.py
- Generate predictions on test set
- Outputs: Predictions with probabilities

### 05_model_evaluation.py
- Evaluate model performance
- Outputs: Metrics, confusion matrices, comparison plots

## Prerequisites

1. **Install dependencies**: `uv sync --extra all`

2. **Configuration**: Set up config file (e.g., `config/experiments/clinc150/tiny.yaml`)

3. **Ollama** (for RAG-LLM models):
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ```

4. **OpenAI** (optional):
   - Set `OPENAI_API_KEY` environment variable
   - Enable in `config/algorithm/models_config.yaml`

## Outputs

All outputs in `output/{run_name}/`:
- `data_exploration/` - Statistics and visualizations
- `embeddings/` - FAISS indices
- `models/` - Trained models
- `predictions/` - Predictions
- `results/` - Evaluation metrics

See [Pipeline Documentation](../../docs/pipeline.md) for detailed information.
