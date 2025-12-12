# Scripts

## Overview

Utility scripts for training, demos, evaluation, and API serving.

## Commands

### Training

```bash
# Full pipeline
uv run intent-train --config config/dataset/clinc150/tiny.yaml

# Individual steps
uv run python scripts/pipeline/00_data_loading.py
uv run python scripts/pipeline/01_exploratory_analysis.py
uv run python scripts/pipeline/02_build_embeddings.py
uv run python scripts/pipeline/03_model_training.py
uv run python scripts/pipeline/04_model_prediction.py
uv run python scripts/pipeline/05_model_evaluation.py
```

### Classification

```bash
uv run intent-classify --model-path output/experiment/models/Linear\ SVM/ --text "your text here"
```

### API Server

```bash
# Install API dependencies
uv sync --extra api

# Start server
CONFIG_FILE=config/dataset/clinc150/tiny.yaml uv run api-serve --host 0.0.0.0 --port 8000
```

### Demos

```bash
# Intent classifier demo
uv run python scripts/demos/intent_classifier_demo.py

# Semantic search demo
uv run python scripts/demos/semantic_search_demo.py

# Intent trend analyzer
uv run python scripts/demos/intent_trend_analyzer.py
```

### Hyperparameter Tuning

```bash
uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all
```

## Directories

- **api/**: FastAPI implementation
- **demos/**: Interactive Gradio demos
- **pipeline/**: Training pipeline scripts

See individual README files in each directory for details.
