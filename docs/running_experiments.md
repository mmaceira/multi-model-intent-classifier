# Running Experiments

## Overview

How to run the training pipeline. For background on data splits and algorithms, see `pipeline.md` and `experiments.md`.

## Quickstart

```bash
# Install dependencies
uv sync --extra all

# Run full pipeline (single-label)
uv run python scripts/pipeline/run_all.py --config config/dataset/clinc150/tiny.yaml

# Run full pipeline (multi-label)
uv run python scripts/pipeline/run_all.py --config config/dataset/nlu_plus/tiny.yaml
```

Datasets download automatically from HuggingFace/GitHub.

## Commands

### Full Pipeline

```bash
# Using entry point
uv run intent-train --config config/dataset/clinc150/tiny.yaml

# Or directly
uv run python scripts/pipeline/run_all.py --config config/dataset/clinc150/tiny.yaml
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

## Configuration

Preconfigured configs:
- `config/dataset/clinc150/default.yaml` - Full CLINC150
- `config/dataset/clinc150/tiny.yaml` - Quick test
- `config/dataset/nlu_plus/default.yaml` - Full NLU++
- `config/dataset/nlu_plus/tiny.yaml` - Quick test

```bash
uv run python scripts/pipeline/run_all.py --config config/dataset/clinc150/default.yaml
```

## Outputs

All outputs in `output/{run_name}/`:
- `data_exploration/` - Statistics and visualizations
- `embeddings/` - FAISS indices
- `models/` - Trained models
- `predictions/` - Predictions
- `results/` - Evaluation metrics

See `configuration.md` for model selection and `hyperparameter_tuning.md` for tuning.
