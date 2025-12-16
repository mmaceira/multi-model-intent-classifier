# Running Experiments

## Overview

How to run the training pipeline. For background on data splits and pipeline steps, see `pipeline.md`. For dataset details, see `experiments.md`.

## Quickstart

```bash
# Install dependencies
uv sync --extra all

# Run full pipeline (single-label)
DATASET=clinc150 VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py

# Run full pipeline (multi-label)
DATASET=nlu_plus VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py
```

Datasets download automatically from HuggingFace/GitHub.

## Commands

### Full Pipeline

```bash
# Using entry point
DATASET=clinc150 VARIANT=tiny \
  uv run intent-train

# Or directly
DATASET=clinc150 VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py
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
- `config/experiments/clinc150/default.yaml` - Full CLINC150
- `config/experiments/clinc150/tiny.yaml` - Quick test
- `config/experiments/nlu_plus/default.yaml` - Full NLU++
- `config/experiments/nlu_plus/tiny.yaml` - Quick test

```bash
DATASET=clinc150 VARIANT=default \
  uv run python scripts/pipeline/run_all.py
```

## Outputs

All outputs live under `output/runs/<label_type>/<dataset>/<variant>/`:
- `dataset/` - Statistics, label summaries, and dataset metadata
- `features/` - Embeddings and other feature artefacts
- `models/` - Trained models
- `eval/` - Per‑model evaluation artefacts
- `compare/` - Cross‑model summaries and comparison plots
- `meta/` - Reproducibility package (resolved config, env, git info, manifest)

See `output_schema.md` for the full layout and `config.md` for config layering and selection.
