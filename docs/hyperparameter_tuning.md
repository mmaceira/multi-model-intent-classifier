# Hyperparameter Tuning

## Overview

Hyperparameter tuning script that tunes on validation set, evaluates on test only.

## Quickstart

```bash
# Tune all models
uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all

# Tune specific model
uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --algo nb
```

## Commands

```bash
# Tune all models
uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all

# Increase search samples
uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all --num-samples 50

# Tune specific model
uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --algo svm
```

## What Gets Tuned

- **Naive Bayes**: `alpha` (smoothing)
- **Linear SVM**: `C` (regularization)
- **Transformer LogReg**: `C` for logistic regression
- **Embedding LogReg**: `C` for logistic regression
- **RAG k-majority/LLM**: `top_k` (neighbors)

## Usage

Tuned hyperparameters are loaded automatically by the training pipeline:

1. Run tuning: `uv run python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all`
2. Run training: `uv run python scripts/pipeline/run_all.py`

## Storage

Best hyperparameters saved to:
- `config/algorithm/hyperparameters/{config_name}/best_{model_name}.yaml` (used by training)
- `output/hyperparams_tune/{config_name}/best_{model_name}.yaml` (reference)

Files in `config/algorithm/hyperparameters/` should be committed to git.
