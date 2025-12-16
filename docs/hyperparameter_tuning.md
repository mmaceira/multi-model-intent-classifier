# Hyperparameter Tuning

## Overview

Hyperparameter tuning uses Ray Tune for distributed hyperparameter optimization. Tuning runs on the validation set, with evaluation on the test set only (ML best practice).

## Installation

Hyperparameter tuning requires Ray Tune. Install the `tune` extra:

```bash
uv sync --extra tune
```

When running tuning, use `--active` so Ray workers reuse the current env (prevents `ray` missing inside worker venvs):

```bash
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml
```

## Quickstart

```bash
# Tune all models (default behavior)
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml

# Tune all models explicitly
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml --algo all

# Tune specific model
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml --algo nb
```

## Commands

```bash
# Tune all models (default)
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml

# Tune all models with more samples
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml --algo all --num-samples 50

# Tune specific model
uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml --algo svm

# Tune with custom output directory
uv run --active intent-tune --config config/experiments/nlu_plus/tiny.yaml --algo svm --num-samples 50 --output-dir output/custom_tune
```

## What Gets Tuned

- **Naive Bayes**: `alpha` (smoothing)
- **Linear SVM**: `C` (regularization)
- **Transformer LogReg**: `C` for logistic regression
- **Embedding LogReg**: `C` for logistic regression
- **RAG k-majority/LLM**: `top_k` (neighbors)

## Usage

Tuned hyperparameters are loaded automatically by the training pipeline:

1. Install Ray Tune (see Installation above)
2. Run tuning: `uv run --active intent-tune --config config/experiments/clinc150/tiny.yaml` (tunes all models by default)
3. Run training: `CONFIG_FILE=config/experiments/clinc150/tiny.yaml uv run intent-train`

The `--tune` flag on `intent-train` will also run hyperparameter tuning before training:
```bash
CONFIG_FILE=config/experiments/clinc150/tiny.yaml uv run intent-train --tune
```

## Storage

Best hyperparameters are stored in:

- `config/algorithm/hyperparameters/{config_name}/best_{model_name}.yaml` – **used by the training pipeline**
- `output/hyperparams_tune/{config_name}/best_{model_name}.yaml` – reference copy from Ray Tune

For the tiny CLINC150 config:

- `config_name` is `tiny`
- The training pipeline (via `CONFIG_FILE=config/experiments/clinc150/tiny.yaml`) writes models, predictions, and results under `output/runs/singlelabel/clinc150/tiny/`.

Files in `config/algorithm/hyperparameters/` should be committed to git.
