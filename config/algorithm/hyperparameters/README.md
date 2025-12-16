# Hyperparameters

## Overview

Tuned hyperparameters for all models.

## Files

- `best_{model_name}.yaml` - Individual files for each model
- Example: `best_naive_bayes.yaml`, `best_transformer_logreg.yaml`

## Usage

Hyperparameters are automatically loaded by the model loader during training. If not found, models use defaults from `config/algorithm/models_config.yaml`.

## Generating Hyperparameters

```bash
uv run python scripts/tune_hyperparams.py --config config/experiments/clinc150/tiny.yaml --all
```

This saves results to:
- `config/algorithm/hyperparameters/{config_name}/` (used by pipeline)
- `output/hyperparams_tune/{config_name}/` (reference)

## Git

These files should be committed to the repository as they are:
- Small configuration files
- Reproducible
- Useful for collaboration
- Documentation of hyperparameters used
