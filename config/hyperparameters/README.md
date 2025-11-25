# Hyperparameters Directory

This directory contains tuned hyperparameters for all models.

## Files

- `best_{model_name}.yaml` - Individual files for each model (one file per model)
  - Example: `best_naive_bayes.yaml`, `best_transformer_logreg.yaml`, `best_rag_kmajority.yaml`
  - Each file contains only the hyperparameters for that specific model

## Usage

Hyperparameters are automatically loaded by the model loader when training models. If no hyperparameters are found here, models will use defaults from `config/models_config.yaml`.

## Generating Hyperparameters

Run the hyperparameter tuning script:

```bash
python scripts/tune_hyperparams.py --config config/config.yaml --all
```

This will:
1. Tune hyperparameters using the validation set
2. Save results to `config/hyperparameters/` (used by pipeline)
3. Also save a copy to `output/hyperparams_tune/` (for reference)

## Git

**These files should be committed to the repository** as they are:
- Small configuration files
- Reproducible (same config = same results)
- Useful for collaboration
- Documentation of what hyperparameters were used
