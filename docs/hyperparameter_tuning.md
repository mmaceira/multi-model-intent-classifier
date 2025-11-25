# Hyperparameter Tuning

This repository includes a comprehensive hyperparameter tuning system that follows ML best practices.

## Overview

The hyperparameter tuning script (`scripts/tune_hyperparams.py`) optimizes hyperparameters for all models using the **validation set** (not the test set), ensuring proper model selection without data leakage.

## Supported Models

The tuning script supports hyperparameter optimization for:
- **Naive Bayes**: Tunes `alpha` (smoothing parameter)
- **Linear SVM**: Tunes `C` (regularization parameter)
- **Linear SVM Bigrams**: Tunes `C` (regularization parameter)
- **Transformer LogReg**: Tunes `C` (regularization parameter) - MiniLM + Logistic Regression
- **Embedding LogReg**: Tunes `C` (regularization parameter) - Flexible embeddings (SBERT or OpenAI) + Logistic Regression
- **RAG KMajority**: Tunes `top_k` (number of neighbors)
- **RAG Centroid**: Evaluates default configuration
- **RAG LLM**: Tunes `top_k` (number of neighbors)

## Usage

### Tune All Models

```bash
# Tune all models with default settings
python scripts/tune_hyperparams.py --config config/config.yaml --all

# Tune with more samples for better results
python scripts/tune_hyperparams.py --config config/config.yaml --all --num-samples 50
```

### Tune Specific Model

```bash
# Tune only Naive Bayes
python scripts/tune_hyperparams.py --config config/config.yaml --algo nb --num-samples 30

# Tune only Linear SVM
python scripts/tune_hyperparams.py --config config/config.yaml --algo svm --num-samples 30

# Tune Embedding LogReg (defaults to SBERT embeddings, no API key needed)
python scripts/tune_hyperparams.py --config config/config.yaml --algo embedding_logreg --num-samples 30

# Tune Embedding LogReg with OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key"
python scripts/tune_hyperparams.py --config config/config.yaml --algo embedding_logreg --num-samples 30
```

## Integration with Training Pipeline

**The tuned hyperparameters are automatically loaded and used during training!**

1. **Run hyperparameter tuning** (optional but recommended):
   ```bash
   python scripts/tune_hyperparams.py --config config/config.yaml --all
   ```

2. **Run training pipeline** - it will automatically use tuned hyperparameters:
   ```bash
   python scripts/pipeline/03_model_training.py
   # or
   python scripts/pipeline/run_all.py
   ```

The model loader checks for individual hyperparameter files in `config/hyperparameters/{config_name}/` and automatically applies tuned hyperparameters to models. If no tuned hyperparameters are found, models use defaults from the configuration files.

## Storage Location

Tuned hyperparameters are saved as **individual YAML files** (one per model) to two locations, with **config-specific subdirectories**:

**Primary location (used by model loader):**
- `config/hyperparameters/{config_name}/best_{model_name}.yaml` - One file per model
  - Example: `best_naive_bayes.yaml`, `best_transformer_logreg.yaml`, `best_rag_kmajority.yaml`

**Secondary location (for reference/backup):**
- `output/hyperparams_tune/{config_name}/best_{model_name}.yaml` - Copy for reference

**Example:** If you use `config_tiny_dataset.yaml`, hyperparameters will be saved to:
- `config/hyperparameters/config_tiny_dataset/best_naive_bayes.yaml`
- `config/hyperparameters/config_tiny_dataset/best_transformer_logreg.yaml`
- `config/hyperparameters/config_tiny_dataset/best_rag_kmajority.yaml`
- etc.

This ensures that hyperparameters tuned with different config files are kept separate and don't conflict.

### Example Output Files

`best_naive_bayes.yaml`:
```yaml
alpha: 0.123
```

`best_transformer_logreg.yaml`:
```yaml
C: 2.456
```

`best_rag_kmajority.yaml`:
```yaml
top_k: 15
```

## Committing Hyperparameters to Git

**Yes, you can (and should) commit hyperparameter files!**

The hyperparameter YAML files are:
- ✅ **Small** (typically < 1 KB each)
- ✅ **Reproducible** (same config = same results)
- ✅ **Useful for collaboration** (others can use your tuned hyperparameters)
- ✅ **Documentation** (shows what hyperparameters were used)

Hyperparameters in `config/hyperparameters/` are **not ignored** by `.gitignore` and should be committed to the repository. The `output/hyperparams_tune/` directory is kept for reference but is ignored by git.

**To commit hyperparameters:**
```bash
# After running hyperparameter tuning
# Note: Hyperparameters are stored in config-specific subdirectories
git add config/hyperparameters/*/
git commit -m "Add tuned hyperparameters for all models"
```

**Note:**
- Hyperparameters in `config/hyperparameters/` should be committed (they're configuration)
- Output files (models, predictions, embeddings) remain in `.gitignore` as they are large and experiment-specific

## Best Practices

1. **Always tune before final training**: Run hyperparameter tuning before training your final models
2. **Use validation set**: The tuning script correctly uses the validation set (not test set)
3. **Test set is never used**: The test set remains completely separate for final evaluation only
4. **Reproducible**: Uses the same seed and dataset splits for consistency
5. **Config-specific**: Hyperparameters are stored per config file to avoid conflicts

## Why This Matters

- **Proper ML Workflow**: Uses validation set for hyperparameter selection (standard practice)
- **No Data Leakage**: Test set is never touched during tuning or training
- **Automatic Integration**: Tuned hyperparameters are automatically used in the pipeline
- **Comprehensive**: Supports all models in the pipeline, not just a subset

## Tuning Algorithm

The hyperparameter tuning uses Ray Tune with:
- **Search Space**: Log-uniform distributions for continuous parameters, uniform for discrete
- **Optimization**: TPE (Tree-structured Parzen Estimator) for efficient search
- **Metric**: Macro F1-score on validation set
- **Early Stopping**: Optional early stopping for faster convergence

## Advanced Usage

### Custom Search Spaces

You can modify the search spaces in `scripts/tune_hyperparams.py` for custom hyperparameter ranges.

### Parallel Tuning

Ray Tune automatically parallelizes trials. Adjust `num_samples` and available resources to control parallelism.

### Resume Tuning

Ray Tune supports resuming interrupted tuning sessions. Check Ray Tune documentation for details.
