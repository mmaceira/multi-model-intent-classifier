# Configuration

The project uses YAML configuration files to manage experiments and model settings.

## Configuration Reference

| Key | Type | Default | Description | Env Override |
|-----|------|---------|-------------|--------------|
| `general.seed` | int | 42 | Global random seed | `SEED` |
| `general.run_name` | str | "experiment_100_classes" | Experiment identifier | - |
| `dataset.name` | str | "clinc150" | Dataset loader to use | `DATASET_NAME` |
| `dataset.use_oos` | bool | false | Include out-of-scope examples | - |
| `dataset.max_classes` | int \| None | None | Limit number of classes | - |
| `dataset.max_train_samples` | int \| None | None | Limit training samples | - |
| `dataset.max_test_samples` | int \| None | None | Limit test samples | - |
| `model.embedding_backend` | str | "sbert" | Embedding backend: sbert \| openai | - |
| `model.sbert_model_name` | str | "sentence-transformers/all-MiniLM-L6-v2" | SBERT model name | - |
| `model.openai_model_name` | str | "text-embedding-3-small" | OpenAI model name | - |
| `model.classifier` | str | "linear_svm" | Classifier algorithm | `MODEL_TYPE` |
| `model.rag_top_k` | int | 25 | Neighbors per label for RAG | `RAG_K` |
| `model.llm_model` | str | "ollama/llama3.1:8b" | LLM model for RAG-LLM | `MODEL_ID` |

## Main Configuration (`config/config.yaml`)

The main configuration file controls experiment settings, dataset parameters, and model defaults:

```yaml
# General Configuration
general:
  run_name: "experiment_10_classes"     # Experiment identifier (used in output paths)
  seed: 42                              # Random seed for reproducibility

# Dataset configuration
dataset:
  name: "clinc150"                      # Dataset name
  use_oos: false                        # Include out-of-scope examples

# Paths (all are resolved relative to repo root)
paths:
  data_exploration_dir: "output/${general.run_name}/data_exploration"
  embeddings_dir: "output/${general.run_name}/embeddings"
  models_dir: "output/${general.run_name}/models"
  predictions_dir: "output/${general.run_name}/predictions"
  results_dir: "output/${general.run_name}/results"

# Model configuration
model:
  embedding_backend: "sbert"            # [sbert | openai]
  sbert_model_name: "sentence-transformers/all-MiniLM-L6-v2"
  openai_model_name: "text-embedding-3-small"
  classifier: "linear_svm"             # [linear_svm | naive_bayes | transformer_logreg]
  rag_top_k: 25                         # Number of neighbors for RAG models
  llm_model: "ollama/llama3.1:8b"      # Default LLM for RAG-LLM models
```

### Key Configuration Options

- `general.run_name`: Sets the experiment identifier and output directory name
- `model.llm_model`: Default LLM model for RAG-LLM (supports Ollama, OpenAI, Anthropic, etc.)
- `model.rag_top_k`: Number of similar examples to retrieve for RAG models
- `dataset.use_oos`: Whether to include out-of-scope examples as an extra class
- `dataset.max_classes`: Limit number of classes (None = all 150 classes)
- `dataset.max_train_samples`: Limit training samples (None = all ~23k samples)
- `dataset.max_test_samples`: Limit test samples (None = all ~5.7k samples)

## Model Selection Configuration (`config/models_config.yaml`)

This file controls which models are trained and their specific parameters:

```yaml
models:
  naive_bayes:
    enabled: true
    name: "Naive Bayes"
    class: "NaiveBayesClassifier"

  rag_llm:
    enabled: true
    name: "RAG-LLM"
    class: "RagSklearnAdapter"
    params:
      method: "llm"
      top_k: "${model.rag_top_k}"
      model: "${model.llm_model}"      # Uses value from config.yaml
      use_openai: false                 # false = SBERT embeddings, true = OpenAI embeddings
```

### Model Configuration Tips

- Set `enabled: false` to skip training a model
- Use `${model.rag_top_k}` to reference values from `config.yaml`
- For RAG-LLM, set `use_openai: true` to use OpenAI embeddings (requires `OPENAI_API_KEY`)
- Change `model` parameter to switch LLM providers (Ollama, OpenAI, etc.)

## Multiple Experiment Configurations

The project includes several pre-configured experiment files:
- `config/config.yaml` - Default experiment (full dataset)
- `config/config_10_classes.yaml` - 10 classes experiment (full dataset, 10 classes)
- `config/config_25_classes.yaml` - 25 classes experiment (full dataset)
- `config/config_tiny_dataset.yaml` - Small dataset for quick testing (10 classes, 100 train samples, 50 test samples)

**To use a different configuration file:**

```bash
# Set CONFIG_FILE environment variable
export CONFIG_FILE=config_tiny_dataset.yaml
python scripts/pipeline/run_all.py

# Or inline
CONFIG_FILE=config_25_classes.yaml python scripts/pipeline/run_all.py
```

## Dataset Configuration

See [Experiments](experiments.md) for detailed information on dataset parameters.

### Example Dataset Configuration

```yaml
# Dataset configuration
dataset:
  name: "clinc150"                           # dataset name (CLINC150 intent classification)
  use_oos: false                             # include out-of-scope examples
  max_classes: 10                            # Limit number of classes (None = all 150 classes)
  max_train_samples: 1000                    # Limit training samples (None = all ~23k samples)
  max_test_samples: 500                     # Limit test samples (None = all ~5.7k samples)
```

## Hyperparameter Configuration

Tuned hyperparameters are automatically loaded from `config/hyperparameters/{config_name}/`. See [Hyperparameter Tuning](hyperparameter_tuning.md) for details.

## Environment Variables

The following environment variables can be used:

- `CONFIG_FILE`: Override the default config file (e.g., `CONFIG_FILE=config_tiny_dataset.yaml`)
- `OPENAI_API_KEY`: Required for OpenAI embeddings and LLM models
- `MLFLOW_TRACKING_URI`: Optional MLflow tracking server URI

## Configuration Validation

The project uses Pydantic schemas for type-safe configuration validation. Invalid configurations will raise clear error messages indicating what needs to be fixed.

## Creating Custom Configurations

1. Copy an existing config file:
   ```bash
   cp config/config.yaml config/my_experiment.yaml
   ```

2. Edit `my_experiment.yaml` with your settings

3. Use it:
   ```bash
   CONFIG_FILE=my_experiment.yaml python scripts/pipeline/run_all.py
   ```

## Configuration Best Practices

1. **Use Descriptive Run Names**: Choose `run_name` that describes your experiment
2. **Version Control Configs**: Commit your config files to git
3. **Separate Configs for Different Experiments**: Don't modify the default config, create new ones
4. **Document Changes**: Add comments in config files explaining non-standard settings
5. **Test with Small Datasets First**: Use `config_tiny_dataset.yaml` for initial testing
