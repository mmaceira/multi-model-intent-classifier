# Configuration

## Overview

YAML files and environment variables control datasets, models, and experiments.

## Main Config

Config files in `config/dataset/{dataset_name}/{config_name}.yaml`:

- `general.run_name`, `general.seed`
- `dataset.name`, `dataset.use_oos`, `dataset.max_classes`, `dataset.max_train_samples`
- `model.embedding_backend` (`"sbert"` or `"openai"`) – SBERT is the default
- `model.sbert_model_name` (local SentenceTransformer model)
- `model.openai_model_name` (OpenAI embedding model)
- `model.ollama_embedding_model_name` (Ollama embedding model, e.g. Qwen3 embeddings via litellm)
- `model.llm_model` (default: value from `config/llm_config.yaml`, typically an Ollama model)
- `model.rag_top_k` (number of neighbors)

## Model Selection

`config/algorithm/models_config.yaml` controls which models are trained:

```yaml
models:
  naive_bayes:
    enabled: true
    name: "Naive Bayes"
    class: "NaiveBayesClassifier"
  rag_llm:
    enabled: true
    params:
      top_k: "${model.rag_top_k}"
      model: "${model.llm_model}"
```

Enable/disable models by toggling `enabled`. Use string interpolation to reference main config values.

## Preconfigured Configs

- `config/dataset/clinc150/default.yaml` - Full CLINC150
- `config/dataset/clinc150/tiny.yaml` - Quick test
- `config/dataset/nlu_plus/default.yaml` - Full NLU++
- `config/dataset/nlu_plus/tiny.yaml` - Quick test

## Environment Variables

- `CONFIG_FILE` - Main config file path
- `OPENAI_API_KEY` - Required for OpenAI features
- `SEED` - Random seed
- `SBERT_BATCH`, `OPENAI_BATCH` - Embedding batch sizes

Configuration is validated with Pydantic schemas.
