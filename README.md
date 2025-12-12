# Multi-Model Intent Classifier

Production-ready NLP pipeline for automated intent classification and semantic search.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

Automated intent classification supporting:
- Single-label and multi-label classification
- Multiple algorithms (Naive Bayes, SVM, Transformer-based, RAG)
- Semantic search and document retrieval
- Flexible embedding backends (SBERT, OpenAI)
- LLM integration (Ollama, OpenAI)

## Quickstart

```bash
# Install dependencies
uv sync --extra all

# Run end-to-end pipeline (single-label)
uv run python scripts/pipeline/run_all.py --config config/dataset/clinc150/tiny.yaml

# Run end-to-end pipeline (multi-label)
uv run python scripts/pipeline/run_all.py --config config/dataset/nlu_plus/tiny.yaml
```

Datasets download automatically from HuggingFace/GitHub. The pipeline trains models, generates predictions, and evaluates performance.

## Documentation

- [Installation](docs/installation.md) - Setup and dependencies
- [Pipeline](docs/pipeline.md) - Pipeline steps and data flow
- [Algorithms](docs/algorithms.md) - Model implementations
- [Configuration](docs/configuration.md) - Config files and settings
- [Running Experiments](docs/running_experiments.md) - How to run experiments
- [Development](docs/development.md) - Development setup


## Commands

### Classify via CLI

```bash
# Single-label
uv run intent-classify --model-path output/experiment/models/Linear\ SVM/ --text "what's my account balance?"

# Multi-label
uv run intent-classify --model-path output/experiment_nlu_plus/models/Linear\ SVM/ --text "check my account balance and transfer money"
```

### Train Models

```bash
# Single-label
uv run intent-train --config config/dataset/clinc150/tiny.yaml

# Multi-label
uv run intent-train --config config/dataset/nlu_plus/tiny.yaml
```

### Serve API

```bash
# Install API dependencies
uv sync --extra api

# Start server
CONFIG_FILE=config/dataset/clinc150/tiny.yaml uv run api-serve --host 0.0.0.0 --port 8000
```

### Run Demos

```bash
# Intent classifier demo
uv run python scripts/demos/intent_classifier_demo.py

# Semantic search demo
uv run python scripts/demos/semantic_search_demo.py

# Intent trend analyzer
uv run python scripts/demos/intent_trend_analyzer.py
```

## Models

| Model | Architecture | Use Case |
|-------|--------------|----------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |
| Embedding + LogReg | Flexible embeddings (SBERT/OpenAI) + Logistic Regression | High-accuracy with flexible embedding backend |
| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | Context-aware classification |

See [Algorithms](docs/algorithms.md) for details.



## Configuration

- Main config: `config/dataset/{dataset_name}/{config_name}.yaml` - Dataset, paths, embeddings, LLM
- Models config: `config/algorithm/models_config.yaml` - Which models to train

```bash
# Single-label
uv run intent-train --config config/dataset/clinc150/tiny.yaml

# Multi-label
uv run intent-train --config config/dataset/nlu_plus/tiny.yaml
```

Enable/disable models in `config/algorithm/models_config.yaml`. See [Configuration](docs/configuration.md) for details.



## Development

```bash
# Install dev dependencies
uv sync --extra dev
pre-commit install

# Run tests
uv run pytest -q

# Format code
uv run black intent_classifier/ scripts/
uv run ruff check intent_classifier/ scripts/
```

See [Development](docs/development.md) for details.

## Testing

```bash
# Run all tests
uv run pytest -q

# Run specific test
uv run pytest tests/test_algorithms.py -q
```
