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
DATASET=clinc150 VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py

# Run with a remote Ollama server for embeddings + LLMs (optional)
export MODEL_OLLAMA_ENDPOINT="http://<your-ollama-host>:11434"
export OLLAMA_API_BASE="http://<your-ollama-host>:11434"
DATASET=clinc150 VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py

# Run end-to-end pipeline (multi-label)
DATASET=nlu_plus VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py
```

Datasets download automatically from HuggingFace/GitHub. The pipeline trains models,
generates predictions, and evaluates performance. Outputs are written under:

- `output/runs/<label_type>/<dataset>/<variant>/`

### Using Configuration Files

Preferred selection is via `DATASET` / `VARIANT`:

```bash
# Train models (same as run_all.py)
DATASET=nlu_plus VARIANT=tiny \
  uv run intent-train

# Run with hyperparameter tuning
DATASET=nlu_plus VARIANT=tiny \
  uv run intent-train --tune

# Serve API with specific config and experiment output
DATASET=nlu_plus VARIANT=tiny \
  uv run api-serve --host 0.0.0.0 --port 8000

# Once running, you can test it with:
#   curl http://localhost:8000/health
#   curl http://localhost:8000/ready
#   curl http://localhost:8000/v1/models
#
# And make a prediction (example using the Linear SVM model):
#   curl -X POST http://localhost:8000/v1/predict \
#     -H "Content-Type: application/json" \
#     -d '{"model_id": "linear_svm", "text": "what is my account balance?"}'
#
# Or open http://localhost:8000/docs in your browser for the interactive UI.

```

You can also point to an explicit config file using `CONFIG_FILE`:

```bash
CONFIG_FILE=config/experiments/nlu_plus/tiny.yaml \
  uv run intent-train

uv run intent-tune --config config/experiments/nlu_plus/tiny.yaml
```

## Project Structure

```
multi-model-intent-classifier/
├── intent_classifier/          # Core package
│   ├── algorithms/             # Model implementations
│   ├── api/                    # FastAPI server
│   ├── cli/                    # CLI commands (classify, rag)
│   ├── datasets/               # Dataset loaders
│   ├── evaluation/             # Evaluation metrics and visualization
│   ├── pipeline/               # Training pipeline orchestrator
│   ├── prediction/             # Prediction interfaces
│   ├── rag/                    # RAG implementations
│   └── utils/                  # Shared utilities
├── scripts/                    # Utility scripts
│   ├── demos/                  # Interactive demos
│   ├── dev/                    # Development utilities
│   └── pipeline/               # Pipeline step scripts
├── config/                     # Configuration files
│   ├── base/                  # Shared defaults, providers, prompts
│   ├── datasets/              # Dataset configurations
│   ├── experiments/           # Per-dataset experiment variants
│   └── algorithm/             # Model configurations
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## Commands

### Classify via CLI

```bash
# Single-label (CLINC150 tiny, Linear SVM)
uv run intent-classify \
  --model-path "output/runs/singlelabel/clinc150/tiny/models/linear_svm/" \
  --text "what's my account balance?"

# Multi-label (NLU+ tiny, RAG-CentroidNN)
uv run intent-classify \
  --model-path "output/runs/multilabel/nlu_plus/tiny/models/rag_centroidnn/" \
  --text "check my balance and block my card"
```

See [Classification CLI](docs/classification.md) for all options and examples (JSON/text output, batch mode, direct `.pkl` paths, and more).

### RAG-LLM Classification (Exploration CLI)

`rag-explore` is an **exploratory RAG‑LLM CLI**: it reloads training examples and label
definitions from the current layered config (`config/base`, `config/datasets`,
`config/experiments`) on each run, so you can quickly try different providers, models,
retrieval sizes, and prompt styles **before** baking those choices into your training
config.

```bash
# Prerequisite: run the multi-label pipeline for NLU+ tiny config
DATASET=nlu_plus VARIANT=tiny uv run python scripts/pipeline/run_all.py

# Using Ollama (default provider) with training config defaults
DATASET=nlu_plus VARIANT=tiny uv run rag-explore --provider ollama --model qwen2.5:14b --k 10 --prompt-style short --text "reset my card pin"

# Using OpenAI with default prompt
export OPENAI_API_KEY=sk-...
DATASET=nlu_plus VARIANT=tiny uv run rag-explore --provider openai --model gpt-4o-mini --k 10 --text "reset my card pin"

# Using n8n prompt style (Catalan, email cleaning) with Ollama
DATASET=nlu_plus VARIANT=tiny uv run rag-explore --provider ollama --model qwen2.5:14b --k 10 --prompt-style n8n_prompt --text "reset my card pin"
```

The `rag-explore` command accepts:
- `--provider`: LLM provider (`openai` or `ollama`, default: `ollama`)
- `--model`: LLM model identifier (e.g., `gpt-4o-mini` for OpenAI, `llama3.1:8b` for Ollama)
- `--k`: Number of similar examples to retrieve (default: 10)
- `--prompt-style`: Prompt style (`default`, `short`, or `n8n_prompt`, default: `default`)
  - `default`: Full detailed prompt with all instructions
  - `short`: Concise prompt for faster/cheaper inference
  - `n8n_prompt`: Email cleaning + classification prompt (Catalan)
- `--text`: Text to classify

### Training & Hyperparameter Tuning

The main README shows the basic training and tuning commands in **Quickstart** and **Using Configuration Files**.
For all training options, experiment flows, and Ray Tune configuration details, see:
- [Running Experiments](docs/running_experiments.md)
- [Hyperparameter Tuning](docs/hyperparameter_tuning.md)

### Serve API

```bash
# Install API dependencies (required before first use)
uv sync --extra api

# Start server (single-label)
DATASET=clinc150 VARIANT=tiny \
  uv run api-serve --host 0.0.0.0 --port 8000

# Start server (multi-label)
DATASET=nlu_plus VARIANT=tiny \
  uv run api-serve --host 0.0.0.0 --port 8000
```

The server will:
- Load models from `output/runs/{label_type}/{dataset_name}/{variant}/models/`
- Start on `http://0.0.0.0:8000`
- Provide interactive API docs at `http://localhost:8000/docs`
- Enable CORS and rate limiting (100 requests per 60s)

**Testing the API:**

```bash
# Health check
curl http://localhost:8000/health

# Check if ready (shows number of models loaded)
curl http://localhost:8000/ready

# List available models
curl http://localhost:8000/v1/models

# Get model details
curl http://localhost:8000/v1/models/Linear%20SVM

# Make a prediction
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Linear SVM", "text": "what is my account balance?"}'

# Interactive API documentation
# Open in browser: http://localhost:8000/docs
```

### Run Demos

The demos provide Gradio-based UIs for exploring the trained models and embeddings.
They require the `ui` extra (or `all`) and a completed pipeline run so models and
embeddings exist under `output/` and `embeddings/`:

```bash
# Install UI dependencies once (if you didn’t use --extra all)
uv sync --extra ui

# Intent classifier demo – interactive intent classification & RAG over trained models
DATASET=nlu_plus VARIANT=tiny \
MODELS_PATH=output/runs/multilabel/nlu_plus/tiny/models \
EMBEDDINGS_PATH=output/runs/multilabel/nlu_plus/tiny/features/embeddings/sbert \
uv run python scripts/demos/intent_classifier_demo.py

# Semantic search demo – semantic search over CLINC150 utterances using FAISS + SBERT
DATASET=nlu_plus VARIANT=tiny \
EMBEDDINGS_PATH=output/runs/multilabel/nlu_plus/tiny/features/embeddings/sbert \
uv run python scripts/demos/semantic_search_demo.py

# Intent trend analyzer – LLM‑based trend analysis over retrieved similar utterances
DATASET=nlu_plus VARIANT=tiny \
EMBEDDINGS_PATH=output/runs/multilabel/nlu_plus/tiny/features/embeddings/sbert \
uv run python scripts/demos/intent_trend_analyzer.py
```

### Development Utilities

Development and testing utilities are available in `scripts/dev/`:

```bash
# Test single-label classification
uv run python scripts/dev/test_single_label.py

# Test multi-label classification
uv run python scripts/dev/test_multi_label.py

# Test LLM connections
uv run python scripts/dev/test_llm_connection.py

# Check configuration loaders
uv run python scripts/dev/check_config_loaders.py
```

See `scripts/dev/README.md` for a complete list of development utilities.

## Models

| Model | Embeddings / Features | Notes |
|-------|------------------------|-------|
| Multinomial Naive Bayes | TF-IDF (unigrams) | Fast, lightweight baseline |
| Linear SVM | TF-IDF (unigrams) | Balanced speed and accuracy |
| TF-IDF bigrams + SVM | TF-IDF (uni + bi-grams) | Strong bag-of-words baseline |
| MiniLM + LogReg | MiniLM SentenceTransformer | High-accuracy semantic classifier |
| Embedding + LogReg (SBERT) | SBERT SentenceTransformer | Dense local embeddings, no API |
| Embedding + LogReg (Qwen/Ollama) | Qwen/Ollama embedding model (HTTP) | Same head as SBERT variant with Qwen embeddings |
| RAG-CentroidNN | SBERT FAISS index | Centroid-based RAG over embeddings |
| RAG-kMajority (SBERT/Qwen) | SBERT / Qwen FAISS index | k-NN majority-vote RAG over embeddings |
| RAG-LLM (TF-IDF) | TF-IDF bi-gram retriever + Qwen LLM | Context-aware LLM classifier without embedding index |
| RAG-LLM (SBERT/Qwen embeddings) | SBERT / Qwen FAISS index + Qwen LLM | Embedding-backed RAG-LLM with prompts (default/short/n8n) |

See [Algorithms](docs/algorithms.md) and [Classification CLI](docs/classification.md) for details.



## Configuration

- Base config: `config/base/{defaults,providers,prompts}.yaml`
- Dataset config: `config/datasets/{dataset}.yaml`
- Experiments: `config/experiments/{dataset}/{variant}.yaml`
- Models config: `config/algorithm/models_config.yaml` - Which models to train

```bash
# Single-label
DATASET=clinc150 VARIANT=tiny \
  uv run intent-train

# Multi-label
DATASET=nlu_plus VARIANT=tiny \
  uv run intent-train
```

Enable/disable models in `config/algorithm/models_config.yaml`. See
[Configuration](docs/config.md) for details. For the new output layout, see
[Output Schema](docs/output_schema.md).



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

## Documentation

- [Installation](docs/installation.md) - Setup and dependencies
- [Pipeline](docs/pipeline.md) - Pipeline steps and data flow
- [Experiments](docs/experiments.md) - Datasets and experiment configs
- [Algorithms](docs/algorithms.md) - Conceptual model overview
- [Model Architecture](docs/model_architecture.md) - Technical specs and resources
- [Models](docs/models.md) - Model families and when to use them
- [Classification CLI](docs/classification.md) - CLI usage and examples
- [Configuration](docs/config.md) - Config files and settings
- [Output Schema](docs/output_schema.md) - Run folder structure and artifacts
- [Running Experiments](docs/running_experiments.md) - How to run experiments
- [Hyperparameter Tuning](docs/hyperparameter_tuning.md) - Hyperparameter optimization with Ray Tune
- [LLM Providers](docs/llm_providers.md) - Ollama/OpenAI/Anthropic + embeddings
- [API Server](docs/api.md) - FastAPI serving for trained models
- [Development](docs/development.md) - Development setup
