## Multi-Model Intent Classifier

Production-ready NLP pipeline for automated intent classification, semantic search, and RAG‑LLM exploration.

Supports:
- Single-label and multi-label intent classification
- Multiple algorithms (Naive Bayes, SVM, Transformer-based, RAG, RAG‑LLM)
- Semantic search and document retrieval
- Flexible embedding backends (SBERT, OpenAI, Ollama/Qwen)

See `docs/` for full details.

---

## Overview

This repository provides:
- A configurable training pipeline (single-label and multi-label)
- Multiple intent classifiers and RAG‑based models
- A CLI for classification and RAG exploration
- A FastAPI server for serving trained models
- Optional Gradio demos over trained artifacts

---

## Quickstart

```bash
# Install all dependencies
uv sync --extra all

# Run end-to-end pipeline (single-label)
DATASET=clinc150 VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py

# Run end-to-end pipeline (multi-label)
DATASET=nlu_plus VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py
```

Outputs are written under:

- `output/runs/<label_type>/<dataset>/<variant>/`

For installation details and environment variables, see **`docs/installation.md`**.

---

## Commands

- **Train & experiments**

  ```bash
  # Train models using layered config (same as run_all.py)
  DATASET=nlu_plus VARIANT=tiny \
    uv run intent-train

  # Enable Ray Tune hyperparameter search
  DATASET=nlu_plus VARIANT=tiny \
    uv run intent-train --tune
  ```

  See **`docs/running_experiments.md`** and **`docs/hyperparameter_tuning.md`** for all options.

- **Classify via CLI**

  ```bash
  # Single-label (CLINC150 tiny, Linear SVM)
  uv run intent-classify \
    --model-path "output/runs/singlelabel/clinc150/tiny/models/linear_svm/" \
    --text "what's my account balance?"
  ```

  See **`docs/classification.md`** for multi-label examples, batch mode, and output formats.

- **RAG‑LLM exploration (`rag-explore`)**

  ```bash
  DATASET=nlu_plus VARIANT=tiny \
    uv run rag-explore --provider ollama --model qwen2.5:14b --k 10 \
    --prompt-style short --text "reset my card pin"
  ```

  Prompt styles, providers, and model options are described in **`docs/llm_providers.md`**.

- **Serve API**

  ```bash
  # Install API dependencies once
  uv sync --extra api

  # Start server (example)
  DATASET=clinc150 VARIANT=tiny \
    uv run api-serve --host 0.0.0.0 --port 8000
  ```

  API endpoints and examples live in **`docs/api.md`**.

- **Demos (Gradio UIs)**

  ```bash
  uv sync --extra ui
  DATASET=nlu_plus VARIANT=tiny \
    uv run python scripts/demos/intent_classifier_demo.py
  ```

  See `scripts/demos/README.md` and **`docs/classification.md`** for screenshots and usage.

---

## Configuration

Configuration is layered:

- **Base**: `config/base/{defaults,providers,prompts}.yaml`
- **Dataset**: `config/datasets/{dataset}.yaml`
- **Experiments**: `config/experiments/{dataset}/{variant}.yaml`
- **Models**: `config/algorithm/models_config.yaml`

Use `DATASET` and `VARIANT` for most workflows:

```bash
# Single-label
DATASET=clinc150 VARIANT=tiny \
  uv run intent-train

# Multi-label
DATASET=nlu_plus VARIANT=tiny \
  uv run intent-train
```

For advanced configuration, see **`docs/config.md`** and **`docs/output_schema.md`**.

---

## Development

```bash
# Install dev dependencies and hooks
uv sync --extra dev
pre-commit install

# Run tests
uv run pytest -q
```

Formatting and development conventions are documented in **`docs/development.md`**.

---

## Testing

```bash
# Run all tests
uv run pytest -q

# Run a specific test file
uv run pytest tests/test_algorithms.py -q
```

Testing strategy and coverage expectations are described in **`docs/development.md`**.

---

## Documentation

- **Overview & concepts**
  - `docs/pipeline.md` – Pipeline steps and data flow
  - `docs/experiments.md` – Datasets and experiment configs
  - `docs/algorithms.md` – Algorithms and model variants
  - `docs/model_architecture.md` – Technical model architecture
  - `docs/models.md` – When to use each model
- **Usage**
  - `docs/installation.md` – Installation & environment
  - `docs/classification.md` – Classification CLI
  - `docs/api.md` – FastAPI server
  - `docs/running_experiments.md` – Experiments & runs
  - `docs/hyperparameter_tuning.md` – Ray Tune integration
  - `docs/llm_providers.md` – LLM and embedding providers
- **Configuration & outputs**
  - `docs/config.md` – Config files and settings
  - `docs/output_schema.md` – Run folder schema and artifacts
- **Development**
  - `docs/development.md` – Development workflow and tooling

---

## Release

- Use **Commitizen** for all commits and releases:

  ```bash
  cz c        # create a conventional commit
  ```

- Releases and changelog generation follow the Commitizen configuration in `pyproject.toml` / `.cz.toml`.
