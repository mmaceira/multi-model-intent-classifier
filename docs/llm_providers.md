# LLM Providers

## Overview

RAG-LLM supports multiple LLM providers via `litellm`. Default: Ollama (local, no API keys). Easy to switch to OpenAI or other providers.

## Quickstart

### Ollama (Default)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start server
ollama serve

# Pull model
ollama pull llama3.1:8b

# Verify
ollama list
```

#### Ollama embeddings (Qwen)

The pipeline can call Ollama's native **embeddings** endpoint (used by models such as
`qwen3-embedding:latest`) via the `/api/embeddings` route.

- **Endpoint resolution priority** in this project:
  - `MODEL_OLLAMA_ENDPOINT`
  - `OLLAMA_API_BASE`
  - `OLLAMA_HOST`
  - Fallback: `http://localhost:11434`

To point the embedding client at a specific Ollama server (local or remote), set one
of these environment variables before running the pipeline, for example:

```bash
export MODEL_OLLAMA_ENDPOINT="http://<your-ollama-host>:11434"
```

You can verify that the embeddings endpoint and model are working with:

```bash
curl -sS -X POST http://<your-ollama-host>:11434/api/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embedding:latest",
    "prompt": "hello"
  }'
```

If you see an error like `model "qwen3-embedding:latest" not found`, pull the model on
the Ollama host:

```bash
ollama pull qwen3-embedding:latest
```

### OpenAI

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Use an experiment config with model.llm_backend: "openai"
uv run python scripts/pipeline/run_all.py
```

## Configuration

Provider defaults live in `config/base/providers.yaml` under the `providers` block:

```yaml
providers:
  ollama:
    endpoint: "http://localhost:11434"
    llm_default: "ollama/llama3.1:8b"
    embed_default: "sentence-transformers/all-MiniLM-L6-v2"
  openai:
    llm_default: "gpt-4o-mini"
    embed_default: "text-embedding-3-small"
```

The main experiment config then selects a backend:

```yaml
model:
  llm_backend: "ollama"          # or "openai" / "none"
  embedding_backend: "sbert"     # or "openai" / "ollama"
```

Set `OPENAI_API_KEY` when using OpenAI LLMs/embeddings.

## Supported Models

### Ollama
- `ollama/llama3.1:8b` (default)
- `ollama/llama3.2:3b`
- `ollama/mistral`
- See [Ollama library](https://ollama.ai/library) for more

### OpenAI
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic
- `claude-3-5-sonnet`
- `claude-3-opus`
- `claude-3-haiku`

See [litellm documentation](https://docs.litellm.ai/) for full list.

## Embedding Backends

### SBERT (Default)
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- No API key needed
- Fast and privacy-friendly

### OpenAI Embeddings
- API-based using `text-embedding-3-small`
- Requires `OPENAI_API_KEY`
- Set `model.embedding_backend: "openai"` in the experiment configuration

## RAG‑LLM Exploration (`rag-explore`)

Use the `rag-explore` CLI for interactive RAG‑LLM experiments before baking choices into
your training configuration. It reloads label definitions and training examples from the
current layered config on each run.

```bash
# Prerequisite: run the multi-label pipeline for NLU+ tiny
DATASET=nlu_plus VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py

# Using Ollama (default provider) with explicit model and prompt style
DATASET=nlu_plus VARIANT=tiny \
  uv run rag-explore \
    --provider ollama \
    --model qwen2.5:14b \
    --k 10 \
    --prompt-style short \
    --text "reset my card pin"

# Using OpenAI
export OPENAI_API_KEY=sk-...
DATASET=nlu_plus VARIANT=tiny \
  uv run rag-explore \
    --provider openai \
    --model gpt-4o-mini \
    --k 10 \
    --text "reset my card pin"
```

Key options:

- `--provider`: `ollama` (default) or `openai`
- `--model`: provider-specific model ID (e.g. `qwen2.5:14b`, `gpt-4o-mini`)
- `--k`: number of similar examples to retrieve (default: 10)
- `--prompt-style`: `default`, `short`, or `n8n_prompt`
- `--text`: text to classify

LLM defaults (provider, base model, temperature, etc.) live in `config/base/providers.yaml`.

## Commands (switching providers)

```bash
# Switch to OpenAI
export OPENAI_API_KEY="your-key"
# Use an experiment config with model.llm_backend: "openai"
uv run python scripts/pipeline/run_all.py

# Switch back to Ollama
# Use an experiment config with model.llm_backend: "ollama"
uv run python scripts/pipeline/run_all.py
```

## Troubleshooting

- **Ollama model not found**: Run `ollama pull <model-name>`
- **Connection refused**: Ensure `ollama serve` is running
- **SBERT download fails**: Check internet connection
- **OpenAI API key not found**: Set `OPENAI_API_KEY` environment variable
