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

### OpenAI

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Update config: llm_model: "gpt-4o-mini"
uv run python scripts/pipeline/run_all.py
```

## Configuration

### Ollama

Default config:
```yaml
model:
  llm_model: "ollama/llama3.1:8b"
```

No API key needed. SBERT embeddings download automatically.

### OpenAI

Update config:
```yaml
model:
  llm_model: "gpt-4o-mini"
```

Set `OPENAI_API_KEY` environment variable.

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
- Set `use_openai: true` in model configuration

## RAG‑LLM Exploration (`rag-explore`)

Use the `rag-explore` CLI for interactive RAG‑LLM experiments before baking choices into
your training configuration. It reloads label definitions and training examples from the
current dataset config and `config/llm_config.yaml` on each run.

```bash
# Prerequisite: run the multi-label pipeline for NLU+ tiny
CONFIG_FILE=config/dataset/nlu_plus/tiny.yaml \
  uv run python scripts/pipeline/run_all.py

# Using Ollama (default provider) with explicit model and prompt style
CONFIG_FILE=config/dataset/nlu_plus/tiny.yaml \
  uv run rag-explore \
    --provider ollama \
    --model qwen2.5:14b \
    --k 10 \
    --prompt-style short \
    --text "reset my card pin"

# Using OpenAI
export OPENAI_API_KEY=sk-...
CONFIG_FILE=config/dataset/nlu_plus/tiny.yaml \
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

LLM defaults (provider, base model, temperature, etc.) live in `config/llm_config.yaml`.

## Commands (switching providers)

```bash
# Switch to OpenAI
export OPENAI_API_KEY="your-key"
# Update config: llm_model: "gpt-4o-mini"
uv run python scripts/pipeline/run_all.py

# Switch back to Ollama
# Update config: llm_model: "ollama/llama3.1:8b"
uv run python scripts/pipeline/run_all.py
```

## Troubleshooting

- **Ollama model not found**: Run `ollama pull <model-name>`
- **Connection refused**: Ensure `ollama serve` is running
- **SBERT download fails**: Check internet connection
- **OpenAI API key not found**: Set `OPENAI_API_KEY` environment variable
