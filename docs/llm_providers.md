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

## Commands

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
