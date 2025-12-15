# Installation

## Overview

Install dependencies using `uv` (recommended) or `pip`.

## Quickstart
```bash
# Install all dependencies
uv sync --extra all

# Verify installation
uv run pytest tests/test_dataset_clinc150.py -q
```

## Optional Dependencies

| Extra | Includes | Command |
|-------|----------|---------|
| `all` | Everything | `uv sync --extra all` |
| `api` | FastAPI server | `uv sync --extra api` |
| `ui` | Gradio UI | `uv sync --extra ui` |
| `dev` | Development tools | `uv sync --extra dev` |

## Development Setup

For contributing and local development (pre-commit, formatting, full test suite), see
`development.md`.

## Requirements

- Python 3.12+
- `uv` package manager (recommended)
- Internet connection (for dataset downloads)

For OpenAI features, set `OPENAI_API_KEY` in your environment or `.env` file.
