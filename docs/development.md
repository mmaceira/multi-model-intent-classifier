# Development

## Overview

Development setup, code style, and testing guidelines.

## Quickstart

```bash
# Install dev dependencies
uv sync --extra dev

# Set up pre-commit hooks
pre-commit install

# Run tests
uv run pytest -q
```

## Commands

### Code Formatting

```bash
# Format code
uv run black intent_classifier/ scripts/

# Lint code
uv run ruff check intent_classifier/ scripts/
```

### Testing

```bash
# Run all tests
uv run pytest -q

# Run specific test
uv run pytest tests/test_dataset_clinc150.py -q

# Run with coverage
uv run pytest --cov=intent_classifier -q
```

## Code Style

- PEP 8-compliant with type hints on public functions
- Format with `black`, lint with `ruff`
- Use docstrings for all public functions and classes
- Include type hints

## Project Structure

```
.
├── config/              # Configuration files
├── intent_classifier/   # Library code
├── scripts/             # CLI and pipeline scripts
├── tests/               # Test files
└── docs/                # Documentation
```

## Adding New Algorithms

1. Create algorithm in `intent_classifier/algorithms/`
2. Add to `intent_classifier/algorithms/__init__.py`
3. Add configuration in `config/algorithm/models_config.yaml`
4. Write tests in `tests/`

## Adding New Datasets

1. Create loader in `intent_classifier/datasets/`
2. Add to `intent_classifier/datasets/dataset.py`
3. Update config files in `config/`
4. Write tests

## Pre-commit Hooks

Hooks run automatically on `git commit`. Run manually:

```bash
pre-commit run --all-files
```

## Git Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit using Commitizen: `cz c`
3. Run tests and formatting: `uv run pytest -q && uv run black intent_classifier/ scripts/`
4. Push and create pull request
