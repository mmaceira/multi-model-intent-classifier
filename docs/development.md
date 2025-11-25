# Development

This guide covers development setup, code style, testing, and contributing to the project.

## Environment Setup

1. Install development dependencies (using uv):
   ```bash
   uv sync --extra dev
   ```
   Or using pip:
   ```bash
   pip install -e ".[dev]"
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Document all public functions with docstrings
- Run `black` and `flake8` before committing

### Formatting

```bash
# Format code with black
black src/ scripts/

# Check with flake8
flake8 src/ scripts/
```

## Testing

**Important**: Tests require the HuggingFace `datasets` library. Install dev dependencies with `pip install -e .[dev]` or `uv sync --extra dev` before running `pytest`.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dataset_clinc150.py

# Run with coverage
pytest --cov=src

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (APIs, file I/O)

## Project Structure

```
multi-model-intent-classifier/
├── config/              # Configuration files
│   ├── config.yaml     # Main configuration file
│   ├── models_config.yaml # Model selection and parameters
│   └── notebook_setup.py # Configuration setup for pipeline scripts
├── scripts/             # Utility scripts
│   ├── pipeline/        # Training pipeline scripts
│   ├── api/            # FastAPI implementation
│   └── tune_hyperparams.py            # Hyperparameter tuning
├── output/             # Model outputs and results
├── src/                # Main source code
│   ├── algorithms/     # ML algorithms implementation
│   ├── datasets/       # Dataset handling
│   ├── embeddings/     # Embedding generation
│   ├── evaluation/     # Model evaluation tools
│   ├── rag/            # RAG implementation
│   └── utils/          # Utility functions
├── tests/              # Test files
└── docs/               # Documentation
```

## Adding New Algorithms

1. Create a new file in `src/algorithms/`:
   ```python
   # src/algorithms/my_algorithm.py
   from sklearn.base import BaseEstimator, ClassifierMixin

   class MyAlgorithm(BaseEstimator, ClassifierMixin):
       def __init__(self, param1=1.0):
           self.param1 = param1

       def fit(self, X, y):
           # Training logic
           return self

       def predict(self, X):
           # Prediction logic
           return predictions
   ```

2. Add to `src/algorithms/__init__.py`:
   ```python
   from .my_algorithm import MyAlgorithm
   __all__ = [..., 'MyAlgorithm']
   ```

3. Add configuration in `config/models_config.yaml`:
   ```yaml
   my_algorithm:
     enabled: true
     name: "My Algorithm"
     class: "MyAlgorithm"
     params:
       param1: 1.0
   ```

4. Add hyperparameter tuning support in `scripts/tune_hyperparams.py` (optional)

5. Write tests in `tests/`

## Adding New Datasets

1. Create a new dataset loader in `src/datasets/`:
   ```python
   # src/datasets/my_dataset.py
   def load_my_dataset(**kwargs):
       # Load dataset logic
       return X_train, y_train, X_val, y_val, X_test, y_test, classes
   ```

2. Add to `src/datasets/dataset.py`:
   ```python
   from .my_dataset import load_my_dataset

   def get_dataset(dataset_name, **kwargs):
       if dataset_name == "my_dataset":
           return load_my_dataset(**kwargs)
       # ...
   ```

3. Update configuration schema in `src/config_schema.py` if needed

4. Write tests

## Documentation

### Code Documentation
- Use docstrings for all public functions and classes
- Follow Google or NumPy docstring style
- Include type hints

### Documentation Files
- Update relevant docs in `docs/` when adding features
- Keep README.md concise with links to detailed docs
- Add examples for new features

## Git Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Add my feature"
   ```

3. Run tests and formatting:
   ```bash
   pytest
   black src/ scripts/
   flake8 src/ scripts/
   ```

4. Push and create pull request

## Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:
- Code formatting (black)
- Linting (flake8)
- Type checking (mypy, if configured)
- Commit message validation

Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```

## Debugging

### Common Issues

1. **Import errors**: Make sure you've installed the package (`pip install -e .` or `uv sync`)
2. **Config errors**: Check that config files are valid YAML and match the schema
3. **Dataset errors**: Verify dataset download and paths
4. **Memory errors**: Use smaller datasets or reduce batch sizes

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profiling

Profile code performance:
```bash
python -m cProfile -s cumulative scripts/pipeline/03_model_training.py
```

## External Dependencies

See [Installation](installation.md) for details on optional dependencies.

## Code Review Checklist

Before submitting a PR:
- [ ] Code follows PEP 8 style guide
- [ ] Type hints added to functions
- [ ] Docstrings added to public functions
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No hardcoded paths or secrets
- [ ] Config changes are backward compatible (if possible)

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md (if exists)
3. Tag release: `git tag v1.0.0`
4. Push tags: `git push --tags`
5. Create GitHub release with notes

## Getting Help

- Check existing documentation in `docs/`
- Review code examples in `scripts/`
- Check test files for usage examples
- Open an issue for bugs or feature requests
