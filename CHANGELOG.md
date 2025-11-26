# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-XX

### Added
- Unified pipeline runner (`scripts/pipeline/run_all.py`) with support for:
  - `CONFIG_FILE` environment variable override
  - `--tune` flag for hyperparameter tuning
  - `--save-model` flag for model persistence
  - Deterministic seed application across all random number generators
- Hyperparameter tuning module (`intent_classifier/hparam/tune.py`) with `run_tuning()` function
- Console scripts for easy CLI access:
  - `intent-train`: Run the complete training pipeline
  - `intent-classify`: Classify text with trained models
  - `rag-cli`: RAG-LLM classification CLI
  - `api-serve`: Start the FastAPI server
- Comprehensive documentation:
  - Updated README with copy-paste runnable commands
  - Configuration reference table in `docs/configuration.md`
  - CONTRIBUTING.md with development setup instructions
  - CODE_OF_CONDUCT.md
  - SECURITY.md with vulnerability reporting guidelines
- `.env.template` file for environment variable configuration
- Troubleshooting section in README

### Changed
- Consolidated pipeline execution into single `run_all.py` script
- Updated CLI interfaces to match specification:
  - `intent-classify`: Now uses `--model-path` instead of config-based loading
  - `rag-cli`: Updated to use `--provider`, `--model`, `--labels`, `--k`, `--text` arguments
  - `api-serve`: Added `cli()` function with `--host` and `--port` arguments
- Project structure documentation updated to reflect actual directory layout
- All CLI scripts now have fast `--help` (heavy imports moved inside functions)

### Removed
- Deprecated `scripts/run_all_experiments.py` (functionality merged into `run_all.py`)
- Backup file `scripts/rag_cli.py.bak`
- Empty `src/rag/` directory

### Fixed
- Test files moved from repository root to `tests/` directory
- Test file paths updated to work from `tests/` directory
- All placeholders removed from documentation
- Configuration reference table added to documentation

### Security
- Added SECURITY.md with responsible disclosure guidelines
- Environment variable template for secure configuration

[0.1.0]: https://github.com/yourusername/reuters-rag-classifier/releases/tag/v0.1.0
