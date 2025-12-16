# Configuration

## Overview

Configuration files organized by dataset and algorithm.

## Structure

```
config/
├── base/             # Global defaults, providers, prompts
├── datasets/         # Dataset-level configuration (one YAML per dataset, with loader section)
├── experiments/      # Experiment variants per dataset
└── algorithm/        # Algorithm configuration
    ├── models_config.yaml   # Model selection
    └── hyperparameters/     # Tuned hyperparameters
```

## Usage

```bash
# Single-label
uv run python scripts/pipeline/run_all.py --config config/experiments/clinc150/tiny.yaml

# Multi-label
uv run python scripts/pipeline/run_all.py --config config/experiments/nlu_plus/tiny.yaml
```

## Datasets

- **clinc150**: Single-label intent classification (150 intents)
- **nlu_plus**: Multi-label intent classification (68 intents)

## Algorithms

- **models_config.yaml**: Controls which models are trained
- **hyperparameters/**: Contains tuned hyperparameters organized by config name
