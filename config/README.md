# Configuration

## Overview

Configuration files organized by dataset and algorithm.

## Structure

```
config/
├── dataset/          # Dataset configuration files
│   ├── clinc150/     # CLINC150 configs
│   └── nlu_plus/     # NLU++ configs (multi-label)
└── algorithm/        # Algorithm configuration
    ├── models_config.yaml   # Model selection
    └── hyperparameters/     # Tuned hyperparameters
```

## Usage

```bash
# Single-label
uv run python scripts/pipeline/run_all.py --config config/dataset/clinc150/tiny.yaml

# Multi-label
uv run python scripts/pipeline/run_all.py --config config/dataset/nlu_plus/tiny.yaml
```

## Datasets

- **clinc150**: Single-label intent classification (150 intents)
- **nlu_plus**: Multi-label intent classification (68 intents)

## Algorithms

- **models_config.yaml**: Controls which models are trained
- **hyperparameters/**: Contains tuned hyperparameters organized by config name
