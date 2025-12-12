# Configuration Files

This directory contains configuration files organized by dataset and algorithm.

## Structure

```
config/
├── dataset/          # Dataset configuration files
│   ├── clinc150/     # CLINC150 dataset configs
│   │   ├── tiny.yaml    # Quick testing config (limited samples/classes)
│   │   └── default.yaml # Standard config (full dataset)
│   ├── nlu_plus/    # NLU++ dataset (always multi-label)
│   │   ├── default.yaml # Standard config (full dataset)
│   │   └── tiny.yaml    # Quick testing config (limited samples)
│   └── tandem_go/    # Tandem GO dataset (always multi-label, CSV-based)
│       ├── default.yaml # Standard config (full dataset)
│       └── tiny.yaml    # Quick testing config (limited samples)
│
├── algorithm/        # Algorithm configuration files
│   ├── models_config.yaml   # Model selection and hyperparameters
│   └── hyperparameters/      # Best hyperparameters from tuning
│       └── config_tiny_dataset/
│           └── best_*.yaml
```

## Usage

Specify the config file using the `CONFIG_FILE` environment variable:

```bash
# Dataset config examples
CONFIG_FILE=config/dataset/clinc150/tiny.yaml python scripts/pipeline/run_all.py
CONFIG_FILE=config/dataset/clinc150/default.yaml intent-train
CONFIG_FILE=config/dataset/nlu_plus/default.yaml intent-train
CONFIG_FILE=config/dataset/nlu_plus/tiny.yaml python scripts/pipeline/run_all.py
CONFIG_FILE=config/dataset/tandem_go/default.yaml intent-train
CONFIG_FILE=config/dataset/tandem_go/tiny.yaml python scripts/pipeline/run_all.py
```

## Datasets

- **clinc150**: Single-label intent classification dataset (150 intents)
- **nlu_plus**: Always multi-label (68 intents across banking and hotels domains)
- **tandem_go**: Always multi-label (CSV-based dataset with tags extracted from name and description fields)

## Algorithms

- **models_config.yaml**: Controls which models are trained and their configuration
- **hyperparameters/**: Contains tuned hyperparameters organized by config name
