# Experiments

## Overview

Available datasets, data splits, and experiment configs.

## Datasets

### CLINC150

- **Source**: HuggingFace `clinc_oos` (config `plus`)
- **Content**: 150 intents, single-label
- **Size**: ~22,500 examples
- Downloads automatically on first use

### NLU++

- **Source**: GitHub `PolyAI-LDN/task-specific-datasets`
- **Content**: 68 intents, multi-label
- **Size**: ~25,715 examples
- Downloads automatically on first use

## Data Splits

Standard ML practice:
- **Train**: Fit models
- **Validation**: Hyperparameter tuning and model selection
- **Test**: Final evaluation only

Training and tuning never touch the test set. Some analysis scripts merge train+val for larger corpus, but training always uses proper splits.

## Configuration

Control dataset size via `dataset:` section:

- `name`: `"clinc150"` or `"nlu_plus"`
- `use_oos`: Include OOS examples (CLINC150 only)
- `max_classes`: Limit number of intents
- `max_train_samples`, `max_test_samples`, `max_val_samples`: Cap examples per split

## Preconfigured Configs

- `config/dataset/clinc150/default.yaml` - Full dataset
- `config/dataset/clinc150/tiny.yaml` - Quick test
- `config/dataset/nlu_plus/default.yaml` - Full dataset
- `config/dataset/nlu_plus/tiny.yaml` - Quick test

## Outputs

Exploratory analysis generates:
- Class distributions
- Text length statistics
- Vocabulary analysis
- Train/test comparisons

Saved to `output/{run_name}/data_exploration/`.
