# Experiments

## Overview

Datasets and experiment configurations used by the pipeline.

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
- Downloads automatically on first use (via the dataset loader)

See:

- `pipeline.md` for data splits, pipeline steps, and outputs
- `configuration.md` for the `dataset:` section and size controls
- `running_experiments.md` for how to run the pipeline with the preconfigured configs
