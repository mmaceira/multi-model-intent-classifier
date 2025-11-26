## Running Experiments

This guide focuses on **how to run the pipeline**. For background on data splits, configuration, and algorithms, see `pipeline.md`, `experiments.md`, and `configuration.md`.

## Quick start

The CLINC150 dataset is downloaded automatically from HuggingFace on first run.

```bash
# Install dependencies
uv sync --extra all

# (Optional) Set up Ollama for RAG-LLM models
ollama serve
ollama pull llama3.1:8b

# Tiny end-to-end experiment
CONFIG_FILE=config/config_tiny_dataset.yaml python scripts/pipeline/run_all.py --tune --save-model artifacts/model.pkl
```

## Full pipeline vs. individual steps

Run the **full pipeline**:

```bash
# Using the entry point (after installation)
intent-train

# Or directly with Python
python scripts/pipeline/run_all.py
```

This executes all pipeline steps in sequence (data loading → analysis → embeddings → training → prediction → evaluation).

Run **individual steps** (advanced):

```bash
python scripts/pipeline/00_data_loading.py
python scripts/pipeline/01_exploratory_analysis.py
python scripts/pipeline/02_build_embeddings.py
python scripts/pipeline/03_model_training.py
python scripts/pipeline/04_model_prediction.py
python scripts/pipeline/05_model_evaluation.py
```

Each step expects outputs from previous steps; keep the order above. See `pipeline.md` for what each script does.

## Basic Usage in Code

```python
from intent_classifier.datasets.dataset import get_dataset

# Load CLINC150 dataset
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

# Use with any classifier
from intent_classifier.algorithms.linear_svm import LinearSVMClassifier
classifier = LinearSVMClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

## Choosing a configuration

Preconfigured experiment files:

- `config/config.yaml` – default full CLINC150 run.
- `config/config_full_dataset.yaml` – explicit full‑dataset config with its own `run_name`.
- `config/config_10_classes.yaml` – 10‑class subset (faster, still realistic).
- `config/config_25_classes.yaml` – 25‑class subset.
- `config/config_tiny_dataset.yaml` – very small smoke‑test config.

Use a different configuration file by setting `CONFIG_FILE`:

```bash
CONFIG_FILE=config/config_25_classes.yaml python scripts/pipeline/run_all.py
```

See `experiments.md` and `configuration.md` for what each config changes (class counts, sample caps, OOS behavior, etc.).

## Hyperparameters, models, and providers

- **Hyperparameter tuning**: how to run tuning and how tuned values are loaded is described in `hyperparameter_tuning.md`.
- **Which models are trained**: controlled via `config/models_config.yaml` (see `configuration.md` for structure and examples).
- **Embedding and LLM providers**: SBERT vs. OpenAI embeddings and Ollama vs. OpenAI/Anthropic LLMs are covered in `llm_providers.md`.

## Outputs and where to look

All outputs are saved in `output/{run_name}/`, where `run_name` comes from your main config:

- `data_exploration/` – dataset statistics and visualizations.
- `embeddings/` – FAISS indices and metadata.
- `models/` – trained model files.
- `predictions/` – model predictions.
- `results/` – evaluation metrics and comparison plots.

Resource tips (batch sizes, BLAS threads, RAG‑LLM settings) and failure scenarios are covered in more detail in the README troubleshooting section.
