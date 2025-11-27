## Configuration

This project uses **YAML files + environment variables** to control datasets, models, and experiments. Most users only need to edit one of the config files in `config/` and optionally set a few env vars.

## Main config (`config/config.yaml`)

The main config defines:

- **General**: `general.run_name`, `general.seed`.
- **Dataset**: `dataset.name`, `dataset.use_oos`, and optional limits such as `dataset.max_classes`, `dataset.max_train_samples`, `dataset.max_test_samples`.
- **Paths**: where to store embeddings, models, predictions, and results (all relative to the repo root and typically using `${general.run_name}`).
- **Model**:
  - `model.embedding_backend`: `"sbert"` or `"openai"`.
  - `model.sbert_model_name`, `model.openai_model_name`.
  - `model.classifier`: e.g. `"linear_svm"`, `"naive_bayes"`, `"transformer_logreg"`, or a RAG adapter.
  - `model.rag_top_k`: number of neighbors for RAG models.
  - `model.llm_model`: identifier for the default LLM used in RAG‑LLM.

Environment variables can override some of these (e.g. `SEED`, `MODEL_TYPE`, `RAG_K`, `MODEL_ID`, `DATASET_NAME`, `CONFIG_FILE`).

## Model selection (`config/models_config.yaml`)

`models_config.yaml` controls **which models are actually run** and with what adapter classes:

- Each entry under `models:` has:
  - `enabled`: turn a model on/off.
  - `name`: human‑readable label for reports.
  - `class`: Python class name (e.g. `NaiveBayesClassifier`, `RagSklearnAdapter`).
  - Optional `params`: model‑specific arguments such as `method`, `top_k`, `model`, `use_openai`.
- String interpolation lets you reuse values from `config.yaml`, e.g. `top_k: "${model.rag_top_k}"`, `model: "${model.llm_model}"`.

Typical workflow:
- Enable/disable models by toggling `enabled`.
- Adjust RAG parameters in `params` while keeping base defaults in `config.yaml`.

## Multiple experiment configs

The repo ships with several configs such as:

- `config/config.yaml`: default full‑dataset run.
- `config/config_10_classes.yaml`, `config/config_25_classes.yaml`: smaller‑class experiments.
- `config/config_tiny_dataset.yaml`: very small setup for quick tests.

To switch configs, set `CONFIG_FILE`:

```bash
CONFIG_FILE=config_tiny_dataset.yaml python scripts/pipeline/run_all.py
```

## Datasets and hyperparameters

- Dataset‑related options (class limits, sample caps, OOS behavior) live under `dataset:` and are documented in more detail in `experiments.md`.
- Tuned hyperparameters are loaded from `config/hyperparameters/{config_name}/`. See `hyperparameter_tuning.md` for how these files are created and used.

## Environment variables (summary)

Common env vars:

- **Config and dataset**: `CONFIG_FILE`, `DATASET_NAME`, `SEED`.
- **Embedding / LLM**: `OPENAI_API_KEY` (required for OpenAI features), `LLM_MODEL`.
- **API server**: `CORS_ORIGINS`, `MODEL_CACHE_SIZE`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`.
- **Embedding batch sizes and threading**: `SBERT_BATCH`, `OPENAI_BATCH`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.
- **Testing / tracking**: `TEST_REAL_APIS`, `MLFLOW_TRACKING_URI`.

The configuration layer is validated with Pydantic schemas, so invalid or missing fields should produce clear error messages when you start the pipeline or API.
