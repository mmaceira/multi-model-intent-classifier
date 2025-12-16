## Configuration

## Overview

Configuration is layered to reduce duplication and make experiments reproducible:

1. **Base defaults** – generic knobs and sensible defaults
2. **Providers** – LLM and embedding providers + default model ids
3. **Datasets** – dataset‑specific settings (labels, splits, limits)
4. **Experiments** – small overrides per dataset/variant

At load time, these layers are **deep‑merged** into a single config, and a `resolved`
section is attached with effective model and path choices.

## File layout

- **Base config**
  - `config/base/defaults.yaml` – global defaults and model knobs (no provider ids, no paths)
  - `config/base/providers.yaml` – endpoints + default provider model ids
  - `config/base/prompts.yaml` – prompt styles and templates for RAG/LLM components

- **Dataset config**
  - `config/datasets/{dataset}.yaml`
  - Contains only dataset‑specific information, for example:
    - Dataset name and source
    - Single‑ vs multi‑label (`multilabel: true/false`)
    - OOS usage, max classes, sample caps, language hints

- **Experiment config**
  - `config/experiments/{dataset}/{variant}.yaml`
  - Contains minimal overrides for a concrete run, for example:
    - `model.embedding_backend` (`"sbert"`, `"openai"`, `"ollama"`)
    - `model.llm_backend` (`"openai"`, `"ollama"`, or `"none"`)
    - `model.rag_top_k`
    - `analysis.*` flags (e.g. `analysis.threshold_sweep`)

- **Models config**
  - `config/algorithm/models_config.yaml`
  - Controls which algorithms are trained and how they read from the main config
    (using string interpolation like `${model.embedding_backend}` or `${resolved.llm_model}`).

## Layering and merge order

The loader always merges configs in this order:

1. `config/base/defaults.yaml`
2. `config/base/providers.yaml`
3. `config/datasets/{dataset}.yaml`
4. Selected experiment config

Later layers override earlier ones. Dictionaries are merged recursively; lists are
replaced as whole lists (no concatenation).

String interpolation (e.g. `${model.embedding_backend}`) happens **after** merging, so
references can point across layers.

## `resolved` section

After merging, the loader computes:

- **`resolved.run_id`**
  - Pattern: `<label_type>/<dataset>/<variant>`
  - Example: `multilabel/nlu_plus/tiny`

- **`resolved.label_type`**
  - Inferred from dataset settings (e.g. `dataset.multilabel`)

- **`resolved.paths`**
  - Derived from `run_id`:
    - Root: `output/runs/<run_id>/`
    - Mirrors to top‑level `config["paths"]` for convenience

- **`resolved.embedding_model`**
  - Uses `model.embedding_backend` to pick from provider defaults:
    - `"sbert"` → `model.sbert_model`
    - `"openai"` → `providers.openai.embed_default`
    - `"ollama"` → `providers.ollama.embed_default`

- **`resolved.llm_model`**
  - Uses `model.llm_backend` to pick from provider defaults:
    - `"openai"` → `providers.openai.llm_default`
    - `"ollama"` → `providers.ollama.llm_default`
    - `"none"` → `null`

- **Provider flags**
  - `resolved.ollama_endpoint`
  - `resolved.openai_api_key_present` (boolean)

Legacy overrides such as `model.llm_model` are no longer supported; use
`model.llm_backend` and provider defaults instead.

## Selecting configs via environment

You can select the experiment either by **explicit config file** or by
**dataset/variant shorthand**:

- **Highest priority**: `CONFIG_FILE`

  ```bash
  CONFIG_FILE=config/experiments/nlu_plus/tiny.yaml \
    uv run python scripts/pipeline/run_all.py
  ```

- **Preferred shorthand**: `DATASET` + `VARIANT`

  ```bash
  DATASET=nlu_plus VARIANT=tiny \
    uv run python scripts/pipeline/run_all.py
  ```

If both are set, `CONFIG_FILE` wins. The shorthand resolves to
`config/experiments/{DATASET}/{VARIANT}.yaml` and automatically infers the matching
dataset config under `config/datasets/`.

## Other environment variables

- **`OPENAI_API_KEY`** – Required for OpenAI LLM/embeddings
- **`SEED`** – Optional override for the random seed
- **`SBERT_BATCH`, `OPENAI_BATCH`** – Embedding batch sizes

For more details on legacy configs and how models read from the merged config, see
`configuration.md` and `config/algorithm/models_config.yaml`.
