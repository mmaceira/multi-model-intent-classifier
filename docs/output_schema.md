## Output Schema

## Overview

Pipeline runs write into a single, self‑contained folder per experiment under:

- **Pattern**: `output/runs/<label_type>/<dataset>/<variant>/`
- **Example**: `output/runs/multilabel/nlu_plus/tiny/`

Each run directory is **read‑only after completion** and contains everything needed to
reproduce results (configs, metrics, models, logs).

## Top‑level layout

Inside `output/runs/<run_id>/` the layout is:

- **`README.md`**: Human‑friendly overview of the run (dataset, models, top metrics, pointers)
- **`manifest.json`**: Flat list of all files in the run with size and hash
- **`meta/`**: Reproducibility and provenance metadata
- **`dataset/`**: Dataset‑level analysis and label statistics
- **`features/`**: Embeddings and other feature artifacts
- **`models/`**: One subfolder per trained model
- **`eval/`**: Model‑level evaluation artifacts
- **`compare/`**: Cross‑model comparison tables and plots
- **`llm_logs/`**: Normalized RAG / LLM request logs

## `meta/` – Reproducibility package

- **`meta/config_resolved.yaml`**: Fully merged configuration, including the `resolved` section
- **`meta/config_sources.json`**: List of loaded config files + SHA256 for each
- **`meta/command.txt`**: Original CLI invocation and key environment selections
- **`meta/git.json`**: Commit hash, branch, and dirty flag at run time
- **`meta/env.json`**: Python version, OS, and dependency snapshot
- **`meta/timestamps.json`**: Start/end timestamps and durations per pipeline stage

These files let you re‑create a run by re‑using the exact configs and environment.

## `dataset/` – Dataset analysis

Typical content:

- **Label statistics** (counts, support, imbalance indicators)
- **Label co‑occurrence matrix** (e.g. `label_cooccurrence.parquet`)
- Optional **hardest examples** tables (without raw text by default; uses hashes)

Everything in `dataset/` is **model‑agnostic** and derived from ground‑truth labels.

## `features/` – Embeddings and other features

Features are grouped first by type, then backend:

- **`features/embeddings/<backend>/`**
  - FAISS indices
  - Embedding metadata (`meta.jsonl`, stats)

Backends include e.g. `sbert`, `openai`, or `ollama`. The resolved embedding backend is
recorded under `resolved.embedding_model` in `meta/config_resolved.yaml`.

## `models/` – Trained models and cards

Each trained model lives under a stable, slugified id:

- **Pattern**: `models/<model_id>/`
  - Serialized model artifacts (pickles, vectorizers, scalers, checkpoints)
  - **`model_card.md`** – Auto‑generated model card
  - **`metadata.json`** – Display name, model class, config snippet, hashes

The `model_id` is lowercase with underscores, e.g.:

- `"TF-IDF bigrams + SVM"` → `tfidf_bigrams_svm`
- `"MiniLM + LogReg"` → `minilm_logreg`

## `eval/` – Per‑model metrics and analysis

- **Pattern**: `eval/<model_id>/`
  - Summary metrics (JSON / small CSV)
  - Per‑label metrics tables
  - Confusion matrices or equivalent plots
  - Optional **PR curves** (`pr_summary.json`, plots under `figures/`)
  - Optional **threshold sweep** (`threshold_sweep.parquet` + plot)
  - Optional **error analysis** under `error_analysis/`

Evaluation artifacts use compact binary formats (e.g. Parquet) for large tables and CSV
only for small summaries.

## `compare/` – Cross‑model comparison

Typical content:

- **`summary_metrics.*`**: One row per model with key metrics
- Comparison plots under `compare/figures/`
- Any additional tables used to pick a “best” model per label or metric

This is the main entry point when scanning a run to see which model performs best.

## `llm_logs/` – RAG / LLM request logs

LLM and RAG traffic is normalized into JSONL files:

- **`llm_logs/rag_llm/requests.jsonl`** – One record per successful request
- **`llm_logs/rag_llm/failures.jsonl`** – One record per failed request

Each record includes:

- Timestamp, provider, and model
- Prompt style and retrieved example ids
- Prompt/response sizes (characters or tokens)
- Latency and success/error flags

Use these logs to analyze latency, cost, and failure modes of LLM‑based models.
