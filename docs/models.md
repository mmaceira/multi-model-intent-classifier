## Models

## Overview

This project includes several **model families** that trade off speed, accuracy, and
explainability. Use this page as a “when to use what” guide; for implementation details
see `algorithms.md` and `model_architecture.md`.

## Model families

- **Bag‑of‑words baselines**
  - **Multinomial Naive Bayes**
  - **Linear SVM**
  - **TF‑IDF bigrams + SVM**
  - **When to use**: Fast baselines, low‑resource environments, sanity checks before
    moving to embeddings or LLMs.

- **Embedding + linear head**
  - **MiniLM + Logistic Regression**
  - **SBERT / OpenAI / Ollama embeddings + Logistic Regression**
  - **When to use**: Strong accuracy on both single‑ and multi‑label tasks, good
    balance of performance and cost, production‑ready semantic understanding.

- **Classical RAG classifiers**
  - **RAG‑CentroidNN**
  - **RAG‑kMajority (SBERT / Qwen)**
  - **When to use**: You want **explanations** via nearest neighbors, few‑shot
    behavior, and simple, controllable behavior without an LLM in the loop.

- **RAG‑LLM classifiers**
  - **RAG‑LLM (TF‑IDF)**
  - **RAG‑LLM (SBERT / Qwen embeddings)**
  - **When to use**: Rich, context‑aware decisions, natural‑language explanations, and
    flexible prompting – at the cost of higher latency and LLM usage.

## Choosing a model

- **For production baselines**
  - Start with **Linear SVM** or **TF‑IDF bigrams + SVM**.
  - Add **MiniLM + LogReg** or **SBERT + LogReg** when you need better accuracy and
    semantic robustness.

- **For explainability and auditability**
  - Prefer **RAG‑CentroidNN** or **RAG‑kMajority**.
  - These expose retrieved examples and class centroids, which are easy to inspect and
    reason about.

- **For complex, nuanced intents**
  - Use **RAG‑LLM** variants when intents are ambiguous, under‑represented, or require
    subtle textual cues.
  - Combine with prompt styles (`default`, `short`, `n8n_prompt`) to tune cost vs.
    quality.

- **For experimentation**
  - Run multiple families in the same experiment; use `compare/summary_metrics.*` in
    the run output to pick a default model.
  - Use the generated **model cards** under `models/<model_id>/model_card.md` to
    understand trade‑offs for each model in context.
