# Demos

## Overview

Interactive Gradio demos for exploring the intent classification system.

## Demos

### Intent Classifier Demo

Comprehensive web interface for intent classification and semantic search.

```bash
uv run python scripts/demos/intent_classifier_demo.py
```

**Features:**
- Multiple model selection (Naive Bayes, SVM, Transformer, RAG)
- Real-time classification
- Semantic search for RAG models
- Confidence scores and top predictions

### Semantic Search Demo

Semantic search interface using vector embeddings.

```bash
uv run python scripts/demos/semantic_search_demo.py
```

**Features:**
- Search for similar utterances
- Adjustable number of results
- Similarity scores

### Intent Trend Analyzer

Analyze intent trends using semantic search and LLM analysis.

```bash
uv run python scripts/demos/intent_trend_analyzer.py
```

**Features:**
- Retrieve similar past utterances
- LLM-powered relevance classification
- Trend analysis

## Prerequisites

1. **Trained Models**: Run training pipeline first
   ```bash
   uv run python scripts/pipeline/run_all.py
   ```

2. **Embeddings**: FAISS indices (built automatically by pipeline)

3. **Dependencies**: Install with `uv sync --extra ui`

4. **LLM Access** (for trend analyzer):
   - Ollama: `ollama serve` and `ollama pull llama3.1:8b`
   - OpenAI: Set `OPENAI_API_KEY`

## Configuration

Demos automatically load configuration from `CONFIG_FILE` (default: `config/experiments/clinc150/tiny.yaml`).
