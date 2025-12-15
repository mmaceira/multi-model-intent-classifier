## Classification CLI

### Model overview: embeddings and features

The main models shipped in this project use the following embeddings / features:

- **Classic TF-IDF models**
  - Naive Bayes: TF-IDF (unigrams)
  - Linear SVM: TF-IDF (unigrams)
  - TF-IDF bigrams + SVM: TF-IDF (unigrams + bigrams)
- **Transformer / dense embeddings**
  - MiniLM + LogReg: MiniLM SentenceTransformer embeddings
  - Embedding + LogReg (SBERT): SBERT SentenceTransformer embeddings
  - Embedding + LogReg (Qwen/Ollama): Qwen/Ollama embedding model via HTTP
- **RAG over embeddings**
  - RAG-kMajority (SBERT): SBERT FAISS index (`sbert/index.faiss`, `meta.jsonl`)
  - RAG-kMajority (Qwen/Ollama): Ollama/Qwen FAISS index if present, else SBERT
  - RAG-CentroidNN: SBERT FAISS index (centroids per label)
- **RAG-LLM (retrieval + LLM via Qwen/Ollama)**
  - RAG-LLM (TF-IDF, default/short/n8n): TF-IDF bi-gram retriever over raw texts
  - RAG-LLM (SBERT embeddings, default prompt): SBERT FAISS index
  - RAG-LLM (Qwen embeddings, default prompt): Ollama/Qwen FAISS index if present, else SBERT
  - RAG-LLM (OpenAI-embeddings): OpenAI embeddings + OpenAI LLM (disabled by default)

### Overview

Use the `intent-classify` command to run intent classification with any trained model (single-label or multi-label).

Models are saved under:

- `output/singlelabel/{dataset}/{config}/models/{Model Name}/`
- `output/multilabel/{dataset}/{config}/models/{Model Name}/`

### Quick Examples

```bash
# Single-label (CLINC150 tiny, Linear SVM)
uv run intent-classify \
  --model-path "output/singlelabel/clinc150/tiny/models/Linear SVM/" \
  --text "what's my account balance?"

# Multi-label (NLU+ tiny, RAG-CentroidNN)
uv run intent-classify \
  --model-path "output/multilabel/nlu_plus/tiny/models/RAG-CentroidNN/" \
  --text "check my balance and block my card"
```

### Usage

```bash
uv run intent-classify --model-path PATH [--text TEXT | --batch-file FILE] [--output {json,text}]
```

#### Arguments

- `--model-path` (required): Path to a saved model file (`.pkl` / `.joblib`) or a model directory containing `model.pkl`.
- `--text`: Single text to classify (mutually exclusive with `--batch-file`).
- `--batch-file`: Path to a file with one text per line (mutually exclusive with `--text`).
- `--output`: Output format (`json` default, or `text`).

### More Examples

- **Single text, JSON output (default)**:

  ```bash
  uv run intent-classify \
    --model-path "output/singlelabel/clinc150/tiny/models/MiniLM + LogReg/" \
    --text "reset my password"
  ```

- **Single text, text output**:

  ```bash
  uv run intent-classify \
    --model-path "output/multilabel/nlu_plus/tiny/models/Embedding + LogReg/" \
    --text "close my account and stop card payments" \
    --output text
  ```

- **Batch classification from file**:

  ```bash
  uv run intent-classify \
    --model-path "output/singlelabel/clinc150/tiny/models/Linear SVM/" \
    --batch-file queries.txt
  ```

- **Using a direct `.pkl` file**:

  ```bash
  uv run intent-classify \
    --model-path "output/singlelabel/clinc150/tiny/models/Linear SVM/model.pkl" \
    --text "what is my account balance?"
  ```
