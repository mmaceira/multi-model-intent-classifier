## Classification CLI

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
