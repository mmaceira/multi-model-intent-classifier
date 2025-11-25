# Model Architecture

This document provides detailed information about the model architectures used in this project.

## Model Comparison

| Model | Architecture | Use Case |
|-------|--------------|---------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |
| Embedding + LogReg | Flexible embeddings (SBERT/OpenAI) + Logistic Regression | High-accuracy with flexible embedding backend |
| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | Context-aware classification with flexible LLM provider |

## Technical Specifications

### 1. Multinomial Naive Bayes (`nb_tfidf`)

**Implementation**: Bag-of-words TF-IDF vectorization with probabilistic modeling

**Pipeline**: Text preprocessing → TF-IDF vectorization → Model training → Fast inference

**Performance**:
- Training: ~2 min/2M docs
- Inference: 60k docs/s
- Resources: < 2GB RAM, CPU-only

**Key Features**:
- Fastest inference speed
- Minimal resource requirements
- Good baseline for comparison
- Probabilistic output

### 2. Linear SVM (`svm_linear`, `svm_bigram`)

**Implementation**: Uni- & bi-gram features with L2 regularization

**Pipeline**: Text preprocessing → N-gram extraction → TF-IDF → SVM training

**Performance**:
- Training: 7-8 min
- Inference: 12-15k docs/s
- Resources: < 5GB RAM, CPU-only

**Key Features**:
- Calibrated probability estimates (enabled by default)
- Robust to overfitting
- Good balance of speed and accuracy
- Linear decision boundaries (interpretable)

### 3. MiniLM + Logistic Regression (`transformer_logreg`)

**Implementation**: Transformer embeddings with Logistic Regression head

**Pipeline**: Text preprocessing → MiniLM embedding → Dimensionality reduction → Classification

**Performance**:
- Training: ~45 min on A10 GPU
- Inference: 1k docs/s
- Resources: 12GB GPU

**Key Features**:
- High accuracy through semantic understanding
- Dense vector representations
- GPU acceleration
- Captures semantic relationships

### 4. Embedding + Logistic Regression (`embedding_logreg`)

**Implementation**: Flexible embedding backend (SBERT or OpenAI) with Logistic Regression head

**Pipeline**: Text preprocessing → Embedding (SBERT local or OpenAI API) → StandardScaler → Classification

**Features**:
- **SBERT mode** (default): Local embeddings, no API key needed
- **OpenAI mode**: API-based embeddings, requires OPENAI_API_KEY
- Same architecture as MiniLM + LogReg but with flexible embedding backend

**Performance**:
- Training: ~45 min (SBERT) or depends on API rate limits (OpenAI)
- Inference: 1k docs/s (SBERT) or depends on API rate limits (OpenAI)
- Resources: 12GB GPU (SBERT) or minimal (OpenAI, API-based)

**Key Features**:
- Flexible embedding backend
- Can use local or API-based embeddings
- Same high accuracy as MiniLM + LogReg
- API mode requires minimal local resources

### 5. RAG Classification Implementation (`rag_faiss`)

**Implementation**: The RAG (Retrieval-Augmented Generation) classification system combines the power of semantic search with large language models to make accurate classification decisions.

#### Architecture Components

1. **Document Embedding**
   - Input documents are converted into dense vector representations
   - Uses `openai text-embedding-3-small` or `sentence-transformers/all-MiniLM-L6-v2` for high-quality embeddings
   - Embeddings capture semantic meaning and document context

2. **Context Retrieval**
   - FAISS (Facebook AI Similarity Search) is used for efficient similarity search
   - For each document, retrieves `top_k` (default: 5) most similar documents
   - Similar documents serve as contextual examples for classification
   - Optimized for speed with approximate nearest neighbor search

3. **Classification Methods**:
   - **k-Majority**: Simple majority voting among retrieved neighbors
   - **Centroid**: Uses centroid of retrieved examples for classification
   - **LLM**: Uses large language models (Ollama/OpenAI) to make context-aware decisions

#### LLM Classification Details

- Uses litellm which supports multiple LLM providers (Ollama, OpenAI, Anthropic, etc.)
- Default: Ollama ("ollama/llama3.1:8b") for local, cost-free inference
- Easy to switch to OpenAI models (e.g., "gpt-4o-mini", "gpt-4o") by changing the `model` parameter
- Provides the model with:
  - Document to classify
  - Retrieved similar documents as context
  - List of valid classification labels
- Returns JSON-formatted classification decisions

**Pipeline**: Document embedding → FAISS index → Query embedding → ANN search → Classification (k-Majority/Centroid/LLM)

**Performance**:
- Index Build: ~25 min
- Query Speed: 200 QPS (k-Majority/Centroid), 150-180 QPS (LLM)
- Resources: 16GB RAM + LLM (for LLM mode)

**Key Features**:
- Context-aware classification
- Explainable predictions (can show retrieved examples)
- Few-shot learning capability
- Flexible LLM provider support

## Model Strengths

- **Naive Bayes**: Fastest inference, suitable for real-time applications
- **Linear SVM**: Best balance of speed and accuracy
- **MiniLM + LogReg**: Highest accuracy, best for precision-critical tasks
- **RAG Models**: Best for semantic understanding and context-aware classification

## Resource Requirements

| Model | CPU | RAM | GPU | Storage |
|-------|-----|-----|-----|---------|
| Naive Bayes | ✓ | 2GB | - | 500MB |
| Linear SVM | ✓ | 5GB | - | 1GB |
| MiniLM + LogReg | - | 8GB | 12GB | 2GB |
| RAG Models | ✓ | 16GB | - | 5GB |

## Performance Metrics

### Model Performance on CLINC150

| Model | Architecture | Accuracy | Macro-F1 | Training Time | Inference Speed | Resource Usage |
|-------|--------------|----------|----------|---------------|-----------------|----------------|
| Naive Bayes | TF-IDF + Naive Bayes | - | - | ~2 min | 60k docs/s | < 2GB RAM |
| Linear SVM | TF-IDF + SVM | - | - | 7-8 min | 12-15k docs/s | < 5GB RAM |
| MiniLM + LogReg | Transformer + Logistic Regression | - | - | ~45 min | 1k docs/s | 12GB GPU |
| RAG-CentroidNN | FAISS + Nearest Neighbors | - | - | ~25 min | 200 QPS | 16GB RAM |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | - | - | ~30 min | 150-180 QPS | 16GB RAM |

## Scaling Guidance

| Method | Data Scaling | Guidance |
|--------|-------------|----------|
| Naive Bayes/SVM | Linear cost | Train on full corpus |
| Transformer + LR | Diminishing returns after ~300k docs | Cap training set |
| RAG FAISS | Linear indexing | Index everything, cap generation |
| Cross-Encoder | Linear infer-time cost | Re-rank only top-k passages |

### Performance Optimization

- **Batch Processing**: Process documents in batches for higher throughput
- **Caching**: Implement results caching for frequent queries
- **Quantization**: Use quantized models for lower memory footprint
- **Hybrid Approach**: Use lightweight models for first-pass filtering

## Implementation Details

All models are implemented in the `src/` directory:
- `src/algorithms/`: Traditional ML algorithms (Naive Bayes, SVM, etc.)
- `src/rag/`: RAG implementations (k-Majority, Centroid, LLM)
- `src/embeddings/`: Embedding generation utilities
- `src/model.py`: Core model interface and utilities

Each model implements a consistent interface compatible with scikit-learn's API, making them easy to use and compare.
