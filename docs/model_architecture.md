# Model Architecture

## Overview

Model architectures and technical specifications.

## Models

| Model | Architecture | Use Case |
|-------|--------------|----------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |
| Embedding + LogReg | Flexible embeddings (SBERT/OpenAI) + Logistic Regression | High-accuracy with flexible embedding backend |
| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | Context-aware classification |

## Technical Specifications

### Multinomial Naive Bayes

- **Implementation**: TF-IDF vectorization + probabilistic modeling
- **Performance**: ~2 min training, 60k docs/s inference, < 2GB RAM
- **Features**: Fastest inference, minimal resources, probabilistic output

### Linear SVM

- **Implementation**: Uni- & bi-gram features with L2 regularization
- **Performance**: 7-8 min training, 12-15k docs/s inference, < 5GB RAM
- **Features**: Calibrated probabilities, interpretable, robust

### MiniLM + Logistic Regression

- **Implementation**: Transformer embeddings + logistic regression
- **Performance**: ~45 min training (GPU), 1k docs/s inference, 12GB GPU
- **Features**: High accuracy, semantic understanding, GPU acceleration

### Embedding + Logistic Regression

- **Implementation**: Flexible embedding backend + logistic regression
- **Modes**: SBERT (local) or OpenAI (API)
- **Performance**: ~45 min training (SBERT), depends on API (OpenAI)
- **Features**: Flexible backend, same accuracy as MiniLM+LogReg

### RAG Models

- **Implementation**: FAISS retrieval + classification methods
- **Components**: Document embedding, FAISS index, retrieval, classification
- **Variants**:
  - **k-majority**: Majority vote on nearest neighbors
  - **Centroid**: Nearest class centroid
  - **LLM**: Context-aware LLM decision
- **Performance**: ~25 min index build, 150-200 QPS, 16GB RAM
- **Features**: Context-aware, explainable, few-shot learning

## Resource Requirements

| Model | CPU | RAM | GPU | Storage |
|-------|-----|-----|-----|---------|
| Naive Bayes | ✓ | 2GB | - | 500MB |
| Linear SVM | ✓ | 5GB | - | 1GB |
| MiniLM + LogReg | - | 8GB | 12GB | 2GB |
| RAG Models | ✓ | 16GB | - | 5GB |

## Implementation

- Algorithms: `intent_classifier/algorithms/`
- RAG components: `intent_classifier/rag/`
- All models implement scikit-learn compatible API
