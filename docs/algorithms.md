# Algorithms

This document describes the algorithms used in this project, how they are implemented, and why they were chosen.

## Overview

The project implements multiple classification algorithms, ranging from traditional machine learning approaches to modern transformer-based models and Retrieval-Augmented Generation (RAG) techniques. Each algorithm is chosen for different use cases based on performance, speed, and resource requirements.

## Model Comparison

| Model | Architecture | Use Case |
|-------|--------------|---------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |
| Embedding + LogReg | Flexible embeddings (SBERT/OpenAI) + Logistic Regression | High-accuracy with flexible embedding backend |
| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | Context-aware classification with flexible LLM provider |

## 1. Multinomial Naive Bayes (`nb_tfidf`)

### Implementation

**Why**: Naive Bayes is chosen for its simplicity, speed, and effectiveness on text classification tasks. It's particularly good for baseline comparisons and real-time applications where speed is critical.

**How**:
- **Text Preprocessing**: Standard text cleaning (lowercasing, punctuation removal)
- **TF-IDF Vectorization**: Converts text to numerical features using Term Frequency-Inverse Document Frequency
- **Model Training**: Multinomial Naive Bayes classifier with Laplace smoothing
- **Inference**: Fast probabilistic classification

**Pipeline**: Text preprocessing → TF-IDF vectorization → Model training → Fast inference

**Performance**:
- Training: ~2 min/2M docs
- Inference: 60k docs/s
- Resources: < 2GB RAM, CPU-only

**Use Cases**:
- Real-time classification where speed is critical
- Baseline model for comparison
- Resource-constrained environments

## 2. Linear SVM (`svm_linear`, `svm_bigram`)

### Implementation

**Why**: Linear SVM provides an excellent balance between speed and accuracy. It's robust to overfitting and works well with high-dimensional sparse features like TF-IDF vectors.

**How**:
- **Text Preprocessing**: Standard text cleaning
- **N-gram Extraction**: Uni-grams and bi-grams capture word order and context
- **TF-IDF Vectorization**: Converts n-grams to numerical features
- **SVM Training**: Linear SVM with L2 regularization
- **Calibration**: Probability calibration enabled by default for better probability estimates

**Pipeline**: Text preprocessing → N-gram extraction → TF-IDF → SVM training

**Performance**:
- Training: 7-8 min
- Inference: 12-15k docs/s
- Resources: < 5GB RAM, CPU-only

**Use Cases**:
- Production systems requiring good accuracy and reasonable speed
- When interpretability is important (linear decision boundaries)
- Balanced performance requirements

## 3. MiniLM + Logistic Regression (`transformer_logreg`)

### Implementation

**Why**: Transformer embeddings capture semantic meaning better than bag-of-words approaches. MiniLM is chosen for its balance between quality and speed.

**How**:
- **Text Preprocessing**: Standard text cleaning
- **MiniLM Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` to generate dense vector representations
- **Dimensionality Reduction**: Optional PCA or feature selection
- **Classification**: Logistic Regression on embeddings

**Pipeline**: Text preprocessing → MiniLM embedding → Dimensionality reduction → Classification

**Performance**:
- Training: ~45 min on A10 GPU
- Inference: 1k docs/s
- Resources: 12GB GPU

**Use Cases**:
- High-accuracy requirements
- Semantic understanding is important
- When you have GPU resources available

## 4. Embedding + Logistic Regression (`embedding_logreg`)

### Implementation

**Why**: Provides flexibility in embedding backends while maintaining the same classification architecture. Allows switching between local (SBERT) and API-based (OpenAI) embeddings.

**How**:
- **Text Preprocessing**: Standard text cleaning
- **Embedding Generation**:
  - **SBERT mode** (default): Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`, no API key needed
  - **OpenAI mode**: API-based embeddings using `text-embedding-3-small`, requires `OPENAI_API_KEY`
- **Feature Scaling**: StandardScaler normalization
- **Classification**: Logistic Regression on embeddings

**Pipeline**: Text preprocessing → Embedding (SBERT local or OpenAI API) → StandardScaler → Classification

**Features**:
- **SBERT mode** (default): Local embeddings, no API key needed
- **OpenAI mode**: API-based embeddings, requires OPENAI_API_KEY
- Same architecture as MiniLM + LogReg but with flexible embedding backend

**Performance**:
- Training: ~45 min (SBERT) or depends on API rate limits (OpenAI)
- Inference: 1k docs/s (SBERT) or depends on API rate limits (OpenAI)
- Resources: 12GB GPU (SBERT) or minimal (OpenAI, API-based)

**Use Cases**:
- When you want flexibility in embedding backends
- API-based embeddings for resource-constrained environments
- Local embeddings for privacy-sensitive applications

## 5. RAG Classification Implementation (`rag_faiss`)

### Implementation

**Why**: RAG (Retrieval-Augmented Generation) combines semantic search with large language models to make context-aware classification decisions. This approach leverages both the power of dense embeddings for retrieval and LLMs for understanding context.

**How**:

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

**LLM Classification Details**:
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

**Use Cases**:
- When semantic understanding and context are critical
- Few-shot learning scenarios
- When you need explainable predictions (can show retrieved examples)
- Applications requiring high-quality classification with context awareness

## Algorithm Selection Guide

Choose an algorithm based on your requirements:

1. **Speed Critical**: Use Naive Bayes or Linear SVM
2. **Accuracy Critical**: Use MiniLM + LogReg or Embedding + LogReg
3. **Context Awareness**: Use RAG-LLM
4. **Resource Constrained**: Use Naive Bayes or Linear SVM
5. **GPU Available**: Use MiniLM + LogReg or Embedding + LogReg (SBERT mode)
6. **No GPU, API Available**: Use Embedding + LogReg (OpenAI mode)
7. **Local Only, No API**: Use Naive Bayes, Linear SVM, or RAG with Ollama

## Implementation Details

All algorithms are implemented in the `src/algorithms/` directory:
- `naive_bayes.py`: Multinomial Naive Bayes implementation
- `linear_svm.py`: Linear SVM implementation
- `transformer_logreg.py`: MiniLM + Logistic Regression
- `embedding_logreg.py`: Flexible embedding + Logistic Regression
- `openai_logreg.py`: OpenAI-specific embedding + Logistic Regression

RAG implementations are in `src/rag/`:
- `rag_kmajority.py`: k-Majority voting
- `centroid_nn.py`: Centroid-based classification
- `rag_llm.py`: LLM-based classification
- `adapter_sklearn.py`: Sklearn-compatible adapter for RAG models
