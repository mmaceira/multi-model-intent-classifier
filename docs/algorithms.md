# Algorithms

## Overview

High-level overview of models: what they do, how they represent text, and when to use them.

- **Multinomial Naive Bayes**: Bag-of-words baseline
- **Linear SVM**: Linear classifier on TF-IDF features
- **MiniLM + Logistic Regression**: Transformer embeddings + classifier
- **Embedding + Logistic Regression**: Pluggable embedding backends + classifier
- **RAG models**: Retrieval-augmented, context-aware classification

## Models

### 1. Multinomial Naive Bayes

- **Core idea**: TF-IDF vectors with conditional independence assumption
- **Text representation**: Bag-of-words, word frequencies
- **Use when**: Fast baseline needed, resources limited, classes have strong keywords

### 2. Linear SVM

- **Core idea**: Linear decision boundary on TF-IDF features
- **Text representation**: Uni-grams or bi-grams with TF-IDF
- **Use when**: Balanced speed and accuracy needed, interpretability important

### 3. MiniLM + Logistic Regression

- **Core idea**: Transformer embeddings + logistic regression
- **Text representation**: Dense semantic embeddings
- **Use when**: High accuracy needed, semantic understanding important

### 4. Embedding + Logistic Regression

- **Core idea**: Pluggable embedding backends + logistic regression
- **Text representation**: SBERT (local) or OpenAI (API) embeddings
- **Use when**: Need flexibility in embedding backend, trade latency/cost/privacy

### 5. RAG Models

- **Core idea**: Retrieve similar examples, classify with context
- **Text representation**: Dense embeddings + FAISS index
- **Variants**:
  - **k-majority**: Majority vote on nearest neighbors
  - **Centroid**: Classify by nearest class centroid
  - **RAG-LLM**: LLM makes decision based on retrieved context
- **Use when**: Context-aware classification needed, explanations important, few-shot learning

## Implementation

- Algorithms: `intent_classifier/algorithms/`
- RAG components: `intent_classifier/rag/`
