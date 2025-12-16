# Intent Classifier

## Overview

Core source code for the intent classification system.

## Structure

```
intent_classifier/
├── algorithms/     # ML algorithms
├── datasets/       # Dataset handling
├── evaluation/     # Model evaluation
├── hparam/         # Hyperparameter tuning
├── pipeline/       # Pipeline orchestration
├── prediction/     # Prediction utilities
├── rag/            # RAG implementation
└── utils/          # Utility functions
```

## Core Modules

### model.py

Core model interfaces and implementations:
- `TextClassifier` (abstract base class)
- Model implementations from `algorithms/`

### algorithms/

- `naive_bayes.py` - Multinomial Naive Bayes
- `linear_svm.py` - Linear SVM
- `transformer_logreg.py` - MiniLM + Logistic Regression
- `embedding_logreg.py` - Flexible embeddings + Logistic Regression

### rag/

- `adapter_sklearn.py` - Scikit-learn compatibility adapter
- `rag_kmajority.py` - k-Majority RAG
- `rag_llm/` - LLM-based RAG classifier
- `centroid_nn.py` - Centroid-based classifier

### datasets/

- `dataset.py` - Main dataset loading interface
- `generic_loader.py` - Generic dataset loader (HuggingFace, GitHub, CSV, JSON)

### evaluation/

- `evaluation.py` - Main evaluation functions
- `metrics_singlelabel.py` - Single-label metrics
- `metrics_multilabel.py` - Multi-label metrics
- `visualization.py` - Visualization tools

## Usage

```python
from intent_classifier.datasets.dataset import get_dataset
from intent_classifier.algorithms.linear_svm import LinearSVMClassifier

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

# Train classifier
classifier = LinearSVMClassifier()
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)
```
