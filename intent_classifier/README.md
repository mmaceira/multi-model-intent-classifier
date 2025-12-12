# Source Code Documentation

This directory contains the core source code for the CLINC150 RAG Classifier project. The code is organized into several modules, each responsible for a specific aspect of the system.

## Directory Structure

```
intent_classifier/
├── algorithms/     # Machine learning algorithms implementation
├── datasets/       # Dataset handling (generic loader system)
├── evaluation/     # Model evaluation and metrics
├── hparam/         # Hyperparameter tuning strategies
├── pipeline/        # Pipeline orchestration
├── prediction/      # Prediction utilities (single-label and multi-label)
├── rag/            # RAG implementation and utilities
├── utils/          # Utility functions and helpers
├── exploration.py   # Data exploration and visualization
├── model.py         # Core model interfaces and implementations
└── training.py     # Training utilities and pipeline
```

## Core Modules

### model.py

The central model definitions and interfaces:

- `TextClassifier` (abstract base class): Core interface for all text classifiers
- `NaiveBayesClassifier`: Naive Bayes classifier implementation (from `algorithms.naive_bayes`)
- `LinearSVMClassifier`: Support Vector Machine classifier implementation (from `algorithms.linear_svm`)
- `TransformerLogReg`: Transformer-based classifier implementation (from `algorithms.transformer_logreg`)
- `EmbeddingLogReg`: Embedding-based classifier implementation (from `algorithms.embedding_logreg`)

```python
# Example usage
from intent_classifier.model import TextClassifier
from intent_classifier.algorithms.linear_svm import LinearSVMClassifier

# Create a classifier instance
classifier = LinearSVMClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
```

### prediction/

Handles all aspects of model inference and prediction for both single-label and multi-label classification:

- `base.py`: Base prediction utilities
- `singlelabel.py`: Single-label prediction functions
- `multilabel.py`: Multi-label prediction functions

The prediction module provides utilities for loading models and making predictions, with separate handling for single-label and multi-label scenarios.

### training.py

Manages model training and hyperparameter optimization:

- `train_model`: High-level training function
- `ModelTrainer`: Configurable training pipeline
- `TrainingConfig`: Configuration dataclass
- `evaluate_during_training`: Training-time evaluation

```python
# Example usage
from intent_classifier.training import train_model
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train model
model, metrics = train_model(
    X_train, y_train,
    X_val, y_val,
    model_type="svm_linear"
)
```

### exploration.py

Tools for dataset exploration and visualization:

- `DataExplorer`: Main class for data exploration
- `visualize_class_distribution`: Topic visualization
- `analyze_text_lengths`: Text length analysis
- `generate_wordclouds`: Word cloud generation
- `export_analysis`: Export analysis results

## Module Documentation

### utils/

Utility functions and helpers:

- `config_loader.py`: Configuration loading and management with YAML support
- `embeddings.py`: Embedding generation utilities (SBERT, OpenAI)
- `file_ops.py`: File handling and IO operations
- `label_utils.py`: Label processing utilities (single-label, multi-label)
- `model_factory.py`: Factory for creating model instances
- `model_loader.py`: Model loading utilities with hyperparameter support
- `model_registry.py`: Model registry for managing available models
- `model_utils.py`: Model utility functions
- `paths.py`: Path resolution utilities
- `retry.py`: Retry logic for API calls
- `seed.py`: Random seed management
- `text_utils.py`: Text processing utilities
- `types.py`: Type definitions
- `warnings_config.py`: Warning configuration

### datasets/

Dataset handling using a generic loader system:

- `dataset.py`: Main dataset loading interface with automatic discovery of datasets from config files
- `generic_loader.py`: Generic dataset loader that supports multiple source types (HuggingFace, GitHub JSON, CSV, local JSON)

Datasets are automatically discovered from `config/dataset/{dataset_name}/loader.yaml` files. Supported datasets include:
- `clinc150`: Single-label intent classification (from HuggingFace)
- `nlu_plus`: Multi-label intent classification (from GitHub)

### utils/embeddings.py

Embedding generation and management utilities:

- Embedding generation using SBERT (local) or OpenAI (API)
- FAISS index management for efficient similarity search
- Embedding utilities for RAG models

### algorithms/

Contains implementations of various machine learning algorithms used for classification:

- `naive_bayes.py`: Multinomial Naive Bayes implementation with TF-IDF
- `linear_svm.py`: Linear SVM implementation (unigrams and bigrams) with calibration support
- `transformer_logreg.py`: Transformer embeddings (MiniLM) + Logistic Regression
- `embedding_logreg.py`: Flexible embedding backends (SBERT/OpenAI) + Logistic Regression

### rag/

RAG implementation and utilities:

- `adapter_sklearn.py`: Scikit-learn compatibility adapter for RAG models
- `rag_kmajority.py`: K-Majority RAG implementation (majority voting on retrieved neighbors)
- `rag_llm/`: LLM-based RAG classifier with support for multiple LLM providers (Ollama, OpenAI, etc.)
  - `classifier.py`: Main RAG-LLM classifier implementation
  - `prompts/`: Prompt templates for single-label and multi-label classification
- `centroid_nn.py`: Centroid-based nearest neighbors classifier
- `build_index.py`: FAISS index building utilities
- `vector_store.py`: Vector store implementation
- `retrieval.py`: Document retrieval system
- `classifier_base.py`: Base class for RAG classifiers

### evaluation/

Model evaluation and metrics with support for both single-label and multi-label classification:

- `base.py`: Base evaluation utilities
- `evaluation.py`: Main evaluation functions
- `metrics.py`: Core metrics calculation
- `metrics_singlelabel.py`: Single-label specific metrics
- `metrics_multilabel.py`: Multi-label specific metrics
- `singlelabel.py`: Single-label evaluation functions
- `multilabel.py`: Multi-label evaluation functions
- `utils.py`: Evaluation utilities
- `visualization.py`: Visualization tools for evaluation results

## Design Patterns

The codebase implements several design patterns to ensure maintainability and extensibility:

1. **Factory Pattern**: Used in `model.py` to create different classifier types
2. **Strategy Pattern**: Used for different vectorization strategies
3. **Adapter Pattern**: Used in RAG to adapt between different APIs
4. **Singleton Pattern**: Used for configuration and logging
5. **Builder Pattern**: Used for complex pipeline construction

## Code Style Guide

1. **Naming Conventions**
   - Use snake_case for variables and functions
   - Use PascalCase for classes
   - Use UPPER_CASE for constants
   - Prefix private members with underscore

2. **Documentation**
   - Include docstrings for all public functions and classes
   - Use type hints for function parameters and return values
   - Document complex algorithms with comments

3. **Code Structure**
   - Follow the Single Responsibility Principle
   - Keep functions focused and concise
   - Use meaningful variable names
   - Limit line length to 100 characters
