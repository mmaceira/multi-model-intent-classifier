# Source Code Documentation

This directory contains the core source code for the CLINC150 RAG Classifier project. The code is organized into several modules, each responsible for a specific aspect of the system.

## Directory Structure

```
src/
├── algorithms/     # Machine learning algorithms implementation
├── datasets/       # Dataset handling and preprocessing
├── embeddings/     # Embedding generation and management
├── evaluation/     # Model evaluation and metrics
├── exploration.py  # Data exploration and visualization
├── model.py        # Core model interfaces and implementations
├── prediction.py   # Prediction utilities and inference pipeline
├── rag/            # RAG implementation and utilities
├── training.py     # Training utilities and pipeline
└── utils/          # Utility functions and helpers
```

## Core Modules

### model.py

The central model definitions and interfaces:

- `TextClassifier` (abstract base class): Core interface for all text classifiers
- `TextClassifier`: Main classifier implementation with model factory functionality
- `NBClassifier`: Naive Bayes classifier implementation
- `SVMClassifier`: Support Vector Machine classifier implementation
- `BERTClassifier`: Transformer-based classifier implementation

```python
# Example usage
from src.model import TextClassifier
from src.algorithms.linear_svm import LinearSVMClassifier

# Create a classifier instance
classifier = LinearSVMClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
```

### prediction.py

Handles all aspects of model inference and prediction:

- `classify_text`: End-to-end text classification function
- `TextPredictionPipeline`: Reusable prediction pipeline
- `ModelRegistry`: Factory pattern for model instantiation
- `ModelCache`: Caching system for efficient model loading

```python
# Example usage
from src.prediction import classify_text

# Classify a single document
result = classify_text("What is the weather today?", model_type="bert_lr")
print(f"Predicted class: {result['class']}")
print(f"Confidence: {result['confidence']}")
```

### training.py

Manages model training and hyperparameter optimization:

- `train_model`: High-level training function
- `ModelTrainer`: Configurable training pipeline
- `TrainingConfig`: Configuration dataclass
- `evaluate_during_training`: Training-time evaluation

```python
# Example usage
from src.training import train_model
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

- `logging.py`: Logging configuration with rotating file handlers
- `config.py`: Configuration management with YAML support
- `file_utils.py`: File handling and IO operations
- `time_utils.py`: Time measurement and benchmarking
- `nlp_utils.py`: NLP-specific utilities

### datasets/

Dataset handling and preprocessing:

- `clinc150.py`: CLINC150 intent classification dataset loader
- `dataset.py`: Main dataset loading interface
- `preprocessing.py`: Text preprocessing pipeline with multiple cleaning options
- `augmentation.py`: Data augmentation techniques for expanded training
- `validation.py`: Data validation and quality checking tools

### embeddings/

Embedding generation and management:

- `transformer.py`: Transformer-based embeddings with model providers
- `faiss_index.py`: FAISS index management for efficient similarity search
- `embedding_utils.py`: Utility functions for embedding manipulation
- `cache.py`: Caching system for embedding reuse and persistence

### algorithms/

Contains implementations of various machine learning algorithms used for classification:

- `naive_bayes.py`: Multinomial Naive Bayes implementation with custom smoothing
- `svm.py`: Support Vector Machine with optimized hyperparameters
- `logistic_regression.py`: Logistic Regression with transformer embeddings
- `ensemble.py`: Ensemble methods including voting and stacking

### rag/

RAG implementation and utilities:

- `__init__.py`: Module initialization and configuration
- `adapter_sklearn.py`: Scikit-learn compatibility adapter
- `rag_kmajority.py`: K-Majority RAG implementation
- `rag_llm.py`: LLM-based RAG implementation
- `centroid_nn.py`: Centroid-based nearest neighbors
- `build_index.py`: Index building utilities
- `vector_store.py`: Vector store implementation
- `retrieval.py`: Document retrieval system
- `classifier_base.py`: Base class for RAG classifiers

### evaluation/

Model evaluation and metrics:

- `metrics.py`: Custom evaluation metrics beyond standard sklearn
- `cross_validation.py`: Stratified cross-validation for imbalanced datasets
- `error_analysis.py`: In-depth error analysis and misclassification detection
- `benchmark.py`: Performance benchmarking across hardware configurations

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
