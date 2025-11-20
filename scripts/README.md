# CLINC150 Intent Classification - Utility Scripts

This directory contains production-ready utility scripts for the CLINC150 Intent Classification project. These scripts provide command-line interfaces for model training, demonstration, evaluation, and API serving.

## Table of Contents
- [Scripts Overview](#scripts-overview)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Performance Considerations](#performance-considerations)
- [Deployment Guide](#deployment-guide)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

## Scripts Overview

### reuters_news_classifier_demo.py

A comprehensive demonstration script for intent classification (legacy name - works with CLINC150 dataset).

**Purpose:** Showcase the full pipeline of intent classification and semantic search capabilities with a user-friendly interface.

**Features:**
- Supports all model types: Naive Bayes, SVM, BERT, and RAG variants
- Interactive mode with real-time classification
- Batch processing of multiple articles
- Detailed performance metrics and visualizations
- Export results to various formats (JSON, CSV, HTML)

**Usage Example:**
```bash
# Classify a single document
python scripts/reuters_news_classifier_demo.py --model bert_lr --input "path/to/news/article.txt"

# Batch processing mode
python scripts/reuters_news_classifier_demo.py --model svm_linear --batch "path/to/articles/*.txt" --output results.csv

# Interactive demo mode
python scripts/reuters_news_classifier_demo.py --interactive
```

### semantic_search_demo.py

A demonstration script for semantic search capabilities using vector embeddings.

**Purpose:** Provide an interface for semantic search across the CLINC150 dataset using different embedding models.

**Features:**
- Configurable embedding models
- Adjustable similarity thresholds
- Result ranking and filtering
- Interactive mode for exploration
- Export search results with metadata

**Usage Example:**
```bash
# Basic search
python scripts/semantic_search_demo.py --query "interest rates federal reserve" --top_k 5

# With specific model and threshold
python scripts/semantic_search_demo.py --query "crude oil production" --model openai --top_k 10 --threshold 0.75

# Interactive mode
python scripts/semantic_search_demo.py --interactive
```

**Parameters:**
- `--query`: The search query text
- `--top_k`: Number of results to return (default: 5)
- `--model`: Embedding model to use (default: "all-MiniLM-L6-v2", options: "openai", "all-MiniLM-L6-v2", "paraphrase-mpnet-base-v2")
- `--threshold`: Minimum similarity threshold (default: 0.6)
- `--batch`: Path to file with multiple queries
- `--output`: Path to save results (default: None, prints to console)
- `--interactive`: Start in interactive mode

### news_trend_analyzer.py

A script for analyzing intent trends and evolution over time (legacy script - may need adaptation for CLINC150).

**Purpose:** Generate insights about intent distribution and trends in the dataset with time-based analysis.

**Features:**
- Time-series analysis of topic frequency
- Trend detection and visualization
- Topic correlation over time
- Seasonal pattern identification
- Export to multiple formats

**Usage Example:**
```bash
# Basic trend analysis
python scripts/news_trend_analyzer.py --start_date 2023-01-01 --end_date 2023-12-31

# Focus on specific topics
python scripts/news_trend_analyzer.py --topics "crude,oil,energy" --resolution weekly

# Generate visualization
python scripts/news_trend_analyzer.py --visualize --output trends.html
```

**Parameters:**
- `--start_date`: Start date for analysis (format: YYYY-MM-DD)
- `--end_date`: End date for analysis (format: YYYY-MM-DD)
- `--topics`: Comma-separated list of topics to analyze (default: all)
- `--resolution`: Time resolution (daily, weekly, monthly, quarterly)
- `--visualize`: Generate visualizations
- `--output`: Output path for results and visualizations

### tune_hyperparams.py

A script for hyperparameter optimization using Ray Tune.

**Purpose:** Find optimal hyperparameters for different classification models to maximize performance metrics.

**Features:**
- Distributed hyperparameter search
- Multiple search algorithms (Random, Bayesian, BOHB)
- Cross-validation integration
- Early stopping for efficiency
- Comprehensive logging and reporting

**Usage Example:**
```bash
# Tune Naive Bayes model
python scripts/tune_hyperparams.py --config config/config.yaml --algo nb --num-samples 30

# Tune SVM with specific search space
python scripts/tune_hyperparams.py --config config/config.yaml --algo svm --search-space "config/svm_params.json"

# Tune BERT model with GPU
python scripts/tune_hyperparams.py --config config/config.yaml --algo bert_lr --gpu --num-samples 10
```

**Parameters:**
- `--config`: Path to YAML configuration file (required)
- `--algo`: Algorithm to tune (`nb` for Naïve Bayes, `svm` for SVM, `bert_lr` for BERT+LR)
- `--num-samples`: Number of hyperparameter combinations to try (default: 30)
- `--search-space`: Custom search space definition file
- `--output`: Path to save results (default: "output/hyperparams/")
- `--gpu`: Use GPU for training if available
- `--cpus`: Number of CPU cores to use (default: auto-detect)
- `--metric`: Metric to optimize (default: "f1_macro")

**Tunes:**
- C parameter for Logistic Regression
- alpha parameter for Multinomial Naïve Bayes
- Learning rate for transformer models
- Batch size and other training parameters

## API Documentation

The `api` directory contains a FastAPI implementation for serving the trained models.

### API Architecture
- `main.py`: The FastAPI application entry point
- `model_loader.py`: Model loading and management
- `routes/`: API endpoint definitions
- `schemas/`: Pydantic models for request/response validation
- `middleware/`: Request processing middleware
- `services/`: Business logic implementation

### API Endpoints

#### 1. Classification
- **Endpoint**: POST `/api/v1/classify`
- **Description**: Classify a document into one of the predefined topics
- **Request Body**: JSON with `text` field containing document content
- **Response**: JSON with predicted topic, confidence, and metadata
- **Optional Parameters**:
  - `model`: Model to use (default: configured in settings)
  - `return_confidence`: Whether to return confidence scores (default: true)
  - `return_alternatives`: Whether to return alternative classifications (default: false)

#### 2. Semantic Search
- **Endpoint**: POST `/api/v1/search`
- **Description**: Find semantically similar documents
- **Request Body**: JSON with `query` text and optional parameters
- **Response**: Array of similar documents with similarity scores
- **Optional Parameters**:
  - `top_k`: Number of results to return (default: 5)
  - `threshold`: Minimum similarity threshold (default: 0.6)
  - `embedding_model`: Model to use for embeddings (default: configured in settings)

#### 3. Model Management
- **Endpoint**: GET `/api/v1/models`
- **Description**: List available models
- **Response**: Array of model information (name, type, metrics)

- **Endpoint**: POST `/api/v1/models/load`
- **Description**: Load a model into memory
- **Request Body**: JSON with `model_id` and optional parameters
- **Response**: Status of the load operation

- **Endpoint**: DELETE `/api/v1/models/{model_id}`
- **Description**: Unload a model from memory
- **Response**: Status of the unload operation

### Running the API Server

```bash
# Development mode
uvicorn scripts.api.main:app --reload --host 127.0.0.1 --port 8000

# Production mode
uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Authentication
The API supports multiple authentication methods:
- API key authentication (via `X-API-Key` header)
- JWT token authentication
- OAuth2 (configurable via settings)

1. **Classification**
   - POST `/api/v1/classify`
   - Input: Text document
   - Output: Predicted topic and confidence
```

### Batch Processing
```python
from scripts.news_trend_analyzer import NewsTrendAnalyzer

analyzer = NewsTrendAnalyzer()
results = analyzer.analyze_batch(
    articles=["article1.txt", "article2.txt"],
    output_format="json"
)
```

### Custom Model Training
```python
from scripts.tune_hyperparams import HyperparameterTuner

tuner = HyperparameterTuner(
    config_path="config/hyperparams.yaml",
    algorithm="lr"
)
best_params = tuner.optimize()
```
