# Reuters RAG Classifier Scripts

This directory contains utility scripts for the Reuters RAG Classifier project. These scripts help with model training, hyperparameter tuning, demonstration, and API serving.

## Table of Contents
- [Scripts Overview](#scripts-overview)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Development Notes](#development-notes)

## Scripts Overview

### reuters_news_classifier_demo.py

A comprehensive demonstration script showcasing the capabilities of the Reuters news classifier.

**Purpose:** Demonstrate the full pipeline of news classification and semantic search capabilities.

**Usage:**
```bash
python scripts/reuters_news_classifier_demo.py --model bert_lr --input "path/to/news/article.txt"
```

**Features:**
- Support for all model types (Naïve Bayes, SVM, BERT, RAG)
- Batch processing of multiple articles
- Detailed performance metrics
- Visualization of results

### semantic_search_demo.py

A demonstration script for the semantic search capabilities using FAISS.

**Purpose:** Showcase the semantic search functionality across the Reuters corpus.

**Usage:**
```bash
python scripts/semantic_search_demo.py --query "your search query" --top_k 5
```

**Parameters:**
- `--query`: The search query text
- `--top_k`: Number of results to return (default: 5)
- `--model`: Embedding model to use (default: "all-MiniLM-L6-v2")

### news_trend_analyzer.py

A script for analyzing news trends and topic evolution over time.

**Purpose:** Generate insights about topic distribution and trends in the news corpus.

**Usage:**
```bash
python scripts/news_trend_analyzer.py --start_date 2023-01-01 --end_date 2023-12-31
```

**Features:**
- Topic distribution analysis
- Trend visualization
- Time-series analysis
- Export to various formats (CSV, JSON, HTML)

### tune_hyperparams.py

A script to perform hyperparameter optimization using Ray Tune.

**Purpose:** Find the best hyperparameters for the classification models.

**Usage:**
```bash
python scripts/tune_hyperparams.py --config path/to/config.yaml --algo [nb|lr] --num-samples 30
```

**Parameters:**
- `--config`: Path to YAML configuration file (required)
- `--algo`: Algorithm to tune (`nb` for Naïve Bayes, `lr` for Logistic Regression)
- `--num-samples`: Number of hyperparameter combinations to try (default: 30)
- `--output`: Path to save results (default: "output/hyperparams/")

**Tunes:**
- C parameter for Logistic Regression
- alpha parameter for Multinomial Naïve Bayes
- Learning rate for transformer models
- Batch size and other training parameters

## API Documentation

The `api` directory contains a FastAPI implementation for serving the trained models.

### Main Components
- `main.py`: The FastAPI application
- `model_loader.py`: Utility for loading trained models
- `routes/`: API endpoint definitions
- `schemas/`: Pydantic models for request/response validation

### Running the API Server

```bash
# Development mode
uvicorn scripts.api.main:app --reload

# Production mode
uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

1. **Classification**
   - POST `/api/v1/classify`
   - Input: Text document
   - Output: Predicted topic and confidence

2. **Semantic Search**
   - POST `/api/v1/search`
   - Input: Query text
   - Output: Similar documents

3. **Model Management**
   - GET `/api/v1/models`
   - POST `/api/v1/models/load`
   - DELETE `/api/v1/models/{model_id}`

## Usage Examples

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

## Development Notes

### Adding New Scripts
1. Create a new Python file in the `scripts` directory
2. Add proper documentation and type hints
3. Include command-line argument parsing using `argparse`
4. Add error handling and logging
5. Update this README with the new script's documentation

### Testing Scripts
```bash
# Run script tests
pytest tests/scripts/

# Check code style
flake8 scripts/
black scripts/
```

### Logging
All scripts use the standard Python logging module with the following configuration:
- Log level: INFO
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Output: Both console and file (`logs/scripts.log`) 