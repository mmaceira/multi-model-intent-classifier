# Reuters RAG Classifier Scripts

This directory contains utility scripts for the Reuters RAG Classifier project. These scripts help with model training, hyperparameter tuning, demonstration, and API serving.

## Scripts Overview

### gradio_demo.py

A demo application built with Gradio that allows you to interactively test the trained models.

**Purpose:** Demonstrate model capabilities through a web interface where you can select a model, input text, and get classification or search results.

**Usage:**
```bash
python scripts/gradio_demo.py
```

**Features:**
- Select from different model types (Naïve Bayes, SVM, BERT, RAG)
- Input your own text for classification
- Adjust top-k results for RAG search

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

**Tunes:**
- C parameter for Logistic Regression
- alpha parameter for Multinomial Naïve Bayes

## API Subdirectory

The `api` directory contains a FastAPI implementation for serving the trained models.

**Main Files:**
- `main.py`: The FastAPI application
- `model_loader.py`: Utility for loading trained models

**To run the API server:**
```bash
uvicorn scripts.api.main:app --reload
```

For more detailed information about the API, please refer to the [API README](api/README_API.md). 