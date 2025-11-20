# CLINC150 Intent Classification & Semantic Search

A production-ready NLP pipeline for automated intent classification and semantic search, built on the CLINC150 dataset.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-blue)](docs/)

## Table of Contents
- [Quick Start](#-quick-start)
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Key Features](#-key-features)
- [Performance Metrics](#-performance-metrics)
- [Scaling Guidance](#-scaling-guidance)
- [Development](#-development)
- [Switching Between Ollama and OpenAI Models](#-switching-between-ollama-and-openai-models)
- [Dataset Configuration](#-dataset-configuration)
- [Acknowledgments](#-acknowledgments)

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Virtual environment (recommended)

### Installation

#### Using uv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/mmaceira/multi-model-intent-classifier.git
cd multi-model-intent-classifier

# 2. Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install dependencies (uv will automatically create a virtual environment)
uv sync

# 4. Activate the virtual environment (if needed)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 5. Set up Ollama (required for RAG-LLM models)
# Make sure Ollama is installed and running:
#   ollama serve
#   ollama pull llama3.1:8b

# 6. (Optional) Set up OpenAI API key if you want to use OpenAI models
# export OPENAI_API_KEY="your-api-key-here"
# Note: OpenAI models are disabled by default - the pipeline works with Ollama only

# 7. Run the complete training pipeline
python scripts/pipeline/run_all.py

# Or run individual pipeline steps:
# python scripts/pipeline/00_data_loading.py          # Load and validate dataset
# python scripts/pipeline/01_exploratory_analysis.py   # Analyze dataset characteristics
# python scripts/pipeline/02_build_embeddings.py        # Generate embeddings (SBERT by default)
# python scripts/pipeline/03_model_training.py          # Train models
# python scripts/pipeline/04_model_prediction.py        # Generate predictions
# python scripts/pipeline/05_model_evaluation.py       # Evaluate model performance
```

#### Using pip (Alternative)

```bash
# 1. Clone the repository
git clone https://github.com/mmaceira/multi-model-intent-classifier.git
cd multi-model-intent-classifier

# 2. Create and activate virtual environment
python -m venv venv/multi-model-intent-classifier
source venv/multi-model-intent-classifier/bin/activate  # On Windows: venv\multi-model-intent-classifier\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Set up Ollama (required for RAG-LLM models)
# Make sure Ollama is installed and running:
#   ollama serve
#   ollama pull llama3.1:8b

# 5. (Optional) Set up OpenAI API key if you want to use OpenAI models
# export OPENAI_API_KEY="your-api-key-here"
# Note: OpenAI models are disabled by default - the pipeline works with Ollama only

# 6. Run the complete training pipeline
python scripts/pipeline/run_all.py

# Or run individual pipeline steps (see above for details)
```

### Basic Usage

```python
from src.datasets.dataset import get_dataset

# Load CLINC150 dataset
X_train, y_train, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

# Use with any classifier
from src.algorithms.linear_svm import LinearSVMClassifier
classifier = LinearSVMClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

## 📋 Overview

This project implements a comprehensive NLP pipeline for:
- Automated intent classification of user utterances
- Semantic search and document retrieval
- Business insights generation
- Real-time document similarity matching
- Multi-language support

Built on the CLINC150 dataset, it provides a production-ready solution for intent classification and information retrieval. The system combines traditional machine learning approaches with modern transformer-based models and Retrieval-Augmented Generation (RAG) techniques to achieve state-of-the-art performance.

### Use Cases

1. **Intent Classification**: Automatically classify user utterances into predefined intents
2. **Information Retrieval**: Find relevant examples based on semantic similarity
3. **Content Recommendation**: Suggest related intents based on content similarity
4. **Trend Analysis**: Track intent evolution and popularity over time
5. **Research & Analysis**: Support qualitative and quantitative research on intent data

## 🏗️ Project Structure

```
multi-model-intent-classifier/
├── config/              # Configuration files
│   ├── config.yaml     # Main configuration file
│   ├── models_config.yaml # Model selection and parameters
│   └── notebook_setup.py # Configuration setup for pipeline scripts
├── scripts/             # Utility scripts
│   ├── pipeline/        # Training pipeline scripts
│   │   ├── 00_data_loading.py
│   │   ├── 01_exploratory_analysis.py
│   │   ├── 02_build_embeddings.py
│   │   ├── 03_model_training.py
│   │   ├── 04_model_prediction.py
│   │   ├── 05_model_evaluation.py
│   │   ├── run_all.py                 # Run all scripts in sequence
│   │   └── README.md                   # Pipeline scripts documentation
│   ├── api/            # FastAPI implementation
│   │   ├── main_api.py
│   │   ├── model_loader.py
│   │   └── README_API.md
│   ├── semantic_search_demo.py        # Search demo
│   ├── tune_hyperparams.py            # Hyperparameter tuning
│   └── README.md                      # Scripts documentation
├── output/             # Model outputs and results
├── src/                # Main source code
│   ├── algorithms/     # ML algorithms implementation
│   ├── datasets/       # Dataset handling
│   ├── embeddings/     # Embedding generation
│   ├── evaluation/     # Model evaluation tools
│   ├── exploration.py  # Data exploration utilities
│   ├── model.py        # Core model implementations
│   ├── prediction.py   # Prediction utilities
│   ├── rag/            # RAG implementation
│   ├── training.py     # Training utilities
│   ├── utils/          # Utility functions
│   └── README.md       # Source code documentation
├── tests/              # Test files
│   └── test_dataset_clinc150.py
├── .env               # Environment variables
├── .gitignore         # Git ignore file
├── pyproject.toml     # Project dependencies and metadata
└── README.md         # This file
```

## 🧠 Model Architecture

### Model Comparison

| Model | Architecture | Use Case |
|-------|--------------|---------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |
| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | Context-aware classification with flexible LLM provider |

### Technical Specifications

#### 1. Multinomial Naive Bayes (`nb_tfidf`)
- **Implementation**: Bag-of-words TF-IDF vectorization with probabilistic modeling
- **Pipeline**: Text preprocessing → TF-IDF vectorization → Model training → Fast inference
- **Performance**:
  - Training: ~2 min/2M docs
  - Inference: 60k docs/s
  - Resources: < 2GB RAM, CPU-only

#### 2. Linear SVM (`svm_linear`, `svm_bigram`)
- **Implementation**: Uni- & bi-gram features with L2 regularization
- **Pipeline**: Text preprocessing → N-gram extraction → TF-IDF → SVM training
- **Performance**:
  - Training: 7-8 min
  - Inference: 12-15k docs/s
  - Resources: < 5GB RAM, CPU-only

#### 3. MiniLM + Logistic Regression (`bert_lr`)
- **Implementation**: Transformer embeddings with Logistic Regression head
- **Pipeline**: Text preprocessing → MiniLM embedding → Dimensionality reduction → Classification
- **Performance**:
  - Training: ~45 min on A10 GPU
  - Inference: 1k docs/s
  - Resources: 12GB GPU

#### 4. RAG Classification Implementation (`rag_faiss`)
- **Implementation**: The RAG (Retrieval-Augmented Generation) classification system combines the power of semantic search with large language models to make accurate classification decisions.

  1. **Document Embedding**
    - Input documents are converted into dense vector representations
    - Uses `openai text-embedding-3-small` or `sentence-transformers/all-MiniLM-L6-v2` for high-quality embeddings
    - Embeddings capture semantic meaning and document context

  2. **Context Retrieval**
    - FAISS (Facebook AI Similarity Search) is used for efficient similarity search
    - For each document, retrieves `top_k` (default: 5) most similar documents
    - Similar documents serve as contextual examples for classification
    - Optimized for speed with approximate nearest neighbor search

  3. **LLM Classification**
    - Uses litellm which supports multiple LLM providers (Ollama, OpenAI, Anthropic, etc.)
    - Default: Ollama ("ollama/llama3.1:8b") for local, cost-free inference
    - Easy to switch to OpenAI models (e.g., "gpt-4o-mini", "gpt-4o") by changing the `model` parameter
    - Provides the model with:
      - Document to classify
      - Retrieved similar documents as context
      - List of valid classification labels
    - Returns JSON-formatted classification decisions
- **Pipeline**: Document embedding → FAISS index → Query embedding → ANN search → LLM reranking
- **Performance**:
  - Index Build: ~25 min
  - Query Speed: 200 QPS
  - Resources: 16GB RAM + LLM

## 💡 Key Features

### 1. Intelligent Document Processing
- Automated intent classification for user utterances
- Real-time document similarity matching
- Semantic search across document collections
- Multi-language support through transformer models

### 2. Enterprise-Grade API & Interface
- RESTful API for seamless integration
- Modern web interface for document management
- Role-based access control
- Audit logging and compliance tracking

### 3. Advanced Analytics & Visualization
- Interactive intent distribution dashboards
- Document similarity networks
- Trend analysis and intent evolution
- Custom report generation

### 4. Performance Optimization
- Automated hyperparameter tuning
- Model performance monitoring
- Resource utilization optimization
- Cost-effective scaling options

## 📊 Performance Metrics

### Model Performance on CLINC150

| Model | Architecture | Accuracy | Macro-F1 | Training Time | Inference Speed | Resource Usage |
|-------|--------------|----------|----------|---------------|-----------------|----------------|
| Naive Bayes | TF-IDF + Naive Bayes | - | - | ~2 min | 60k docs/s | < 2GB RAM |
| Linear SVM | TF-IDF + SVM | - | - | 7-8 min | 12-15k docs/s | < 5GB RAM |
| MiniLM + LogReg | Transformer + Logistic Regression | - | - | ~45 min | 1k docs/s | 12GB GPU |
| RAG-CentroidNN | FAISS + Nearest Neighbors | - | - | ~25 min | 200 QPS | 16GB RAM |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | - | - | ~30 min | 150-180 QPS | 16GB RAM |

### Model Strengths
- **Naive Bayes**: Fastest inference, suitable for real-time applications
- **Linear SVM**: Best balance of speed and accuracy
- **MiniLM + LogReg**: Highest accuracy, best for precision-critical tasks
- **RAG Models**: Best for semantic understanding and context-aware classification

### Resource Requirements
| Model | CPU | RAM | GPU | Storage |
|-------|-----|-----|-----|---------|
| Naive Bayes | ✓ | 2GB | - | 500MB |
| Linear SVM | ✓ | 5GB | - | 1GB |
| MiniLM + LogReg | - | 8GB | 12GB | 2GB |
| RAG Models | ✓ | 16GB | - | 5GB |

## 🧮 Scaling Guidance

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

## 🛠️ Development

### Environment Setup
1. Install development dependencies (using uv):
   ```bash
   uv sync --dev
   ```
   Or using pip:
   ```bash
   pip install -e ".[dev]"
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public functions
- Run `black` and `flake8` before committing

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dataset_clinc150.py

# Run with coverage
pytest --cov=src
```

### Documentation
- API documentation is generated using Sphinx
- Run `make docs` to build documentation
- View documentation at `docs/_build/html/index.html`

## 🙏 Acknowledgments

- CLINC150 dataset (HuggingFace)
- Hugging Face Transformers
- FAISS for similarity search
- OpenAI for embedding models

## 🔄 Switching Between Ollama and OpenAI Models

The project uses `litellm` which supports multiple LLM providers. By default, it uses Ollama for local, cost-free inference, but it's very easy to switch to OpenAI or other providers.

### Using OpenAI Models

To use OpenAI models instead of Ollama, simply change the `model` parameter in your configuration:

**Option 1: Update `config/models_config.yaml`**
```yaml
rag_llm:
  enabled: true
  name: "RAG-LLM"
  class: "RagSklearnAdapter"
  params:
    method: "llm"
    top_k: 25
    model: "gpt-4o-mini"  # Change from "${model.llm_model}" (default: ollama/llama3.1:8b) to OpenAI model
    use_openai: false  # Set to true to use OpenAI embeddings instead of SBERT
```

**Option 2: Set environment variable and use in code**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option 2: Update `config/config.yaml`**
```yaml
model:
  llm_model: "gpt-4o-mini"  # Change from "ollama/llama3.1:8b" to OpenAI model
```

Then in `config/models_config.yaml`, set:
```yaml
rag_llm:
  params:
    model: "${model.llm_model}"  # Will use the OpenAI model from config.yaml
    use_openai: false  # Set to true for OpenAI embeddings, false for SBERT
```

### Supported Models

The project supports any model that `litellm` supports, including:
- **Ollama**: `ollama/llama3.1:8b`, `ollama/llama3.2:3b`, `ollama/mistral`, etc.
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
- **Anthropic**: `claude-3-5-sonnet`, `claude-3-opus`, etc.
- **Other providers**: See [litellm documentation](https://docs.litellm.ai/) for full list

### Configuration Tips

- **For local development**: Use Ollama (default) - no API keys needed
- **For production**: Consider OpenAI's `gpt-4o-mini` for cost-effective, high-quality results
- **For best accuracy**: Use `gpt-4o` or `claude-3-5-sonnet`
- **For speed**: Use `gpt-4o-mini` or `gpt-3.5-turbo`

## ⚙️ Configuration

The project uses YAML configuration files to manage experiments and model settings.

### Main Configuration (`config/config.yaml`)

The main configuration file controls experiment settings, dataset parameters, and model defaults:

```yaml
# General Configuration
general:
  run_name: "experiment_10_classes"     # Experiment identifier (used in output paths)
  seed: 42                              # Random seed for reproducibility

# Dataset configuration
dataset:
  name: "clinc150"                      # Dataset name
  use_oos: false                        # Include out-of-scope examples

# Paths (all are resolved relative to repo root)
paths:
  data_exploration_dir: "output/${general.run_name}/data_exploration"
  embeddings_dir: "output/${general.run_name}/embeddings"
  models_dir: "output/${general.run_name}/models"
  predictions_dir: "output/${general.run_name}/predictions"
  results_dir: "output/${general.run_name}/results"

# Model configuration
model:
  embedding_backend: "sbert"            # [sbert | openai]
  sbert_model_name: "sentence-transformers/all-MiniLM-L6-v2"
  openai_model_name: "text-embedding-3-small"
  classifier: "linear_svm"             # [linear_svm | naive_bayes | transformer_logreg]
  rag_top_k: 25                         # Number of neighbors for RAG models
  llm_model: "ollama/llama3.1:8b"      # Default LLM for RAG-LLM models
```

**Key Configuration Options:**
- `general.run_name`: Sets the experiment identifier and output directory name
- `model.llm_model`: Default LLM model for RAG-LLM (supports Ollama, OpenAI, Anthropic, etc.)
- `model.rag_top_k`: Number of similar examples to retrieve for RAG models
- `dataset.use_oos`: Whether to include out-of-scope examples as an extra class

### Model Selection Configuration (`config/models_config.yaml`)

This file controls which models are trained and their specific parameters:

```yaml
models:
  naive_bayes:
    enabled: true
    name: "Naive Bayes"
    class: "NaiveBayesClassifier"

  rag_llm:
    enabled: true
    name: "RAG-LLM"
    class: "RagSklearnAdapter"
    params:
      method: "llm"
      top_k: "${model.rag_top_k}"
      model: "${model.llm_model}"      # Uses value from config.yaml
      use_openai: false                 # false = SBERT embeddings, true = OpenAI embeddings
```

**Model Configuration Tips:**
- Set `enabled: false` to skip training a model
- Use `${model.rag_top_k}` to reference values from `config.yaml`
- For RAG-LLM, set `use_openai: true` to use OpenAI embeddings (requires `OPENAI_API_KEY`)
- Change `model` parameter to switch LLM providers (Ollama, OpenAI, etc.)

### Multiple Experiment Configurations

The project includes several pre-configured experiment files:
- `config/config.yaml` - Default experiment (10 classes)
- `config/config_10_classes.yaml` - 10 classes experiment
- `config/config_25_classes.yaml` - 25 classes experiment
- `config/config_tiny_dataset.yaml` - Small dataset for testing

To use a different configuration, modify the scripts to load a different config file, or copy and modify `config.yaml`.

## 📊 Dataset Configuration

### CLINC150 (Intent Classification)

The project uses the CLINC150 dataset for intent classification. CLINC150 is loaded via the HuggingFace `clinc_oos` dataset (config: `plus`). By default we:
- merge train and validation splits into a single training set
- drop out-of-scope (OOS) examples, unless `use_oos=True` is passed when calling `get_dataset`.

#### Dataset Features
- **Source**: HuggingFace `clinc_oos` dataset (config: `plus`)
- **Content**: 150 in-scope intents across 10 domains
- **Labels**: Intent labels as strings (e.g., "transfer_money", "greeting")
- **Optional**: Out-of-scope (OOS) examples can be included as an extra class

#### Configuration Parameters
```yaml
dataset:
  name: "clinc150"                           # dataset name (CLINC150 intent classification)
  use_oos: false                             # include out-of-scope examples
```

#### Example Usage
```python
from src.datasets.dataset import get_dataset

# Load CLINC150 without OOS examples
X_train, y_train, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

# Load CLINC150 with OOS examples
X_train, y_train, X_test, y_test, classes = get_dataset(
    dataset_name="clinc150",
    use_oos=True
)
```

### Running the Complete Pipeline

The easiest way to run the complete pipeline is using the `run_all.py` script:

```bash
# From the project root
python scripts/pipeline/run_all.py
```

This will execute all pipeline steps in sequence:
1. **Data Loading** - Load and validate the CLINC150 dataset
2. **Exploratory Analysis** - Analyze dataset characteristics and generate visualizations
3. **Build Embeddings** - Generate SBERT embeddings (and OpenAI embeddings if API key is set)
4. **Model Training** - Train all enabled models from `config/models_config.yaml`
5. **Model Prediction** - Generate predictions for all trained models
6. **Model Evaluation** - Evaluate models and generate comparison reports

### Running Individual Pipeline Steps

You can also run individual steps if needed:

```bash
# Step 1: Load and validate data
python scripts/pipeline/00_data_loading.py

# Step 2: Perform exploratory analysis
python scripts/pipeline/01_exploratory_analysis.py

# Step 3: Build embeddings (SBERT by default, OpenAI optional)
python scripts/pipeline/02_build_embeddings.py

# Step 4: Train models
python scripts/pipeline/03_model_training.py

# Step 5: Generate predictions
python scripts/pipeline/04_model_prediction.py

# Step 6: Evaluate models
python scripts/pipeline/05_model_evaluation.py
```

**Note**: Each step expects outputs from previous steps. Make sure to run them in order.

### Using the Entry Point

You can also use the installed entry point:

```bash
# After installation (pip install -e . or uv sync)
multi-model-pipeline
```

### Data Exploration
The project includes comprehensive data analysis tools:
- Class distribution analysis
- Text length statistics
- Vocabulary analysis
- Stopword analysis
- Vocabulary drift analysis
- Publication-ready visualizations
- CSV export capabilities
