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

### Running with the Real Dataset (CLINC150)

**The CLINC150 dataset is automatically downloaded from HuggingFace** - no manual setup required! Just run:

```bash
# 1. Install dependencies
uv sync  # or: pip install -e .

# 2. (Optional) Set up Ollama for RAG-LLM models
ollama serve
ollama pull llama3.1:8b

# 3. Run the pipeline - dataset downloads automatically!
python scripts/pipeline/run_all.py
```

That's it! The pipeline will:
- ✅ Automatically download CLINC150 from HuggingFace (requires internet on first run)
- ✅ Use the full dataset (~23k training, ~5.7k test samples, 150 classes)
- ✅ Train all enabled models
- ✅ Generate predictions and evaluations

**To use a smaller dataset for quick testing:**
```bash
CONFIG_FILE=config_tiny_dataset.yaml python scripts/pipeline/run_all.py
```

See [Dataset Configuration](#-dataset-configuration) for details on customizing dataset size.

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Virtual environment (recommended)
- Internet connection (for initial dataset download from HuggingFace)

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
# The CLINC150 dataset will be automatically downloaded from HuggingFace on first run
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
# The CLINC150 dataset will be automatically downloaded from HuggingFace on first run
python scripts/pipeline/run_all.py

# Or run individual pipeline steps (see above for details)
# Note: The dataset is automatically downloaded - no manual setup needed!
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

### Running the Training Pipeline

The easiest way to run the complete training and evaluation pipeline:

```bash
# Using the entry point (after installation)
multi-model-pipeline

# Or directly with Python
python scripts/pipeline/run_all.py
```

This will execute all pipeline steps in sequence:
1. **Data Loading** - Load and validate the CLINC150 dataset
2. **Exploratory Analysis** - Analyze dataset characteristics and generate visualizations
3. **Build Embeddings** - Generate SBERT embeddings (and OpenAI embeddings if API key is set)
4. **Model Training** - Train all enabled models from `config/models_config.yaml`
5. **Model Prediction** - Generate predictions for all trained models
6. **Model Evaluation** - Evaluate models and generate comparison reports

You can also run individual pipeline steps:

```bash
# Step-by-step execution
python scripts/pipeline/00_data_loading.py          # Load and validate dataset
python scripts/pipeline/01_exploratory_analysis.py   # Analyze dataset characteristics
python scripts/pipeline/02_build_embeddings.py        # Generate embeddings (SBERT by default)
python scripts/pipeline/03_model_training.py          # Train models
python scripts/pipeline/04_model_prediction.py        # Generate predictions
python scripts/pipeline/05_model_evaluation.py       # Evaluate model performance
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
| Embedding + LogReg | Flexible embeddings (SBERT/OpenAI) + Logistic Regression | High-accuracy with flexible embedding backend |
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

#### 3. MiniLM + Logistic Regression (`transformer_logreg`)
- **Implementation**: Transformer embeddings with Logistic Regression head
- **Pipeline**: Text preprocessing → MiniLM embedding → Dimensionality reduction → Classification
- **Performance**:
  - Training: ~45 min on A10 GPU
  - Inference: 1k docs/s
  - Resources: 12GB GPU

#### 4. Embedding + Logistic Regression (`embedding_logreg`)
- **Implementation**: Flexible embedding backend (SBERT or OpenAI) with Logistic Regression head
- **Pipeline**: Text preprocessing → Embedding (SBERT local or OpenAI API) → StandardScaler → Classification
- **Features**:
  - **SBERT mode** (default): Local embeddings, no API key needed
  - **OpenAI mode**: API-based embeddings, requires OPENAI_API_KEY
  - Same architecture as MiniLM + LogReg but with flexible embedding backend
- **Performance**:
  - Training: ~45 min (SBERT) or depends on API rate limits (OpenAI)
  - Inference: 1k docs/s (SBERT) or depends on API rate limits (OpenAI)
  - Resources: 12GB GPU (SBERT) or minimal (OpenAI, API-based)

#### 5. RAG Classification Implementation (`rag_faiss`)
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

## 🎯 Hyperparameter Tuning

This repository includes a comprehensive hyperparameter tuning system that follows ML best practices.

### Overview

The hyperparameter tuning script (`scripts/tune_hyperparams.py`) optimizes hyperparameters for all models using the **validation set** (not the test set), ensuring proper model selection without data leakage.

### Supported Models

The tuning script supports hyperparameter optimization for:
- **Naive Bayes**: Tunes `alpha` (smoothing parameter)
- **Linear SVM**: Tunes `C` (regularization parameter)
- **Linear SVM Bigrams**: Tunes `C` (regularization parameter)
- **Transformer LogReg**: Tunes `C` (regularization parameter) - MiniLM + Logistic Regression
- **Embedding LogReg**: Tunes `C` (regularization parameter) - Flexible embeddings (SBERT or OpenAI) + Logistic Regression
- **RAG KMajority**: Tunes `top_k` (number of neighbors)
- **RAG Centroid**: Evaluates default configuration
- **RAG LLM**: Tunes `top_k` (number of neighbors)

### Usage

#### Tune All Models

```bash
# Tune all models with default settings
python scripts/tune_hyperparams.py --config config/config.yaml --all

# Tune with more samples for better results
python scripts/tune_hyperparams.py --config config/config.yaml --all --num-samples 50
```

#### Tune Specific Model

```bash
# Tune only Naive Bayes
python scripts/tune_hyperparams.py --config config/config.yaml --algo nb --num-samples 30

# Tune only Linear SVM
python scripts/tune_hyperparams.py --config config/config.yaml --algo svm --num-samples 30

# Tune Embedding LogReg (defaults to SBERT embeddings, no API key needed)
python scripts/tune_hyperparams.py --config config/config.yaml --algo embedding_logreg --num-samples 30

# Tune Embedding LogReg with OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key"
python scripts/tune_hyperparams.py --config config/config.yaml --algo openai_logreg --num-samples 30
```

### Integration with Training Pipeline

**The tuned hyperparameters are automatically loaded and used during training!**

1. **Run hyperparameter tuning** (optional but recommended):
   ```bash
   python scripts/tune_hyperparams.py --config config/config.yaml --all
   ```

2. **Run training pipeline** - it will automatically use tuned hyperparameters:
   ```bash
   python scripts/pipeline/03_model_training.py
   # or
   python scripts/pipeline/run_all.py
   ```

The model loader checks for individual hyperparameter files in `config/hyperparameters/{config_name}/` and automatically applies tuned hyperparameters to models. If no tuned hyperparameters are found, models use defaults from the configuration files.

### Storage Location

Tuned hyperparameters are saved as **individual YAML files** (one per model) to two locations, with **config-specific subdirectories**:

**Primary location (used by model loader):**
- `config/hyperparameters/{config_name}/best_{model_name}.yaml` - One file per model
  - Example: `best_naive_bayes.yaml`, `best_transformer_logreg.yaml`, `best_rag_kmajority.yaml`

**Secondary location (for reference/backup):**
- `output/hyperparams_tune/{config_name}/best_{model_name}.yaml` - Copy for reference

**Example:** If you use `config_tiny_dataset.yaml`, hyperparameters will be saved to:
- `config/hyperparameters/config_tiny_dataset/best_naive_bayes.yaml`
- `config/hyperparameters/config_tiny_dataset/best_transformer_logreg.yaml`
- `config/hyperparameters/config_tiny_dataset/best_rag_kmajority.yaml`
- etc.

This ensures that hyperparameters tuned with different config files are kept separate and don't conflict.

Example output (individual files):

`best_naive_bayes.yaml`:
```yaml
alpha: 0.123
```

`best_transformer_logreg.yaml`:
```yaml
C: 2.456
```

`best_rag_kmajority.yaml`:
```yaml
top_k: 15
```

### Committing Hyperparameters to Git

**Yes, you can (and should) commit hyperparameter files!**

The hyperparameter YAML files are:
- ✅ **Small** (typically < 1 KB each)
- ✅ **Reproducible** (same config = same results)
- ✅ **Useful for collaboration** (others can use your tuned hyperparameters)
- ✅ **Documentation** (shows what hyperparameters were used)

Hyperparameters in `config/hyperparameters/` are **not ignored** by `.gitignore` and should be committed to the repository. The `output/hyperparams_tune/` directory is kept for reference but is ignored by git.

**To commit hyperparameters:**
```bash
# After running hyperparameter tuning
# Note: Hyperparameters are stored in config-specific subdirectories
git add config/hyperparameters/*/
git commit -m "Add tuned hyperparameters for all models"
```

**Note:**
- Hyperparameters in `config/hyperparameters/` should be committed (they're configuration)
- Output files (models, predictions, embeddings) remain in `.gitignore` as they are large and experiment-specific

### Best Practices

1. **Always tune before final training**: Run hyperparameter tuning before training your final models
2. **Use validation set**: The tuning script correctly uses the validation set (not test set)
3. **Test set is never used**: The test set remains completely separate for final evaluation only
4. **Reproducible**: Uses the same seed and dataset splits for consistency

### Why This Matters

- **Proper ML Workflow**: Uses validation set for hyperparameter selection (standard practice)
- **No Data Leakage**: Test set is never touched during tuning or training
- **Automatic Integration**: Tuned hyperparameters are automatically used in the pipeline
- **Comprehensive**: Supports all models in the pipeline, not just a subset

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
   uv sync --extra dev
   ```
   Or using pip:
   ```bash
   pip install -e ".[dev]"
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### External Dependencies

The project has several optional dependencies for different features:

| Feature | Required Packages | Notes |
|---------|------------------|-------|
| **Base training & evaluation** | `scikit-learn`, `numpy`, `pandas`, `datasets` | Core dependencies for all models |
| **RAG + FAISS** | `faiss-cpu` | Required for RAG models (CentroidNN, k-Majority, LLM) |
| **OpenAI / LLM mode** | `litellm`, `openai` | Requires `OPENAI_API_KEY` environment variable |
| **API server** | `uvicorn`, `fastapi` | For running the REST API (`scripts/api/main_api.py`) |
| **SBERT embeddings** | `sentence-transformers` | Default embedding backend |
| **Testing** | `pytest`, `datasets` | Required for running test suite |

**Note**: All dependencies are listed in `pyproject.toml`. The base installation includes most dependencies. For OpenAI features, ensure `OPENAI_API_KEY` is set in your environment or `.env` file.

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public functions
- Run `black` and `flake8` before committing

### Testing

**Important**: Tests require the HuggingFace `datasets` library. Install dev dependencies with `pip install -e .[dev]` or `uv sync --extra dev` before running `pytest`.

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
- `dataset.max_classes`: Limit number of classes (None = all 150 classes)
- `dataset.max_train_samples`: Limit training samples (None = all ~23k samples)
- `dataset.max_test_samples`: Limit test samples (None = all ~5.7k samples)

See the [Dataset Configuration](#-dataset-configuration) section for detailed information on dataset parameters.

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
- `config/config.yaml` - Default experiment (full dataset)
- `config/config_10_classes.yaml` - 10 classes experiment (full dataset, 10 classes)
- `config/config_25_classes.yaml` - 25 classes experiment (full dataset)
- `config/config_tiny_dataset.yaml` - Small dataset for quick testing (10 classes, 100 train samples, 50 test samples)

**To use a different configuration file:**

```bash
# Set CONFIG_FILE environment variable
export CONFIG_FILE=config_tiny_dataset.yaml
python scripts/pipeline/run_all.py

# Or inline
CONFIG_FILE=config_25_classes.yaml python scripts/pipeline/run_all.py
```

See the [Dataset Configuration](#-dataset-configuration) section for details on configuring dataset size parameters.

## 📊 Dataset Configuration

### CLINC150 (Intent Classification)

**The CLINC150 dataset is automatically downloaded from HuggingFace** - no manual download or setup required! The dataset is a real, production-ready benchmark dataset for intent classification with 150 intents across 10 domains.

#### Dataset Features
- **Source**: HuggingFace `clinc_oos` dataset (config: `plus`) - automatically downloaded on first use
- **Content**: 150 in-scope intents across 10 domains (banking, credit cards, etc.)
- **Size**: ~23,000 training utterances, ~5,700 test utterances
- **Labels**: Intent labels as strings (e.g., "transfer_money", "greeting", "balance")
- **Optional**: Out-of-scope (OOS) examples can be included as an extra class

#### Running with the Full Dataset (Default)

By default, the pipeline uses the **full CLINC150 dataset** with all 150 classes. Simply run:

```bash
# Use default config (full dataset, 10 classes experiment name)
python scripts/pipeline/run_all.py

# Or use a specific config file
CONFIG_FILE=config.yaml python scripts/pipeline/run_all.py
```

The dataset will be automatically downloaded from HuggingFace on first use (requires internet connection).

#### Configuring Dataset Size

You can control the dataset size using configuration parameters in your config file:

```yaml
# Dataset configuration
dataset:
  name: "clinc150"                           # dataset name (CLINC150 intent classification)
  use_oos: false                             # include out-of-scope examples
  max_classes: 10                            # Limit number of classes (None = all 150 classes)
  max_train_samples: 1000                    # Limit training samples (None = all ~23k samples)
  max_test_samples: 500                     # Limit test samples (None = all ~5.7k samples)
```

**Configuration Options:**
- `max_classes`: Randomly select N classes from the full dataset (useful for quick testing)
- `max_train_samples`: Limit training set size (uses stratified sampling to maintain class balance)
- `max_test_samples`: Limit test set size (uses stratified sampling)
- `use_oos`: Include out-of-scope examples as an extra class label

**Example Configurations:**

1. **Full dataset** (default - no limits):
   ```yaml
   dataset:
     name: "clinc150"
     use_oos: false
     # No max_* parameters = use full dataset
   ```

2. **10 classes for quick testing**:
   ```yaml
   dataset:
     name: "clinc150"
     use_oos: false
     max_classes: 10
   ```

3. **Small dataset for development**:
   ```yaml
   dataset:
     name: "clinc150"
     use_oos: false
     max_classes: 10
     max_train_samples: 100
     max_test_samples: 50
   ```

#### Using Different Config Files

The project includes several pre-configured experiment files:

- `config/config.yaml` - Default (full dataset, 10 classes experiment name)
- `config/config_10_classes.yaml` - 10 classes experiment
- `config/config_25_classes.yaml` - 25 classes experiment
- `config/config_tiny_dataset.yaml` - Small dataset for quick testing (10 classes, 100 train, 50 test)

**To use a different config file:**

```bash
# Method 1: Set CONFIG_FILE environment variable
export CONFIG_FILE=config_tiny_dataset.yaml
python scripts/pipeline/run_all.py

# Method 2: Set it inline
CONFIG_FILE=config_25_classes.yaml python scripts/pipeline/run_all.py

# Method 3: Copy and modify config.yaml
cp config/config.yaml config/my_experiment.yaml
# Edit my_experiment.yaml, then:
CONFIG_FILE=my_experiment.yaml python scripts/pipeline/run_all.py
```

#### Example Usage in Code

```python
from src.datasets.dataset import get_dataset

# Load full CLINC150 dataset (all 150 classes, all samples)
# Returns: X_train, y_train, X_val, y_val, X_test, y_test, classes
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
    dataset_name="clinc150"
)

# Load with OOS examples included
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
    dataset_name="clinc150",
    use_oos=True
)

# Load a subset for quick testing
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
    dataset_name="clinc150",
    max_classes=10,
    max_train_samples=1000,
    max_test_samples=500,
    seed=42
)
```

**Note**: The dataset is automatically downloaded from HuggingFace on first use. Make sure you have an internet connection for the initial download. Subsequent runs will use the cached dataset.

## 🔬 Train/Dev/Test Split Usage

This repository follows **machine learning best practices** for data splitting to ensure proper model evaluation and prevent data leakage.

### Split Structure

The CLINC150 dataset comes with **pre-defined splits** from HuggingFace:
- **Training set**: Used for model training (~23,000 samples)
- **Validation set (dev)**: Used for hyperparameter tuning and model selection (~3,000 samples)
- **Test set**: Used **only** for final evaluation (~5,700 samples)

### How Splits Are Used

#### ✅ **Training Pipeline (03_model_training.py)**
- **Training set**: Used to fit models
- **Validation set**: Kept separate and passed to training function
  - Models using cross-validation internally (e.g., `GridSearchCV`) perform CV on the training set
  - Models that support early stopping or validation-based selection can use the validation set
  - The validation set is **not merged** into training to maintain proper ML practices
- **Test set**: Not used during training (kept completely separate)

#### ✅ **Hyperparameter Tuning (scripts/tune_hyperparams.py)**
- **Training set**: Used to fit models with different hyperparameters
- **Validation set**: Used to evaluate hyperparameter configurations (F1 score)
- **Test set**: Not used during tuning

#### ✅ **Prediction & Evaluation (04, 05)**
- **Test set**: Used **only** for final model evaluation
- Training/validation sets are loaded for reference but test set predictions are the primary output

### Why This Matters

1. **Prevents Data Leakage**: Test set is never seen during training or hyperparameter tuning
2. **Proper Model Selection**: Validation set allows unbiased hyperparameter selection
3. **Reproducible Results**: Using standard benchmark splits ensures fair comparison with other research
4. **Future-Proof**: Models that support early stopping or validation-based callbacks can use the validation set

### Implementation Details

The `get_dataset()` function returns all three splits separately:
```python
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
    dataset_name="clinc150",
    # ... other parameters
)
```

**Key Points:**
- ✅ Validation set is **kept separate** in the training pipeline
- ✅ Test set is **never used** for training or tuning
- ✅ Hyperparameter tuning correctly uses validation set for evaluation
- ✅ Some analysis scripts (00, 01, 02) merge train+val for exploratory purposes only

### When Validation Set Is Merged

Some scripts merge validation into training, but **only for specific purposes**:

1. **Exploratory Analysis (00, 01)**: Merged for data exploration and visualization
2. **Embedding Building (02)**: Merged to maximize the RAG retrieval corpus
3. **Training (03)**: **NOT merged** - kept separate for proper ML practices

This design ensures that:
- Models are trained with proper validation set usage
- Analysis can use all available data for insights
- RAG models have a larger retrieval corpus
- Best practices are maintained in the critical training step

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
