# CLINC150 Intent Classification & Semantic Search

A production-ready NLP pipeline for automated intent classification and semantic search, built on the CLINC150 dataset.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🚀 Quick Start

**The CLINC150 dataset is automatically downloaded from HuggingFace** - no manual setup required!

```bash
# 1. Install dependencies
uv sync  # or: pip install -e .

# 2. (Optional) Set up Ollama for RAG-LLM models
ollama serve
ollama pull llama3.1:8b

# 3. Run the pipeline - dataset downloads automatically!
python scripts/pipeline/run_all.py
```

That's it! The pipeline will automatically download CLINC150, train all enabled models, and generate predictions and evaluations.

**For quick testing with a smaller dataset:**
```bash
CONFIG_FILE=config_tiny_dataset.yaml python scripts/pipeline/run_all.py
```

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) folder:

- **[Installation](docs/installation.md)** - Detailed installation instructions and dependencies
- **[Algorithms](docs/algorithms.md)** - Algorithms used, implementation details, and why they were chosen
- **[Experiments](docs/experiments.md)** - Data types, splits, and selection strategies
- **[Running Experiments](docs/running_experiments.md)** - How to run experiments from quick start to advanced usage
- **[Configuration](docs/configuration.md)** - Configuration files and settings
- **[Hyperparameter Tuning](docs/hyperparameter_tuning.md)** - Hyperparameter optimization guide
- **[Model Architecture](docs/model_architecture.md)** - Detailed model architecture specifications
- **[LLM Providers](docs/llm_providers.md)** - Switching between Ollama, OpenAI, and other providers
- **[Performance](docs/performance.md)** - Performance metrics, resource requirements, and scaling
- **[Development](docs/development.md)** - Development setup, code style, and contributing

## 📋 Overview

This project implements a comprehensive NLP pipeline for:
- Automated intent classification of user utterances
- Semantic search and document retrieval
- Business insights generation
- Real-time document similarity matching
- Multi-language support

Built on the CLINC150 dataset, it provides a production-ready solution for intent classification and information retrieval. The system combines traditional machine learning approaches with modern transformer-based models and Retrieval-Augmented Generation (RAG) techniques.

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
├── docs/                # Documentation
├── scripts/             # Utility scripts
│   ├── pipeline/        # Training pipeline scripts
│   ├── api/            # FastAPI implementation
│   └── tune_hyperparams.py
├── output/             # Model outputs and results
├── src/                # Main source code
│   ├── algorithms/     # ML algorithms implementation
│   ├── datasets/       # Dataset handling
│   ├── embeddings/     # Embedding generation
│   ├── evaluation/     # Model evaluation tools
│   └── rag/            # RAG implementation
└── tests/              # Test files
```

## 🧠 Models

The pipeline includes multiple classification algorithms:

| Model | Architecture | Use Case |
|-------|--------------|---------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |
| Embedding + LogReg | Flexible embeddings (SBERT/OpenAI) + Logistic Regression | High-accuracy with flexible embedding backend |
| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | Context-aware classification |

See [Algorithms](docs/algorithms.md) and [Model Architecture](docs/model_architecture.md) for detailed information.

## 💡 Key Features

- **Multiple Algorithms**: From traditional ML to modern transformer-based and RAG models
- **Flexible Embeddings**: Support for SBERT (local) and OpenAI (API) embeddings
- **LLM Integration**: Support for Ollama (local) and OpenAI/Anthropic (API) via litellm
- **Hyperparameter Tuning**: Automated hyperparameter optimization with Ray Tune
- **Production-Ready**: Type-safe configuration, calibrated probabilities, comprehensive logging
- **Best Practices**: Proper train/validation/test splits, no data leakage

## 🚀 Running the Pipeline

### Complete Pipeline

```bash
# Using the entry point (after installation)
multi-model-pipeline

# Or directly with Python
python scripts/pipeline/run_all.py
```

### Individual Steps

```bash
python scripts/pipeline/00_data_loading.py          # Load and validate dataset
python scripts/pipeline/01_exploratory_analysis.py   # Analyze dataset characteristics
python scripts/pipeline/02_build_embeddings.py        # Generate embeddings
python scripts/pipeline/03_model_training.py          # Train models
python scripts/pipeline/04_model_prediction.py        # Generate predictions
python scripts/pipeline/05_model_evaluation.py       # Evaluate model performance
```

See [Running Experiments](docs/running_experiments.md) for more details.

## ⚙️ Configuration

The project uses YAML configuration files. See [Configuration](docs/configuration.md) for details.

**Quick example:**
```bash
# Use a different config file
CONFIG_FILE=config_tiny_dataset.yaml python scripts/pipeline/run_all.py
```

## 🎯 Hyperparameter Tuning

Tune hyperparameters before training:

```bash
# Tune all models
python scripts/tune_hyperparams.py --config config/config.yaml --all

# Tune specific model
python scripts/tune_hyperparams.py --config config/config.yaml --algo svm --num-samples 30
```

Tuned hyperparameters are automatically used during training. See [Hyperparameter Tuning](docs/hyperparameter_tuning.md) for details.

## 📊 Performance

See [Performance](docs/performance.md) for detailed metrics and scaling guidance.

**Quick reference:**
- **Naive Bayes**: Fastest inference (60k docs/s), minimal resources
- **Linear SVM**: Balanced speed and accuracy (12-15k docs/s)
- **Transformer Models**: Highest accuracy (~1k docs/s, requires GPU)
- **RAG Models**: Context-aware classification (150-200 QPS)

## 🛠️ Development

See [Development](docs/development.md) for development setup, code style, and contributing guidelines.

**Quick setup:**
```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Format code
black src/ scripts/
```

## 🙏 Acknowledgments

- CLINC150 dataset (HuggingFace)
- Hugging Face Transformers
- FAISS for similarity search
- OpenAI for embedding models
- litellm for multi-provider LLM support

## 📖 Additional Resources

- [Installation Guide](docs/installation.md)
- [Algorithms Documentation](docs/algorithms.md)
- [Experiments Guide](docs/experiments.md)
- [Running Experiments](docs/running_experiments.md)
- [Configuration Guide](docs/configuration.md)
- [Hyperparameter Tuning](docs/hyperparameter_tuning.md)
- [Model Architecture](docs/model_architecture.md)
- [LLM Providers](docs/llm_providers.md)
- [Performance Guide](docs/performance.md)
- [Development Guide](docs/development.md)
