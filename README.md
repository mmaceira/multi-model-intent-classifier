# Reuters News Topic Classification & Semantic Search

A production-ready NLP pipeline for automated news article classification and semantic search, built on the Reuters-21578 corpus.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
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
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/reuters-rag-classifier.git
cd reuters-rag-classifier

# 2. Create and activate virtual environment
python -m venv venv/reuters-rag-classifier
source venv/reuters-rag-classifier/bin/activate  # On Windows: venv\reuters-rag-classifier\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Run the analysis notebooks
jupyter notebook notebooks/
```

### Basic Usage

```python
from src.model import ReutersClassifier

# Initialize the classifier
classifier = ReutersClassifier(model_type="bert_lr")

# Classify a news article
text = "Your news article text here..."
result = classifier.predict(text)
print(f"Predicted topic: {result['topic']}")
print(f"Confidence: {result['confidence']}")
```

## 📋 Overview

This project implements a comprehensive NLP pipeline for:
- Automated topic classification of news articles
- Semantic search and document retrieval
- Business insights generation
- Real-time document similarity matching
- Multi-language support

Built on the Reuters-21578 corpus, it provides a production-ready solution for news analysis and information retrieval.

## 🏗️ Project Structure

```
reuters-rag-classifier/
├── config/              # Configuration files
│   ├── config.yaml     # Main configuration file
│   └── notebook_setup.py # Notebook configuration
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 00_Data_Loading.ipynb          # Initial data loading and preprocessing
│   ├── 01_Exploratory_Analysis.ipynb  # Data exploration and visualization
│   ├── 02_Build_Embeddings.ipynb # Embedding generation and LLM integration
│   ├── 03_Model_Training.ipynb        # Model training and optimization
│   ├── 04_Model_Evaluation.ipynb      # Model evaluation and metrics
│   └── README.md                      # Notebook documentation
├── output/             # Model outputs and results
├── scripts/            # Utility scripts
├── src/                # Main source code
│   ├── algorithms/     # ML algorithms implementation
│   ├── analysis/       # Data analysis modules
│   ├── datasets/       # Dataset handling
│   ├── embeddings/     # Embedding generation
│   ├── rag/           # RAG implementation
│   └── utils/         # Utility functions
├── venv/              # Virtual environment
├── .env              # Environment variables
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## 🧠 Model Architecture

### Model Comparison

| Model | Architecture |  Use Case |
|-------|--------------|---------|
| Multinomial Naive Bayes | TF-IDF + Naive Bayes | Fast, lightweight classification |
| Linear SVM | TF-IDF + SVM | Balanced speed and accuracy |
| MiniLM + LogReg | Transformer + Logistic Regression | High-accuracy classification |

| RAG-CentroidNN | FAISS + Nearest Neighbors | Semantic search and classification |
| RAG-LLM (local) | FAISS + Local LLM | Context-aware classification |
| RAG-LLM (OpenAI) | FAISS + OpenAI LLM | Advanced semantic understanding |

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
- **Implementation**: The RAG (Retrieval-Augmented Generation) classification system combines the power of semantic search with large language models to make accurate classification decisions. Here's a detailed breakdown of how it works:

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
    - Uses OpenAI's GPT models (default: "gpt-4o-mini") for final classification
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
- Automated topic classification for news articles
- Real-time document similarity matching
- Semantic search across document collections
- Multi-language support through transformer models

### 2. Enterprise-Grade API & Interface
- RESTful API for seamless integration
- Modern web interface for document management
- Role-based access control
- Audit logging and compliance tracking

### 3. Advanced Analytics & Visualization
- Interactive topic distribution dashboards
- Document similarity networks
- Trend analysis and topic evolution
- Custom report generation

### 4. Performance Optimization
- Automated hyperparameter tuning
- Model performance monitoring
- Resource utilization optimization
- Cost-effective scaling options

## 📊 Performance Metrics

### 10-Class Experiment Results

| Model | Architecture | Accuracy | Macro-F1 | Training Time | Inference Speed | Resource Usage |
|-------|--------------|----------|----------|---------------|-----------------|----------------|
| Naive Bayes | TF-IDF + Naive Bayes | 0.82 | 0.73 | ~2 min | 60k docs/s | < 2GB RAM |
| Linear SVM | TF-IDF + SVM | 0.87 | 0.82 | 7-8 min | 12-15k docs/s | < 5GB RAM |
| MiniLM + LogReg | Transformer + Logistic Regression | 0.89 | 0.87 | ~45 min | 1k docs/s | 12GB GPU |
| RAG-CentroidNN | FAISS + Nearest Neighbors | 0.85 | 0.83 | ~25 min | 200 QPS | 16GB RAM |
| RAG-LLM (local) | FAISS + Local LLM | 0.84 | 0.82 | ~30 min | 150 QPS | 16GB RAM |
| RAG-LLM (OpenAI) | FAISS + OpenAI LLM | 0.86 | 0.84 | ~35 min | 180 QPS | 16GB RAM |

### Common Error Patterns
The most frequent misclassifications across models:
1. `earn -> acq` (191 instances)
2. `interest -> money-fx` (169 instances)
3. `acq -> earn` (164 instances)
4. `dlr -> money-fx` (100 instances)
5. `corn -> grain` (99 instances)

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

## 🛠️ Development

### Environment Setup
1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
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
pytest tests/test_model.py

# Run with coverage
pytest --cov=src
```

### Documentation
- API documentation is generated using Sphinx
- Run `make docs` to build documentation
- View documentation at `docs/_build/html/index.html`

## 🙏 Acknowledgments

- Reuters-21578 corpus
- Hugging Face Transformers
- FAISS for similarity search
- OpenAI for embedding models

---

<b>© Reuters-RAG-Classifier Project</b>

## 📊 Dataset Configuration

### Dataset Features
- **Source**: Reuters-21578 corpus (10,788 newswire articles)
- **License**: Open license for research/commercial use
- **Content**: Financial and commodities news articles
- **Labels**: Economic topics with rich domain-specific text

### Loading Options
The dataset can be loaded in two modes:

1. **Standard Split** (`split_type="standard"`)
   - Uses NLTK's standard train/test split
   - Configurable number of classes (`n_classes`)
   - Full dataset or top N most frequent classes

2. **Test Split** (`split_type="test"`)
   - Balanced dataset with equal samples per class
   - Configurable number of classes and samples
   - 70/30 train/test split per class
   - Ideal for testing and development

### Configuration Parameters
```yaml
dataset:
  split_type: "test"                         # [standard | test]
  n_classes: 25                              # number of classes to classify
  n_samples_per_class: 100                   # samples per class (null for all)
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
