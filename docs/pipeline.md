# Pipeline Documentation

This document explains the complete training and evaluation pipeline, including the main decisions made at each step: which data is used, which algorithms are applied, and why these choices were made.

## Pipeline Overview

The pipeline consists of 6 sequential steps that transform raw data into trained models and comprehensive evaluations:

1. **Data Loading** - Load and validate the CLINC150 dataset
2. **Exploratory Analysis** - Understand dataset characteristics
3. **Build Embeddings** - Generate semantic embeddings for RAG models
4. **Model Training** - Train multiple classification algorithms
5. **Model Prediction** - Generate predictions on test set
6. **Model Evaluation** - Evaluate and compare model performance

## Step-by-Step Pipeline Breakdown

### Step 0: Data Loading (`00_data_loading.py`)

**Purpose**: Load the CLINC150 dataset from HuggingFace and perform initial validation.

**Data Used**:
- **Source**: CLINC150 dataset from HuggingFace (`clinc_oos` dataset, config: `plus`)
- **Splits**: Training, validation, and test sets (pre-defined by HuggingFace)
- **For Analysis**: Training and validation sets are merged for exploratory purposes only

**Algorithms/Processes**:
- Dataset loading via HuggingFace `datasets` library
- Data validation checks
- Basic statistics generation (class distribution, utterance length)

**Key Decisions & Rationale**:

1. **Why CLINC150?**
   - Real-world benchmark dataset with 150 intents across 10 domains
   - Pre-defined train/validation/test splits ensure reproducibility
   - Standard dataset for intent classification research
   - Automatically downloadable from HuggingFace (no manual setup)

2. **Why merge train+val for analysis?**
   - Maximizes data available for exploratory analysis
   - Provides better statistics on class distributions and vocabulary
   - **Important**: This merge is ONLY for analysis. Training (step 3) keeps validation separate for proper ML practices.

3. **Why validate data early?**
   - Catch data quality issues before expensive training
   - Understand class imbalance and data characteristics
   - Generate baseline statistics for comparison

**Outputs**:
- Processed dataset loaded in memory
- Class distribution visualization
- Utterance length statistics
- Saved to `output/{experiment_name}/data_exploration/`

---

### Step 1: Exploratory Data Analysis (`01_exploratory_analysis.py`)

**Purpose**: Perform comprehensive analysis to understand dataset characteristics, potential challenges, and data quality.

**Data Used**:
- **Training + Validation** (merged): For comprehensive analysis
- **Test set**: For comparison and vocabulary drift analysis
- **Note**: Test set is only used for analysis, never for training decisions

**Algorithms/Processes**:
- Class frequency analysis
- Utterance length distribution analysis
- Vocabulary analysis (word frequencies, stopwords, intent-specific vocabulary)
- Vocabulary drift analysis (comparing train vs test vocabulary)

**Key Decisions & Rationale**:

1. **Why comprehensive vocabulary analysis?**
   - Understand domain-specific terminology
   - Identify potential vocabulary mismatch between train/test
   - Helps explain model performance differences
   - Identifies distinctive words per intent (useful for interpretability)

2. **Why vocabulary drift analysis?**
   - Detects distribution shift between train and test sets
   - Helps explain why some models might underperform
   - Important for understanding generalization

3. **Why analyze class distribution?**
   - Identifies class imbalance (affects model selection)
   - Helps understand data quality
   - Informs sampling strategies if needed

**Outputs**:
- Class distribution plots and statistics
- Vocabulary analysis (CSV and visualizations)
- Vocabulary drift analysis
- Train/test comparison statistics
- Saved to `output/{experiment_name}/data_exploration/`

---

### Step 2: Build Embeddings (`02_build_embeddings.py`)

**Purpose**: Generate dense vector embeddings for semantic search and RAG-based classification.

**Data Used**:
- **Training + Validation** (merged): All training data is indexed for RAG retrieval
- **Why merge?**: Maximizes the retrieval corpus for RAG models, improving their ability to find relevant examples

**Algorithms/Processes**:

1. **SBERT Embeddings** (default, always built):
   - **Model**: `sentence-transformers/all-MiniLM-L6-v2`
   - **Why SBERT?**:
     - Local embeddings (no API keys needed)
     - Fast inference
     - Good quality for semantic search
     - 384-dimensional vectors (efficient storage)
   - **Process**: Encode all training utterances → Build FAISS index

2. **OpenAI Embeddings** (optional, only if `OPENAI_API_KEY` is set):
   - **Model**: `text-embedding-3-small`
   - **Why optional?**:
     - Requires API key and incurs costs
     - SBERT is sufficient for most use cases
     - Only built if explicitly needed
   - **Process**: Same as SBERT but using OpenAI API

**Key Decisions & Rationale**:

1. **Why build embeddings separately?**
   - Embeddings are expensive to compute (especially for large datasets)
   - Multiple models can reuse the same embeddings
   - FAISS indexing is a one-time cost
   - Allows caching and reuse across experiments

2. **Why FAISS for indexing?**
   - Efficient approximate nearest neighbor search
   - Scales to millions of vectors
   - Fast query time (200+ QPS)
   - Industry standard for production semantic search

3. **Why merge train+val for indexing?**
   - RAG models benefit from larger retrieval corpus
   - More examples = better semantic matches
   - Training still uses proper train/val split (step 3)
   - This is a retrieval corpus, not training data

4. **Why SBERT by default?**
   - No API keys required (works out-of-the-box)
   - Local processing (privacy-preserving)
   - Good balance of quality and speed
   - Sufficient for most use cases

5. **Why MiniLM-L6-v2 specifically?**
   - Lightweight (80MB model)
   - Fast inference
   - Good quality for intent classification
   - Widely used in production systems

**Outputs**:
- FAISS index files (`index.faiss`)
- Metadata files (`meta.jsonl`) with text and labels
- Saved to `output/{experiment_name}/embeddings/sbert/` and `output/{experiment_name}/embeddings/openai/`

---

### Step 3: Model Training (`03_model_training.py`)

**Purpose**: Train multiple classification models using proper machine learning practices.

**Data Used**:
- **Training set**: Used to fit models
- **Validation set**: Kept separate for hyperparameter tuning and model selection
- **Test set**: Not used (completely separate, only for final evaluation)

**Algorithms Trained**:

The pipeline trains multiple models in parallel, each chosen for different use cases:

1. **Multinomial Naive Bayes** (`NaiveBayesClassifier`)
   - **Algorithm**: TF-IDF vectorization + Multinomial Naive Bayes
   - **Why included?**: Fast baseline, minimal resources, good for comparison
   - **Data**: Uses training set only

2. **Linear SVM** (`LinearSVMClassifier`)
   - **Algorithm**: TF-IDF (unigrams) + Linear SVM with L2 regularization
   - **Why included?**: Balanced speed and accuracy, production-ready
   - **Data**: Uses training set only

3. **Linear SVM Bigrams** (`LinearSVMClassifier` with bigrams)
   - **Algorithm**: TF-IDF (unigrams + bigrams) + Linear SVM
   - **Why included?**: Captures word order, often better accuracy than unigrams
   - **Data**: Uses training set only

4. **MiniLM + Logistic Regression** (`TransformerLogReg`)
   - **Algorithm**: MiniLM embeddings + Logistic Regression
   - **Why included?**: High accuracy, semantic understanding
   - **Data**: Uses training set only (embeddings pre-computed in step 2)

5. **Embedding + Logistic Regression** (`EmbeddingLogReg`)
   - **Algorithm**: Flexible embeddings (SBERT or OpenAI) + Logistic Regression
   - **Why included?**: Flexibility in embedding backends, same architecture as MiniLM+LogReg
   - **Data**: Uses training set only
   - **Variants**:
     - SBERT embeddings (default, local)
     - OpenAI embeddings (optional, requires API key)

6. **RAG-CentroidNN** (`RAGCentroidNN`)
   - **Algorithm**: FAISS retrieval + Centroid-based classification
   - **Why included?**: Semantic search-based classification, explainable
   - **Data**: Uses pre-built FAISS index from step 2 (train+val merged for larger corpus)

7. **RAG-kMajority** (`RAGKMajority`)
   - **Algorithm**: FAISS retrieval + k-nearest neighbor majority voting
   - **Why included?**: Simple but effective RAG approach
   - **Data**: Uses pre-built FAISS index from step 2

8. **RAG-LLM** (`RAGLLMClassifier`)
   - **Algorithm**: FAISS retrieval + LLM-based classification
   - **Why included?**: Context-aware classification, state-of-the-art approach
   - **Data**: Uses pre-built FAISS index from step 2
   - **LLM Options**:
     - **Default**: Ollama (`ollama/llama3.1:8b`) - local, no API key needed
     - **Optional**: OpenAI models (requires API key)

**Key Decisions & Rationale**:

1. **Why multiple models?**
   - Different models excel in different scenarios (speed vs accuracy)
   - Provides comprehensive comparison
   - Allows selection based on requirements (speed, accuracy, resources)

2. **Why keep validation separate?**
   - **Critical ML best practice**: Prevents overfitting
   - Validation set used for hyperparameter tuning (if done separately)
   - Models using cross-validation perform CV on training set only
   - Future-proof: Models with early stopping can use validation set

3. **Why not merge train+val for training?**
   - Would reduce validation set size (less reliable hyperparameter selection)
   - Prevents proper model selection
   - Violates ML best practices
   - Test set must remain completely untouched

4. **Why different algorithms?**
   - **Traditional ML** (Naive Bayes, SVM): Fast, interpretable, good baselines
   - **Transformer-based** (MiniLM+LogReg, Embedding+LogReg): High accuracy, semantic understanding
   - **RAG-based**: Context-aware, explainable, few-shot learning capabilities

5. **Why RAG models use merged train+val?**
   - RAG models use retrieval, not direct training
   - Larger corpus = better retrieval quality
   - This is a retrieval corpus, not training data
   - Training still respects train/val split for hyperparameter tuning

6. **Why hyperparameter tuning integration?**
   - If tuned hyperparameters exist (from `scripts/tune_hyperparams.py`), they're automatically loaded
   - Tuned hyperparameters use validation set for selection (proper ML practice)
   - Default hyperparameters used if tuning not performed

**Outputs**:
- Trained model files (`.pkl` format)
- Training logs and metadata
- Saved to `output/{experiment_name}/models/{model_name}/`

---

### Step 4: Model Prediction (`04_model_prediction.py`)

**Purpose**: Generate predictions for all trained models on the test set.

**Data Used**:
- **Test set only**: Used for final predictions
- **Training/Validation**: Loaded for reference but not used for predictions

**Algorithms/Processes**:
- Load all trained models from step 3
- Generate predictions on test set
- Save predictions with probabilities (if available)

**Key Decisions & Rationale**:

1. **Why test set only?**
   - Test set is the gold standard for evaluation
   - Training predictions are usually not needed (models are already trained)
   - Saves computation time
   - Prevents accidental data leakage

2. **Why save predictions separately?**
   - Allows evaluation without reloading models (faster)
   - Enables post-hoc analysis
   - Supports evaluation on different metrics later
   - Predictions can be shared without model files

3. **Why skip training predictions?**
   - Usually not needed for evaluation
   - Saves significant computation time
   - Can be generated later if needed

**Outputs**:
- Test set predictions (CSV format)
- Prediction probabilities (if model supports it)
- Saved to `output/{experiment_name}/predictions/{model_name}/test_predictions.csv`

---

### Step 5: Model Evaluation (`05_model_evaluation.py`)

**Purpose**: Comprehensive evaluation of all models using proper metrics and visualizations.

**Data Used**:
- **Test set**: Ground truth labels for evaluation
- **Predictions**: From step 4 (no models are loaded)

**Algorithms/Processes**:
- Load predictions from step 4
- Calculate evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score (macro and micro averages)
  - Per-class metrics
- Generate visualizations:
  - Confusion matrices
  - Model comparison plots
  - Error analysis

**Key Decisions & Rationale**:

1. **Why evaluate from predictions, not models?**
   - Faster (no model loading)
   - Allows evaluation of models trained elsewhere
   - Enables post-hoc analysis
   - Supports evaluation on different metrics

2. **Why multiple metrics?**
   - **Accuracy**: Overall performance
   - **F1-score (macro)**: Handles class imbalance better
   - **F1-score (micro)**: Overall performance across all classes
   - **Per-class metrics**: Identifies which classes are difficult

3. **Why confusion matrices?**
   - Visual representation of errors
   - Identifies common misclassifications
   - Helps understand model behavior
   - Useful for debugging and improvement

4. **Why model comparison plots?**
   - Easy visual comparison of all models
   - Helps select best model for use case
   - Identifies trade-offs (speed vs accuracy)

**Outputs**:
- Evaluation metrics (CSV format)
- Confusion matrices (PNG)
- Model comparison plots (PNG)
- Error analysis reports
- Saved to `output/{experiment_name}/results/`

---

## Pipeline Design Principles

### 1. Proper Data Splitting

**Principle**: Strict separation of train/validation/test sets

**Implementation**:
- Training uses training set only
- Validation set kept separate for hyperparameter tuning
- Test set only used for final evaluation
- Some steps merge train+val for analysis/retrieval (not training)

**Why**: Prevents data leakage, ensures proper model evaluation, follows ML best practices

### 2. Modularity

**Principle**: Each step is independent and can be run separately

**Implementation**:
- Each script checks for existing outputs and skips if present
- Steps can be re-run individually
- Outputs are clearly defined and saved

**Why**: Allows iterative development, debugging, and resuming interrupted runs

### 3. Reproducibility

**Principle**: All experiments are reproducible

**Implementation**:
- Fixed random seeds (configurable)
- Deterministic algorithms where possible
- Configuration files capture all settings
- Outputs are versioned by experiment name

**Why**: Enables fair comparison, debugging, and scientific rigor

### 4. Flexibility

**Principle**: Support multiple algorithms and configurations

**Implementation**:
- Multiple models trained in parallel
- Configurable via YAML files
- Support for different embedding backends
- Support for different LLM providers

**Why**: Allows selection based on requirements (speed, accuracy, resources)

### 5. Efficiency

**Principle**: Avoid redundant computation

**Implementation**:
- Embeddings built once and reused
- Models cached after training
- Predictions saved separately
- Skip steps if outputs exist

**Why**: Saves time and resources, especially for large datasets

## Data Flow Summary

```
Step 0: Data Loading
  └─> CLINC150 dataset (train, val, test splits)
       └─> Basic validation & statistics

Step 1: Exploratory Analysis
  └─> Train+Val (merged for analysis)
  └─> Test (for comparison only)
       └─> Comprehensive analysis & visualizations

Step 2: Build Embeddings
  └─> Train+Val (merged for larger retrieval corpus)
       └─> SBERT embeddings → FAISS index
       └─> OpenAI embeddings → FAISS index (optional)

Step 3: Model Training
  └─> Train set (for fitting models)
  └─> Val set (kept separate, for hyperparameter tuning)
  └─> FAISS indices (for RAG models)
       └─> Multiple trained models

Step 4: Model Prediction
  └─> Trained models
  └─> Test set
       └─> Predictions

Step 5: Model Evaluation
  └─> Predictions
  └─> Test set (ground truth)
       └─> Evaluation metrics & visualizations
```

## Algorithm Selection Rationale

### Traditional ML Models (Naive Bayes, SVM)

**When to use**: Speed-critical applications, resource-constrained environments, baseline comparisons

**Why chosen**:
- Fast inference (60k docs/s for Naive Bayes, 12-15k docs/s for SVM)
- Minimal resources (CPU-only, <5GB RAM)
- Good accuracy for many use cases
- Interpretable (especially linear models)

### Transformer-Based Models (MiniLM+LogReg, Embedding+LogReg)

**When to use**: High-accuracy requirements, semantic understanding important, GPU available

**Why chosen**:
- State-of-the-art accuracy
- Semantic understanding (not just word matching)
- Flexible embedding backends (local or API-based)
- Production-ready

### RAG Models (CentroidNN, kMajority, LLM)

**When to use**: Context-aware classification, explainable predictions, few-shot learning

**Why chosen**:
- Context-aware decisions (uses retrieved examples)
- Explainable (can show retrieved examples)
- Few-shot learning capabilities
- Flexible LLM providers (local Ollama or API-based)

## Configuration and Customization

All pipeline decisions can be customized via configuration files:

- **Dataset**: Main config file (e.g., `config/dataset/clinc150/tiny.yaml`) - dataset size, classes, splits
- **Models**: `config/models_config.yaml` (which models to train, hyperparameters)
- **Embeddings**: Step 2 automatically detects available backends
- **LLM**: Main config file (Ollama by default, OpenAI optional)

See [Configuration](configuration.md) for detailed configuration options.

## Running the Pipeline

### Complete Pipeline

```bash
python scripts/pipeline/run_all.py
```

### Individual Steps

```bash
python scripts/pipeline/00_data_loading.py
python scripts/pipeline/01_exploratory_analysis.py
python scripts/pipeline/02_build_embeddings.py
python scripts/pipeline/03_model_training.py
python scripts/pipeline/04_model_prediction.py
python scripts/pipeline/05_model_evaluation.py
```

See [Running Experiments](running_experiments.md) for more details.

## Related Documentation

- [Algorithms](algorithms.md) - Detailed algorithm descriptions
- [Experiments](experiments.md) - Dataset and data splitting details
- [Running Experiments](running_experiments.md) - How to run the pipeline
- [Configuration](configuration.md) - Configuration options
- [Hyperparameter Tuning](hyperparameter_tuning.md) - Hyperparameter optimization
- [Model Architecture](model_architecture.md) - Technical model specifications
