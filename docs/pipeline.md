# Pipeline

## Overview

Six sequential steps transform raw data into trained models and evaluations.

1. **Data Loading** - Load and validate dataset
2. **Exploratory Analysis** - Understand dataset characteristics
3. **Build Embeddings** - Generate semantic embeddings for RAG models
4. **Model Training** - Train multiple classification algorithms
5. **Model Prediction** - Generate predictions on test set
6. **Model Evaluation** - Evaluate and compare model performance

For concrete commands and preconfigured configs, see `running_experiments.md`.

## Commands

```bash
# Run full pipeline
DATASET=clinc150 VARIANT=tiny \
  uv run python scripts/pipeline/run_all.py

# Run individual steps
uv run python scripts/pipeline/00_data_loading.py
uv run python scripts/pipeline/01_exploratory_analysis.py
uv run python scripts/pipeline/02_build_embeddings.py
uv run python scripts/pipeline/03_model_training.py
uv run python scripts/pipeline/04_model_prediction.py
uv run python scripts/pipeline/05_model_evaluation.py
```

## Step Details

### Step 0: Data Loading

- **Purpose**: Load dataset from HuggingFace/GitHub and validate
- **Data**: Training, validation, and test splits
- **Outputs**: Processed dataset, class distributions, statistics
- **Saved to**: `output/runs/<label_type>/<dataset>/<variant>/dataset/`

### Step 1: Exploratory Analysis

- **Purpose**: Analyze dataset characteristics
- **Data**: Train+val merged for analysis, test for comparison
- **Outputs**: Class distributions, vocabulary analysis, drift analysis
- **Saved to**: `output/runs/<label_type>/<dataset>/<variant>/dataset/`

### Step 2: Build Embeddings

- **Purpose**: Generate embeddings for RAG models
- **Data**: Train+val merged for larger retrieval corpus
- **Processes**:
  - SBERT embeddings (default, local)
  - OpenAI embeddings (optional, requires API key)
- **Outputs**: FAISS indices and metadata
- **Saved to**: `output/runs/<label_type>/<dataset>/<variant>/features/embeddings/{backend}/`

### Step 3: Model Training

- **Purpose**: Train multiple classification models
- **Data**: Train set for fitting, validation set kept separate
- **Models**: Naive Bayes, SVM, Transformer-based, RAG variants
- **Outputs**: Trained model files
- **Saved to**: `output/runs/<label_type>/<dataset>/<variant>/models/{model_name}/`

### Step 4: Model Prediction

- **Purpose**: Generate predictions on test set
- **Data**: Test set only
- **Outputs**: Predictions with probabilities
- **Saved to**: `output/runs/<label_type>/<dataset>/<variant>/eval/{model_name}/predictions/`

### Step 5: Model Evaluation

- **Purpose**: Evaluate model performance
- **Data**: Test set ground truth + predictions
- **Outputs**: Metrics, confusion matrices, comparison plots
- **Saved to**: `output/runs/<label_type>/<dataset>/<variant>/compare/`

## Data Flow

- **Training**: Uses train set only
- **Validation**: Kept separate for hyperparameter tuning
- **Test**: Used only for final evaluation
- **Analysis/Retrieval**: Train+val merged for larger corpus (not used for training)

## Design Principles

- **Proper Data Splitting**: Strict separation of train/validation/test
- **Modularity**: Each step is independent and can be run separately
- **Reproducibility**: Fixed random seeds, deterministic algorithms
- **Flexibility**: Support multiple algorithms and configurations
- **Efficiency**: Avoid redundant computation, cache embeddings
