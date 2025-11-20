# Training Pipeline Scripts

This directory contains Python scripts that implement the complete training and evaluation pipeline. These scripts can be run from the command line and are suitable for automation and batch processing.

## Scripts Overview

The scripts are designed to be run in sequence, as each builds upon the work of the previous ones:

### 00_data_loading.py
- **Purpose**: Load and preprocess the CLINC150 intent classification dataset
- **Key Components**:
  - Dataset acquisition from HuggingFace (clinc_oos dataset)
  - Data validation and quality checks
  - Basic statistics generation
- **Outputs**:
  - Processed dataset and statistics in `output/experiment_<name>/data_exploration/`
- **Usage**:
  ```bash
  python scripts/pipeline/00_data_loading.py
  ```

### 01_exploratory_analysis.py
- **Purpose**: Analyze the dataset to understand its characteristics
- **Key Components**:
  - Intent distribution analysis
  - Utterance length distribution
  - Vocabulary analysis and word frequencies
  - Vocabulary drift analysis
- **Outputs**:
  - Visualizations and analysis reports in `output/experiment_<name>/data_exploration/`
- **Usage**:
  ```bash
  python scripts/pipeline/01_exploratory_analysis.py
  ```

### 02_build_embeddings.py
- **Purpose**: Generate embeddings for the dataset using transformer models
- **Key Components**:
  - SBERT embeddings (built by default, no API keys needed)
  - OpenAI embeddings (optional, only if `OPENAI_API_KEY` is set)
  - Vector generation for utterances
  - FAISS index creation
- **Outputs**:
  - Embeddings and FAISS indices in `output/experiment_<name>/embeddings/`
- **Usage**:
  ```bash
  python scripts/pipeline/02_build_embeddings.py
  ```
- **Note**: SBERT embeddings are built by default. OpenAI embeddings are optional and only built if `OPENAI_API_KEY` is set.

### 03_model_training.py
- **Purpose**: Train and optimize various classification models
- **Key Components**:
  - Model training procedures
  - Model persistence
- **Outputs**:
  - Trained models and training logs in `output/experiment_<name>/models/`
- **Usage**:
  ```bash
  python scripts/pipeline/03_model_training.py
  ```

### 04_model_prediction.py
- **Purpose**: Demonstrate model inference and prediction capabilities
- **Key Components**:
  - Model loading and prediction pipeline
  - Batch processing
- **Outputs**:
  - Prediction examples and inference benchmarks in `output/experiment_<name>/predictions/`
- **Usage**:
  ```bash
  python scripts/pipeline/04_model_prediction.py
  ```

### 05_model_evaluation.py
- **Purpose**: Evaluate model performance and analyze results
- **Key Components**:
  - Evaluation metrics calculation
  - Confusion matrix visualization
  - Model comparison
- **Outputs**:
  - Evaluation reports and error analysis in `output/experiment_<name>/results/`
- **Usage**:
  ```bash
  python scripts/pipeline/05_model_evaluation.py
  ```

## How to Run

### Running Individual Scripts

Run each script individually from the project root:

```bash
# Step 1: Load and validate data
python scripts/pipeline/00_data_loading.py

# Step 2: Perform exploratory analysis
python scripts/pipeline/01_exploratory_analysis.py

# Step 3: Build embeddings (SBERT by default, OpenAI optional if API key is set)
python scripts/pipeline/02_build_embeddings.py

# Step 4: Train models (will skip models requiring API keys if not set)
python scripts/pipeline/03_model_training.py

# Step 5: Run predictions
python scripts/pipeline/04_model_prediction.py

# Step 6: Evaluate models
python scripts/pipeline/05_model_evaluation.py
```

### Running the Complete Pipeline

**Option 1: Use the provided runner script**

```bash
# From the project root
python scripts/pipeline/run_all.py
```

**Option 2: Run scripts manually in sequence**

```bash
# From the project root
python scripts/pipeline/00_data_loading.py && \
python scripts/pipeline/01_exploratory_analysis.py && \
python scripts/pipeline/02_build_embeddings.py && \
python scripts/pipeline/03_model_training.py && \
python scripts/pipeline/04_model_prediction.py && \
python scripts/pipeline/05_model_evaluation.py
```

**Option 3: Use a bash loop**

```bash
#!/bin/bash
# From the project root
for script in scripts/pipeline/0*.py; do
    echo "Running $script..."
    python "$script"
    if [ $? -ne 0 ]; then
        echo "Error running $script"
        exit 1
    fi
done
```

### Important Notes

1. **Scripts must be run from the project root directory** (where `config/` and `src/` directories are located)

2. **Ollama is the default**: The pipeline works out-of-the-box with Ollama (local, no API keys needed):
   - SBERT embeddings are built by default
   - RAG-LLM models use Ollama by default
   - OpenAI models are disabled by default and optional
   - If you want OpenAI embeddings/models, set `OPENAI_API_KEY` and enable them in config

3. **Output directories are created automatically** based on your configuration

4. **Each script can be run independently**, but they expect outputs from previous scripts:
   - `01_exploratory_analysis.py` needs data from `00_data_loading.py`
   - `03_model_training.py` needs embeddings from `02_build_embeddings.py`
   - `04_model_prediction.py` needs trained models from `03_model_training.py`
   - `05_model_evaluation.py` needs predictions from `04_model_prediction.py`

## Prerequisites

### 1. Install Dependencies

First, make sure all dependencies are installed:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configuration Setup

The scripts use the configuration system defined in `config/config.yaml`. Make sure your configuration is properly set up before running the scripts.

The configuration file defines:
- Experiment name and paths
- Dataset settings (CLINC150 by default)
- Model configurations
- Embedding settings

### 3. Ollama Setup (Required for RAG-LLM models)

**Ollama is the default LLM provider** - make sure Ollama is installed and running:

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Start Ollama service (if not already running)
ollama serve

# Pull the default model (if not already downloaded)
ollama pull llama3.1:8b
```

**Note**: The pipeline works completely with Ollama - no API keys needed! All models use local resources by default.

### 4. Optional: OpenAI Setup

OpenAI models are **disabled by default** and completely optional. If you want to use OpenAI:

1. **Set the API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Enable OpenAI models** in `config/models_config.yaml`:
   ```yaml
   models:
     openai_logreg:
       enabled: true  # Enable OpenAI model
     rag_llm:
       enabled: true
       params:
         use_openai: true  # Use OpenAI embeddings (requires OPENAI_API_KEY)
   ```

3. **OpenAI embeddings** in `02_build_embeddings.py` will be built automatically if `OPENAI_API_KEY` is set.

**Models that can use OpenAI** (all disabled/optional by default):
- `OpenAI + LogReg` - Uses OpenAI embeddings (disabled by default)
- `RAG-LLM` - Flexible model that works with:
  - **Default**: Ollama LLM + SBERT embeddings (no API keys needed)
  - **Option 1**: Ollama LLM + OpenAI embeddings (set `use_openai: true`, requires OPENAI_API_KEY)
  - **Option 2**: OpenAI LLM + SBERT embeddings (change `model` to OpenAI model, requires OPENAI_API_KEY)
  - **Option 3**: OpenAI LLM + OpenAI embeddings (change `model` and set `use_openai: true`, requires OPENAI_API_KEY)
- OpenAI embeddings in `02_build_embeddings.py` (optional, only if OPENAI_API_KEY is set)

## Output Directory Structure

All outputs are organized by experiment. Each experiment (e.g., `experiment_10_classes`, `experiment_25_classes`, etc.) contains the following subdirectories:

- `data_exploration/`: Data statistics, class distributions, and exploratory analysis outputs
- `embeddings/`: Embedding vectors and FAISS indices (organized by embedding type, e.g., `openai/`, `sbert/`)
- `models/`: Trained model files and training logs (organized by model type)
- `predictions/`: Model predictions (organized by model type)
- `results/`: Evaluation metrics, error analysis, and visualizations

## Troubleshooting

### Common Issues

**1. "No API key provided" errors**
- **Solution**: OpenAI models are disabled by default. The pipeline works fine without OpenAI. If you want to use OpenAI models, enable them in `config/models_config.yaml` and set `OPENAI_API_KEY` environment variable.

**2. "Module not found" errors**
- **Solution**: Make sure you're running from the project root and dependencies are installed:
  ```bash
  uv sync  # or pip install -e .
  ```

**3. "Configuration file not found"**
- **Solution**: Make sure you're running scripts from the project root directory where `config/` folder exists.

**4. Models failing to initialize**
- **Solution**: Check the logs for specific error messages. Common causes:
  - Ollama not running (for RAG-LLM models): Start `ollama serve`
  - Missing embeddings (run `02_build_embeddings.py` first)
  - Configuration errors (check `config/models_config.yaml`)
  - OpenAI models enabled but no API key (disable them or set API key)

**5. CUDA/GPU errors**
- **Solution**: The scripts will fall back to CPU if GPU is not available. For GPU support, ensure PyTorch with CUDA is installed.

### Getting Help

- Check the logs: Each script prints detailed information about what it's doing
- Review configuration: Ensure `config/config.yaml` and `config/models_config.yaml` are correct
- Check prerequisites: Verify all dependencies are installed and services (like Ollama) are running

## Features

- Scripts work out-of-the-box with Ollama (no API keys needed)
- OpenAI models are disabled by default and completely optional
- Scripts are designed for command-line execution
- All output is printed to stdout/stderr (no interactive display)
- Scripts can be easily integrated into automation pipelines
- Error handling is explicit - models that fail to initialize are skipped with warnings
- Scripts can be run in headless environments
