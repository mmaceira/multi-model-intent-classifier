# Running Experiments

This guide explains how to run experiments, from quick start to advanced usage.

## Quick Start

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

## Running the Complete Pipeline

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

## Running Individual Pipeline Steps

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

## Basic Usage in Code

```python
from src.datasets.dataset import get_dataset

# Load CLINC150 dataset
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

# Use with any classifier
from src.algorithms.linear_svm import LinearSVMClassifier
classifier = LinearSVMClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

## Using Different Configuration Files

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

## Hyperparameter Tuning

Before running experiments, you may want to tune hyperparameters. See [Hyperparameter Tuning](hyperparameter_tuning.md) for detailed instructions.

**Quick start for hyperparameter tuning:**

```bash
# Tune all models with default settings
python scripts/tune_hyperparams.py --config config/config.yaml --all

# Tune with more samples for better results
python scripts/tune_hyperparams.py --config config/config.yaml --all --num-samples 50

# Tune specific model
python scripts/tune_hyperparams.py --config config/config.yaml --algo svm --num-samples 30
```

**The tuned hyperparameters are automatically loaded and used during training!**

## Model Selection

Control which models are trained by editing `config/models_config.yaml`:

```yaml
models:
  naive_bayes:
    enabled: true  # Set to false to skip this model
    name: "Naive Bayes"
    class: "NaiveBayesClassifier"

  linear_svm:
    enabled: true
    name: "Linear SVM"
    class: "LinearSVMClassifier"
```

Set `enabled: false` to skip training a specific model.

## Output Locations

All outputs are saved to `output/{run_name}/` where `run_name` comes from your config file:

- `output/{run_name}/data_exploration/` - Data analysis visualizations and statistics
- `output/{run_name}/embeddings/` - Generated embeddings (SBERT/OpenAI)
- `output/{run_name}/models/` - Trained model files
- `output/{run_name}/predictions/` - Model predictions
- `output/{run_name}/results/` - Evaluation results and comparisons

## Running with Different Embedding Backends

### SBERT (Default, Local)

No additional setup needed. SBERT embeddings are generated locally.

### OpenAI Embeddings

Requires `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
python scripts/pipeline/run_all.py
```

Then set `use_openai: true` in your model configuration for models that support it.

## Running with Different LLM Providers

### Ollama (Default, Local)

```bash
# Start Ollama
ollama serve

# Pull model
ollama pull llama3.1:8b

# Run pipeline
python scripts/pipeline/run_all.py
```

### OpenAI

Requires `OPENAI_API_KEY` and configuration update:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Update `config/config.yaml`:
```yaml
model:
  llm_model: "gpt-4o-mini"  # Change from "ollama/llama3.1:8b"
```

See [LLM Providers](llm_providers.md) for more details.

## Troubleshooting

### Dataset Download Issues

If the dataset download fails:
- Check your internet connection
- Verify HuggingFace access (may require login for some datasets)
- Check disk space (dataset is cached after first download)

### Out of Memory

If you run out of memory:
- Use a smaller dataset: `CONFIG_FILE=config_tiny_dataset.yaml`
- Reduce `max_train_samples` in your config
- Disable GPU-intensive models in `config/models_config.yaml`

### Model Training Fails

- Check that previous pipeline steps completed successfully
- Verify embeddings were generated (check `output/{run_name}/embeddings/`)
- Ensure required dependencies are installed
- Check logs for specific error messages

### Slow Performance

- Use smaller dataset for testing
- Disable models you don't need
- Use CPU-only models (Naive Bayes, Linear SVM) for faster iteration
- Consider using OpenAI embeddings if local GPU is slow

## Resource & Stability

For full dataset runs, the following mitigations are in place to prevent crashes and memory issues:

### Memory Optimizations

1. **Vectors Not Stored in Metadata**: Embeddings are stored only in FAISS indices, not in metadata JSONL files. This prevents several GB of RAM/disk usage.

2. **Streaming Metadata Writing**: Metadata is written to disk in a streaming fashion to reduce peak RAM usage.

3. **Configurable Batch Sizes**: Control embedding batch sizes via environment variables:
   ```bash
   export SBERT_BATCH=32      # Default: 32 (was 64)
   export OPENAI_BATCH=32     # Default: 32 (was 50)
   ```

4. **Capped Parallelism**: TransformerLogReg uses `n_jobs=2` and `cv=3` by default (instead of `-1` and `5`) to prevent oversubscription of CPU cores and RAM.

5. **BLAS Thread Limits**: For full dataset runs, set these environment variables to prevent thread oversubscription:
   ```bash
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   ```

### RAG-LLM Configuration

- **RAG-LLM is disabled by default** in `config/models_config.yaml` for full dataset runs (uses too much RAM/VRAM with 8B models).
- To enable for small configs (tiny/10/25-class), set `rag_llm.enabled: true` in `config/models_config.yaml`.
- For full runs with RAG-LLM, consider using a smaller model:
  ```yaml
  model:
    llm_model: "ollama/qwen2.5:0.5b"  # or "ollama/phi3:mini"
  ```

### Resuming Interrupted Runs

If a run is interrupted:
- **Embeddings**: Check if `output/{run_name}/embeddings/` exists. If present, embeddings step will be skipped.
- **Models**: Check `output/{run_name}/models/` - existing models are automatically skipped during training.
- **Force Rebuild**: To rebuild embeddings, set `FORCE_REBUILD=1`:
  ```bash
  FORCE_REBUILD=1 python scripts/pipeline/02_build_embeddings.py
  ```

### Resource Requirements for Full Dataset

- **RAM**: 16GB+ recommended (8GB minimum with optimizations)
- **Disk**: ~5GB for embeddings and models
- **CPU**: Multi-core recommended for parallel training
- **GPU**: Optional (only for TransformerLogReg if using GPU)

### Sanity Checks

After building embeddings, verify:
- FAISS index size matches metadata line count
- Metadata JSONL files do NOT contain `"vector"` keys (vectors are only in FAISS)
- Check `output/{run_name}/embeddings/*/meta.jsonl` - should only have `id`, `label`, `text` fields

## Best Practices

1. **Start Small**: Use `config_tiny_dataset.yaml` for initial testing
2. **Tune Hyperparameters**: Run hyperparameter tuning before final training
3. **Check Outputs**: Review data exploration outputs before training
4. **Version Control**: Commit your config files and hyperparameters
5. **Monitor Resources**: Watch memory and disk usage during training
6. **Use Validation Set**: Never use test set for model selection
7. **Set Environment Variables**: For full runs, set BLAS thread limits and batch sizes as shown above
