# Experiments

This document describes the experimental setup, data types, splits, and selection strategies used in this project.

## Dataset: CLINC150

### Overview

**The CLINC150 dataset is automatically downloaded from HuggingFace** - no manual download or setup required! The dataset is a real, production-ready benchmark dataset for intent classification with 150 intents across 10 domains.

### Dataset Features

- **Source**: HuggingFace `clinc_oos` dataset (config: `plus`) - automatically downloaded on first use
- **Content**: 150 in-scope intents across 10 domains (banking, credit cards, etc.)
- **Size**: ~23,000 training utterances, ~5,700 test utterances
- **Labels**: Intent labels as strings (e.g., "transfer_money", "greeting", "balance")
- **Optional**: Out-of-scope (OOS) examples can be included as an extra class

### Data Types

The dataset consists of:
- **Text**: User utterances (strings)
- **Labels**: Intent labels (strings, 150 classes)
- **Splits**: Pre-defined train/validation/test splits from HuggingFace

## Train/Dev/Test Split Usage

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

**Key Points**:
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

## Dataset Selection and Configuration

### Running with the Full Dataset (Default)

By default, the pipeline uses the **full CLINC150 dataset** with all 150 classes. Simply run:

```bash
# Use default config (full dataset, 10 classes experiment name)
python scripts/pipeline/run_all.py

# Or use a specific config file
CONFIG_FILE=config.yaml python scripts/pipeline/run_all.py
```

The dataset will be automatically downloaded from HuggingFace on first use (requires internet connection).

### Configuring Dataset Size

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

**Configuration Options**:
- `max_classes`: Randomly select N classes from the full dataset (useful for quick testing)
- `max_train_samples`: Limit training set size (uses stratified sampling to maintain class balance)
- `max_test_samples`: Limit test set size (uses stratified sampling)
- `use_oos`: Include out-of-scope examples as an extra class label

**Example Configurations**:

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

### Using Different Config Files

The project includes several pre-configured experiment files:

- `config/config.yaml` - Default (full dataset, 10 classes experiment name)
- `config/config_10_classes.yaml` - 10 classes experiment
- `config/config_25_classes.yaml` - 25 classes experiment
- `config/config_tiny_dataset.yaml` - Small dataset for quick testing (10 classes, 100 train, 50 test)

**To use a different config file**:

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

### Example Usage in Code

```python
from intent_classifier.datasets.dataset import get_dataset

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

## Data Exploration

The project includes comprehensive data analysis tools:
- Class distribution analysis
- Text length statistics
- Vocabulary analysis
- Stopword analysis
- Vocabulary drift analysis
- Publication-ready visualizations
- CSV export capabilities

These are generated automatically when running the exploratory analysis step (`01_exploratory_analysis.py`).
