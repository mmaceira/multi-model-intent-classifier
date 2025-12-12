## Experiments

This page summarizes **available datasets**, how splits work, and which preset experiment configs exist. It is intentionally brief and focuses on choices you need to make when running experiments.

## Available Datasets

### CLINC150

- **Source**: HuggingFace `clinc_oos` (config `plus`), downloaded automatically on first use.
- **Content**: 150 intents across multiple domains, plus optional out‑of‑scope (OOS) examples.
- **Structure**: text utterances + string intent labels (single-label), with predefined train/validation/test splits.
- **Size**: ~22,500 examples total.

The dataset is fetched automatically by the pipeline; you do not need to download anything manually.

### NLU++

- **Source**: GitHub `PolyAI-LDN/task-specific-datasets` (nlupp config), downloaded automatically on first use.
- **Content**: 68 intents across banking and hotels domains.
- **Structure**: text utterances + list of intent labels (multilabel format), with cross-validation folds that are combined and split into train/validation/test.
- **Size**: ~25,715 examples total.
- **Note**: NLU++ is always multilabel (each example can have multiple intent labels).

The dataset is fetched automatically from GitHub; you do not need to download anything manually.

### Tandem GO

- **Source**: Local CSV file (`data/Tandem GO_ Datasets RAG - Classificació v2.csv`).
- **Content**: Multi-label classification dataset with tags extracted from name and description fields.
- **Structure**: text utterances (combined from name and description) + comma-separated tags (multilabel format), with train/validation/test splits created from the combined data.
- **Size**: Varies based on CSV file content.
- **Note**: Tandem GO is always multilabel (each example can have multiple tags). The dataset requires the CSV file to be present in the `data/` directory.

The dataset is loaded from a local CSV file; ensure the file exists before running experiments.

## Train/validation/test usage

The project follows standard ML practice:

- **Train set**: used to fit models.
- **Validation set**: used for hyperparameter tuning and model selection.
- **Test set**: used only for final evaluation.

Key rules:

- Training and tuning never touch the test set.
- Validation remains separate for model selection; it is not merged into training.
- Some analysis and embedding‑building scripts temporarily merge train+val to get a larger corpus, but this merged data is **not** used as a replacement for proper train/val splitting during training.

Typical loading pattern:

```python
from intent_classifier.datasets.dataset import get_dataset

# Single-label dataset (CLINC150)
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="clinc150")

# Multilabel dataset (NLU++)
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="nlu_plus")
# Note: y_train, y_val, y_test are lists of lists (multilabel format)

# Multilabel dataset (Tandem GO)
X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="tandem_go")
# Note: y_train, y_val, y_test are lists of lists (multilabel format)
```

## Dataset size and experiment variants

You control dataset size through the `dataset:` section of your config:

- `name`: `"clinc150"` or `"nlu_plus"`.
- `use_oos`: whether to include OOS examples as an extra class (CLINC150 only).
- `multilabel`: whether to use multilabel format (NLU++ is always multilabel).
- `max_classes`: limit the number of intents (subset of available classes).
- `max_train_samples`, `max_test_samples`, `max_val_samples`: cap the number of examples per split (with stratified sampling for single-label, random sampling for multilabel).

Predefined experiment configs:

- `config/dataset/clinc150/default.yaml`: standard full‑dataset run
- `config/dataset/clinc150/tiny.yaml`: very small subset for fast iteration
- `config/dataset/nlu_plus/default.yaml`: NLU++ standard config
- `config/dataset/nlu_plus/tiny.yaml`: NLU++ quick testing config
- `config/dataset/tandem_go/default.yaml`: Tandem GO standard config
- `config/dataset/tandem_go/tiny.yaml`: Tandem GO quick testing config

Switching configs:

```bash
CONFIG_FILE=config/dataset/clinc150/default.yaml python scripts/pipeline/run_all.py
```

For complete configuration details and all available keys, see `configuration.md`.

## Data exploration outputs

The exploratory analysis step (`01_exploratory_analysis.py`) automatically generates:

- Class distributions.
- Text length statistics.
- Vocabulary and stopword analysis.
- Train/test comparison and vocabulary drift views.

These artifacts are written under `output/{run_name}/data_exploration/` and are useful to sanity‑check your experiments before relying on evaluation metrics.
