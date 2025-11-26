## Experiments

This page summarizes **how CLINC150 is used**, how splits work, and which preset experiment configs exist. It is intentionally brief and focuses on choices you need to make when running experiments.

## Dataset: CLINC150

- **Source**: HuggingFace `clinc_oos` (config `plus`), downloaded automatically on first use.
- **Content**: 150 intents across multiple domains, plus optional out‑of‑scope (OOS) examples.
- **Structure**: text utterances + string intent labels, with predefined train/validation/test splits.

The dataset is fetched automatically by the pipeline; you do not need to download anything manually.

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

X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(dataset_name="clinc150")
```

## Dataset size and experiment variants

You control dataset size through the `dataset:` section of your config:

- `name`: usually `"clinc150"`.
- `use_oos`: whether to include OOS examples as an extra class.
- `max_classes`: limit the number of intents (subset of 150).
- `max_train_samples`, `max_test_samples`: cap the number of examples per split (with stratified sampling).

Predefined experiment configs:

- `config/config.yaml`: default full‑dataset run.
- `config/config_10_classes.yaml`: 10‑class subset.
- `config/config_25_classes.yaml`: 25‑class subset.
- `config/config_tiny_dataset.yaml`: very small subset for fast iteration.

Switching configs:

```bash
CONFIG_FILE=config/config_25_classes.yaml python scripts/pipeline/run_all.py
```

For complete configuration details and all available keys, see `configuration.md`.

## Data exploration outputs

The exploratory analysis step (`01_exploratory_analysis.py`) automatically generates:

- Class distributions.
- Text length statistics.
- Vocabulary and stopword analysis.
- Train/test comparison and vocabulary drift views.

These artifacts are written under `output/{run_name}/data_exploration/` and are useful to sanity‑check your experiments before relying on evaluation metrics.
