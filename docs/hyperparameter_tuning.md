## Hyperparameter Tuning

This project includes a simple, configurable hyperparameter tuning script that follows standard ML practice: **tune on validation, evaluate on test only at the end**.

## What the tuner does

The script `scripts/tune_hyperparams.py` searches over hyperparameters for the main models using the validation set:

- **Naive Bayes**: tunes `alpha` (smoothing).
- **Linear SVM / SVM bigrams**: tunes `C` (regularization strength).
- **Transformer LogReg**: tunes `C` for the logistic regression on MiniLM embeddings.
- **Embedding LogReg**: tunes `C` for the logistic regression on flexible embeddings (SBERT / OpenAI).
- **RAG k‑majority / RAG LLM**: tunes `top_k` (neighbors).
- **RAG centroid**: typically evaluated with a default configuration.

## How to run tuning

**Tune all models for a given config:**

```bash
python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all
```

You can increase `--num-samples` to explore more configurations:

```bash
python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all --num-samples 50
```

**Tune a specific model:**

```bash
python scripts/tune_hyperparams.py --config config/config.yaml --algo nb
python scripts/tune_hyperparams.py --config config/config.yaml --algo svm
python scripts/tune_hyperparams.py --config config/config.yaml --algo embedding_logreg
```

For embedding models that can use OpenAI, set `OPENAI_API_KEY` and toggle the corresponding config flags as usual.

## How tuned values are used

Tuned hyperparameters are **loaded automatically** by the training pipeline:

1. Run tuning (optional but recommended):

```bash
python scripts/tune_hyperparams.py --config config/dataset/clinc150/tiny.yaml --all
```

2. Run training:

```bash
python scripts/pipeline/run_all.py
```

The model loader looks for per‑model files under `config/algorithm/hyperparameters/{config_name}/`. If present, those values override defaults; otherwise, it falls back to the values from your config files.

## Where results are stored

For each config (e.g. `config/dataset/clinc150/tiny.yaml`), best hyperparameters are written as small YAML files:

- **Primary (used by training)**:
  - `config/algorithm/hyperparameters/{config_name}/best_{model_name}.yaml`
- **Secondary (reference only)**:
  - `output/hyperparams_tune/{config_name}/best_{model_name}.yaml`

Using `{config_name}` in the path keeps hyperparameters from different experiments isolated.

## Versioning tuned hyperparameters

Hyperparameter YAMLs in `config/algorithm/hyperparameters/` are:

- Small, reproducible, and safe to commit.
- Useful documentation of what was actually used in experiments.

Typical flow to commit:

```bash
git add config/algorithm/hyperparameters/*/
git commit -m "Add tuned hyperparameters"
```

Files under `output/` remain experiment artifacts and are usually ignored by git.

## Best practices

- **Tune before final training** for each important config.
- **Never use the test set during tuning**; it is reserved for final evaluation only.
- **Keep configs separate**: tuned values are stored per config name, matching how you structure experiments in `config/`.
