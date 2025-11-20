"""
Hyperparameter Tuning Script (Ray Tune)

This script performs distributed hyperparameter optimization for text classification models using Ray Tune. It supports tuning of Logistic Regression and Multinomial Naive Bayes models with configurable search spaces and evaluation metrics.

Key Features:
- Distributed hyperparameter search using Ray Tune
- Supports Logistic Regression (C) and Naive Bayes (alpha) models
- Reads global YAML configuration for dataset and experiment settings
- Cross-validation and macro-F1 evaluation
- Automatic result saving and best parameter export

Usage:
- Place this script in the `scripts/` directory of your project
- Prepare a YAML configuration file with dataset and experiment parameters
- Run the script: `python scripts/tune_hyperparams.py --config path/to/config.yaml --algo lr --num-samples 30`
- Review the output directory for best hyperparameter results

This script is suitable for production and research, enabling efficient and reproducible hyperparameter optimization for text classification models.
"""

import argparse
import os
import sys
from pathlib import Path

import ray
import yaml
from ray import tune
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Add repo root to path for imports
repo_root = os.path.abspath("..")
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__)) + "/.."))

# Import from project
from src.datasets.dataset import get_dataset


def train_nb(config):
    X_train, y_train, X_val, y_val, _ = get_dataset(
        dataset_name=config["dataset"].get("name", "clinc150"),
        use_oos=config["dataset"].get("use_oos", False),
    )

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000)),
            ("nb", MultinomialNB(alpha=config["alpha"])),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


def train_lr(config):
    X_train, y_train, X_val, y_val, _ = get_dataset(
        dataset_name=config["dataset"].get("name", "clinc150"),
        use_oos=config["dataset"].get("use_oos", False),
    )

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100000)),
            ("lr", LogisticRegression(C=config["C"], max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    tune.report({"f1": f1})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--algo", choices=["nb", "lr"], default="lr")
    parser.add_argument("--num-samples", type=int, default=30)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    if args.algo == "nb":
        analysis = tune.run(
            tune.with_parameters(train_nb),
            config={**cfg, "alpha": tune.loguniform(1e-3, 1.0)},
            num_samples=args.num_samples,
            resources_per_trial={"cpu": 1},
            metric="f1",
            mode="max",
        )
    else:
        analysis = tune.run(
            tune.with_parameters(train_lr),
            config={**cfg, "C": tune.loguniform(1e-3, 10)},
            num_samples=args.num_samples,
            resources_per_trial={"cpu": 1},
            metric="f1",
            mode="max",
        )

    best = analysis.get_best_config("f1", "max")
    print("Best hyper‑parameters:", best)

    output_dir = Path("output/hyperparams_tune")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "nb":
        best_param = {"alpha": best["alpha"]}
    else:
        best_param = {"C": best["C"]}
    (output_dir / f"best_{args.algo}.yaml").write_text(yaml.dump(best_param))


if __name__ == "__main__":
    main()
