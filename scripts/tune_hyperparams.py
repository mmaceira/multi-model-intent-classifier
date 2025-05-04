"""scripts/tune_hyperparams.py
Quick CPU‑only hyper‑parameter sweep using Ray Tune.
Tunes:
  • C for Logistic Regression
  • alpha for Multinomial Naïve Bayes
Reads global YAML config for dataset paths.
"""

import argparse
import yaml
from pathlib import Path
import ray
from ray import tune
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import numpy as np
import os
import sys

# Add repo root to path for imports
repo_root = os.path.abspath('..')
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__)) + '/..'))

# Import from project
from src.datasets.dataset import get_dataset

def train_nb(config):
    X_train, y_train, X_val, y_val, _ = get_dataset(
        split_type=config['dataset']['split_type'],
        n_classes=config['general']['n_classes'],
        cutoff_year=config['dataset'].get('cutoff_year', None)
    )
    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000)),
        ('nb', MultinomialNB(alpha=config['alpha']))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    tune.report({"f1": f1})

def train_lr(config):
    X_train, y_train, X_val, y_val, _ = get_dataset(
        split_type=config['dataset']['split_type'],
        n_classes=config['general']['n_classes'],
        cutoff_year=config['dataset'].get('cutoff_year', None)
    )
    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100000)),
        ('lr', LogisticRegression(C=config['C'], max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    tune.report({"f1": f1})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--algo', choices=['nb', 'lr'], default='lr')
    parser.add_argument('--num-samples', type=int, default=30)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    
    # Set environment variables for dataset loading if needed
    os.environ['N_CLASSES'] = str(cfg['general']['n_classes'])

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    if args.algo == 'nb':
        analysis = tune.run(
            tune.with_parameters(train_nb),
            config={**cfg, 'alpha': tune.loguniform(1e-3, 1.0)},
            num_samples=args.num_samples,
            resources_per_trial={'cpu': 1},
            metric="f1",
            mode="max"
        )
    else:
        analysis = tune.run(
            tune.with_parameters(train_lr),
            config={**cfg, 'C': tune.loguniform(1e-3, 10)},
            num_samples=args.num_samples,
            resources_per_trial={'cpu': 1},
            metric="f1",
            mode="max"
        )

    best = analysis.get_best_config("f1", "max")
    print('Best hyper‑parameters:', best)

    output_dir = Path('output/hyperparams_tune')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.algo == 'nb':
        best_param = {'alpha': best['alpha']}
    else:
        best_param = {'C': best['C']}
    (output_dir / f'best_{args.algo}.yaml').write_text(yaml.dump(best_param))

if __name__ == '__main__':
    main()
