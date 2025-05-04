"""High‑level qualitative analysis utilities.

This module expects an *experiment directory* that contains a
`predictions/` sub‑folder with one CSV per model:
    <model_name>_preds.csv

Each CSV must have, at minimum:
    id, text, true_label, pred_label[, confidence]

The helper will:
  • save top‑N high‑confidence mis‑classifications
  • draw and save a confusion matrix
  • export a per‑class classification report
"""

from pathlib import Path
import glob

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def _save_confusion_matrix(cm, labels, title, out_path):
    """Render a confusion‑matrix with matplotlib (no seaborn)."""
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)

    # write counts inside cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _export_top_errors(df, model_name, output_dir, top_n=20):
    """Save a CSV with the top‑N most confident wrong predictions."""
    wrong = df[df['true_label'] != df['pred_label']].copy()
    sort_col = 'confidence' if 'confidence' in wrong.columns else None
    if sort_col:
        wrong = wrong.sort_values(sort_col, ascending=False)
    wrong.head(top_n).to_csv(output_dir / f'{model_name}_top{top_n}_errors.csv', index=False)


def _export_classification_report(df, labels, model_name, output_dir):
    report = classification_report(df['true_label'], df['pred_label'],
                                   labels=labels, output_dict=True)
    pd.DataFrame(report).T.to_csv(output_dir / f'{model_name}_classification_report.csv')


def run_all_qualitative_analyses(experiment_dir, output_dir, top_n: int = 20, verbose: bool = True):
    """Run qualitative analyses for *all* models in an experiment.

    Parameters
    ----------
    experiment_dir : str or Path
        Folder that contains a `predictions/` sub‑dir with CSVs.
    output_dir : str or Path
        Where artefacts (plots, CSVs) will be saved.
    top_n : int, optional
        Number of top‑confidence errors to store.
    verbose : bool, optional
        Log progress to stdout.
    """
    experiment_dir = Path(experiment_dir)
    pred_dir = experiment_dir / 'predictions'
    if not pred_dir.exists():
        raise FileNotFoundError(f'{pred_dir} not found')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = glob.glob(str(pred_dir / '*_preds.csv'))
    if not pred_files:
        raise FileNotFoundError(f'No CSVs like <model>_preds.csv found in {pred_dir}')

    labels_set = None
    for csv_path in pred_files:
        model_name = Path(csv_path).stem.replace('_preds', '')
        if verbose:
            print(f'▶ {model_name}')
        df = pd.read_csv(csv_path)

        if labels_set is None:
            labels_set = sorted(df['true_label'].unique())

        # 1) top errors
        _export_top_errors(df, model_name, output_dir, top_n)
        # 2) confusion matrix
        cm = confusion_matrix(df['true_label'], df['pred_label'], labels=labels_set)
        _save_confusion_matrix(cm, labels_set,
                               f'Confusion matrix – {model_name}',
                               output_dir / f'{model_name}_confusion_matrix.png')
        # 3) per‑class metrics
        _export_classification_report(df, labels_set, model_name, output_dir)

    if verbose:
        print('✔︎ Qualitative artefacts saved to', output_dir.resolve())
