"""
evaluation.py
─────────────
Central place for training every model, computing metrics, and persisting
per‑model artefacts (CSV classification reports + PNG confusion matrices).

Usage (inside the notebook)
───────────────────────────
from evaluation import run_evaluations

results = run_evaluations(
    models=models,                # dict[str, TextClassifier]
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    label_names=label_names       # list[str]
)
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# ──────────────────────────────────────────────────────────
# Utility: nice filenames
# ──────────────────────────────────────────────────────────
def _snug(name: str) -> str:
    """Make a filesystem‑friendly string."""
    return name.replace(" ", "_").replace(" ", "_")


# ──────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────
def run_evaluations(
    models: Dict[str, Any],
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    label_names: List[str],
    results_dir: str | Path = "results",
) -> tuple[Dict[str, Dict[str, Any]], dict | None]:
    """
    Train each model, compute metrics, save artefacts, and return a
    tuple: (results, significance_test), where results is a dictionary with all raw results (preds, reports, etc.),
    and significance_test is a dict with the statistical test result between the two best models (or None if not run).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    results: Dict[str, Dict[str, Any]] = {}
    significance_test = None

    for name, model in models.items():
        print(f"▶ Training\xa0{ name } …", end=" ")

        # ── Fit & predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ── Metrics
        report_dict = classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True
        )
        f1_macro = f1_score(y_test, y_pred, average="macro")
        acc       = accuracy_score(y_test, y_pred)
        cmatrix   = confusion_matrix(y_test, y_pred)

        # ── Persist artefacts
        _persist_report(report_dict, name, results_dir)
        _persist_confusion(cmatrix, name, label_names, results_dir)

        # ── Store in‑memory summary
        results[name] = {
            "f1_macro": f1_macro,
            "accuracy": acc,
            "report":   report_dict,
            "confusion_matrix": cmatrix,
            "y_pred":   y_pred,
        }

        print("done  ✓")

    print(f"\nAll files saved under\xa0{ results_dir.resolve() }")
    

    # ── Statistical test: best two models
    if len(results) >= 2:
        # sort by macro‑F1 descending
        sorted_names = sorted(results, key=lambda n: results[n]['f1_macro'], reverse=True)
        best, second = sorted_names[:2]
        y_pred_best   = results[best]['y_pred']
        y_pred_second = results[second]['y_pred']
        p_val, _diffs = _bootstrap_f1(y_test, y_pred_best, y_pred_second)
        

        significance_test = {
            'best': best,
            'second': second,
            'p_value': float(p_val)
        }
        # Persist to JSON so notebooks can load without re‑training
        import json
        with open(results_dir / "significance_best_vs_second.json", "w") as _fp:
            json.dump(significance_test, _fp, indent=2)
        print(f"Significance test: {best} vs {second} → p\xa0=\xa0{p_val:.4f}")

    # ── Persist ROC curves & error lists
    for name, model in models.items():
        try:
            _persist_roc_curves(model, X_test, y_test, label_names, name, results_dir)
        except Exception as e:
            print(f"⚠ Could not compute ROC for {name}: {e}")

        errors = _collect_worst_errors(model, X_test, y_test, results[name]['y_pred'])
        if errors:
            df_errors = pd.DataFrame(errors, columns=['idx', 'true', 'pred', 'confidence', 'preview'])
            df_errors.to_csv(results_dir / f"{ _snug(name) }_worst_errors.csv", index=False)

    return results, significance_test


# ──────────────────────────────────────────────────────────
# Helpers: persistence
# ──────────────────────────────────────────────────────────
def _persist_report(report: Dict[str, Any], name: str, root: Path) -> None:
    """Save classification_report as CSV."""
    df = pd.DataFrame(report).transpose()
    csv_path = root / f"{ _snug(name) }_report.csv"
    df.to_csv(csv_path, index=True)


def _persist_confusion(
    cm: np.ndarray,
    name: str,
    labels: List[str],
    root: Path
) -> None:
    """Save confusion‑matrix figure as PNG (6×5 in)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion\xa0–\xa0{name}")
    fig.tight_layout()

    png_path = root / f"{ _snug(name) }_confusion.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────
# Extra: Statistical significance & ROC curves
# ──────────────────────────────────────────────────────────
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

def _bootstrap_f1(y_true, y_pred_a, y_pred_b, n_rounds: int = 1000, seed: int = 42):
    """Return p‑value that model A beats model B in macro‑F1 via bootstrap."""
    rng = np.random.RandomState(seed)
    diffs = []
    idx = np.arange(len(y_true))
    for _ in range(n_rounds):
        sample = rng.choice(idx, size=len(idx), replace=True)
        f1_a = f1_score(np.array(y_true)[sample], np.array(y_pred_a)[sample], average='macro')
        f1_b = f1_score(np.array(y_true)[sample], np.array(y_pred_b)[sample], average='macro')
        diffs.append(f1_a - f1_b)
    diffs = np.array(diffs)
    p_val = (diffs <= 0).mean()   # one‑sided: A <= B
    return p_val, diffs


def _persist_roc_curves(model, X_test, y_test, label_names, name, root):
    """Save one‑vs‑rest ROC curves for each label (if supported).

    The function aligns classifier score columns with *label_names* so that
    mismatches between model.classes_ order and our canonical order never
    break the plot.
    """
    import numpy as np
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    # Obtain score matrix
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
        cls_order = getattr(model, "classes_", None)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # decision_function may return shape (n_samples,) for binary
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        cls_order = getattr(model, "classes_", None)
    else:
        return  # Not supported
    # If class order differs, reorder columns
    if cls_order is not None and list(cls_order) != list(label_names):
        order_map = {lab: i for i, lab in enumerate(cls_order)}
        scores = np.column_stack([scores[:, order_map[lab]] for lab in label_names])
    # Binarize y
    y_bin = label_binarize(y_test, classes=label_names)
    if scores.shape[1] != y_bin.shape[1]:
        return  # Skip if still inconsistent
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, lab in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.2, label=f"{lab} (AUC\xa0{roc_auc:.2f})")
    ax.plot([0,1],[0,1],'--',lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves\xa0–\xa0{name}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(root / f"{_snug(name)}_roc.png", dpi=150)
    plt.close(fig)


def _collect_worst_errors(model, texts, y_true, y_pred, n: int = 10):
    """Return top-*n* mis‑classified samples ranked by **confidence**.

    * For classifiers with ``predict_proba`` we take the highest predicted
      probability *p̂*; the higher it is, the more confident the model was.
    * For classifiers exposing ``decision_function`` we take the margin:
      the predicted‑class score minus the second‑best score.
    The function returns the *n* mis‑classified items where the model was
    **most confident but still wrong** – those are the most instructive
    errors to inspect.

    It yields a list of tuples *(idx, true_lbl, pred_lbl, confidence, preview)*.
    """
    confidences = []
    # Try to get probability matrix or decision scores – we recompute only
    # for mis‑classified indices, so this is fast enough.
    proba_mode = hasattr(model, 'predict_proba')
    dec_mode   = hasattr(model, 'decision_function')
    if not proba_mode and not dec_mode:
        # Fallback: rank by text length (arbitrary) if no confidence available
        for idx, (t, y_t, y_p) in enumerate(zip(texts, y_true, y_pred)):
            if y_t != y_p:
                confidences.append((idx, y_t, y_p, 0.0))
    else:
        if proba_mode:
            probs = model.predict_proba(texts)
            for idx, (y_t, y_p, row) in enumerate(zip(y_true, y_pred, probs)):
                if y_t != y_p:
                    conf = float(row.max())
                    confidences.append((idx, y_t, y_p, conf))
        else:
            scores = model.decision_function(texts)
            # Binary case → shape (n_samples,), convert to 2D
            import numpy as np
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            for idx, (y_t, y_p, row) in enumerate(zip(y_true, y_pred, scores)):
                if y_t != y_p:
                    # confidence = margin predicted vs next best
                    top2 = np.sort(row)[-2:]
                    conf = float(top2[1] - top2[0])
                    confidences.append((idx, y_t, y_p, conf))
    # Sort by confidence descending (most confident wrong predictions first)
    confidences.sort(key=lambda tup: tup[3], reverse=True)
    previews = []
    for idx, y_t, y_p, conf in confidences[:n]:
        preview = ' '.join(texts[idx].split()[:40])
        previews.append((idx, y_t, y_p, round(conf, 4), preview))
    return previews