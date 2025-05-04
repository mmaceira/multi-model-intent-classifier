"""Model loader with automatic vectorizer fallback for FastAPI & Gradio."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import joblib

try:
    # Only fails in very slim inference images
    from sklearn.pipeline import Pipeline, make_pipeline
except ImportError:  # pragma: no cover
    Pipeline = tuple  # type: ignore
    def make_pipeline(*steps):  # type: ignore
        raise RuntimeError("scikit‑learn required to build pipelines")

# Where to look for persisted models
MODELS_DIR = Path(
    os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent.parent / "models")
)

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _locate(model_name: str) -> Path:
    """Return the exact .joblib path for *model_name* (file or directory)."""
    direct = MODELS_DIR / f"{model_name}.joblib"
    if direct.exists():
        return direct

    for d in MODELS_DIR.iterdir():
        if d.is_dir() and d.name.lower() == model_name.lower():
            cand = d / "model.joblib"
            if cand.exists():
                return cand

    raise FileNotFoundError(f"Model {model_name!r} not found in {MODELS_DIR}")

# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #
@lru_cache(maxsize=32)
def get_model(model_name: str):
    """Return a *text‑ready* estimator – always accepts raw text."""
    model_path = _locate(model_name)
    artefact = joblib.load(model_path)

    # If the artefact is *just* the classifier, add the sibling vectorizer
    needs_wrap = (
        not hasattr(artefact, "predict") or getattr(artefact, "_expects_vectors", False)
    )
    if needs_wrap:
        vec_path = model_path.with_name("vectorizer.joblib")
        if vec_path.exists():
            vectorizer = joblib.load(vec_path)
            artefact = make_pipeline(vectorizer, artefact)
        else:
            raise AttributeError(
                f"Loaded object for '{model_name}' cannot classify raw text and "
                f"no vectorizer.joblib found next to it."
            )

    return artefact
