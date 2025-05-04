\"\"\"Model loader with automatic vectorizer fallback.\"\"\"
import os
from functools import lru_cache
from pathlib import Path
import joblib

try:
    from sklearn.pipeline import Pipeline, make_pipeline
except ImportError:
    Pipeline = tuple
    def make_pipeline(*steps):  # type: ignore
        raise RuntimeError('scikit‑learn required to build pipelines")

MODELS_DIR = Path(os.getenv('MODELS_DIR',
                   Path(__file__).resolve().parent.parent.parent / 'models'))

def _locate(model_name: str) -> Path:
    direct = MODELS_DIR / f'{model_name}.joblib'
    if direct.exists():
        return direct
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and d.name.lower() == model_name.lower():
            cand = d / 'model.joblib'
            if cand.exists():
                return cand
    raise FileNotFoundError(f'Model {model_name!r} not found in {MODELS_DIR}')

@lru_cache(maxsize=32)
def get_model(model_name: str):
    path = _locate(model_name)
    artefact = joblib.load(path)

    vec_path = path.with_name('vectorizer.joblib')
    if vec_path.exists():
        try:
            is_pipe = isinstance(artefact, Pipeline)
        except Exception:
            is_pipe = False
        if not is_pipe:
            vectorizer = joblib.load(vec_path)
            artefact = make_pipeline(vectorizer, artefact)

    if not hasattr(artefact, 'predict'):
        raise AttributeError(f'Loaded artefact for {model_name!r} lacks .predict')
    return artefact
