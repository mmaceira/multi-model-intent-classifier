"""Model loader with automatic vectorizer fallback for FastAPI & Gradio."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import faiss
import joblib

try:
    # Only fails in very slim inference images
    from sklearn.pipeline import Pipeline, make_pipeline
except ImportError:  # pragma: no cover
    Pipeline = tuple  # type: ignore

    def make_pipeline(*steps):  # type: ignore
        raise RuntimeError("scikit‑learn required to build pipelines")


# Where to look for persisted models
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent.parent / "models"))

# Where to look for embeddings
EMBEDDINGS_DIR = Path(
    os.getenv("EMBEDDINGS_DIR", Path(__file__).resolve().parent.parent.parent / "embeddings")
)

# Model info mapping
MODELS_INFO = {
    "naive_bayes": {"name": "Naive Bayes", "dir": "Naive Bayes", "type": "classifier"},
    "linear_svm": {"name": "Linear SVM", "dir": "Linear SVM", "type": "classifier"},
    "tfidf_svm": {"name": "TF-IDF + SVM", "dir": "TF-IDF bigrams + SVM", "type": "classifier"},
    "minilm_logreg": {"name": "MiniLM + LogReg", "dir": "MiniLM + LogReg", "type": "classifier"},
    "rag_centroid": {"name": "RAG CentroidNN", "dir": "RAG-CentroidNN", "type": "rag"},
    "rag_kmajority": {"name": "RAG k-Majority", "dir": "RAG-kMajority", "type": "rag"},
    "rag_llm_local": {
        "name": "RAG LLM (Local)",
        "dir": "RAG-LLM (local-embeddings)",
        "type": "rag",
    },
    "rag_llm_openai": {
        "name": "RAG LLM (OpenAI)",
        "dir": "RAG-LLM (OpenAI-embeddings)",
        "type": "rag",
    },
}


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _locate(model_identifier: str) -> Path:
    """Return the exact .joblib path for *model_identifier* (file or directory)."""
    # First try direct .joblib files
    direct = MODELS_DIR / f"{model_identifier}.joblib"
    if direct.exists():
        return direct

    # Then try model IDs from MODELS_INFO
    if model_identifier in MODELS_INFO:
        model_dir = MODELS_DIR / MODELS_INFO[model_identifier]["dir"]
        if model_dir.exists():
            cand = model_dir / "model.joblib"
            if cand.exists():
                return cand

    # Finally try directory names
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and d.name.lower() == model_identifier.lower():
            cand = d / "model.joblib"
            if cand.exists():
                return cand

    raise FileNotFoundError(f"Model {model_identifier!r} not found in {MODELS_DIR}")


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #
@lru_cache(maxsize=32)
def get_model(model_identifier: str):
    """Return a *text‑ready* estimator – always accepts raw text."""
    model_path = _locate(model_identifier)

    # Get model info
    model_info = None
    for mid, info in MODELS_INFO.items():
        if mid == model_identifier or info["dir"] == model_path.parent.name:
            model_info = info
            break

    if model_info is None:
        raise ValueError(f"Unknown model type for {model_identifier}")

    if model_info["type"] == "rag":
        # --- RAG: load index, passages and *optional* classifier ------------- #
        # Convention: model identifiers containing "openai" use the OpenAI embedding index
        # Other identifiers (e.g., "rag_llm_local", "rag_centroid") use the SBERT index
        idx_dir = EMBEDDINGS_DIR
        if "openai" in model_identifier:
            index_path = idx_dir / "openai" / "index.faiss"
            meta_path = idx_dir / "openai" / "meta.jsonl"
        else:
            index_path = idx_dir / "sbert" / "index.faiss"
            meta_path = idx_dir / "sbert" / "meta.jsonl"

        index = faiss.read_index(str(index_path)) if index_path.exists() else None

        # Load passages from meta.jsonl
        passages = []
        if meta_path.exists():
            with open(meta_path, "r") as f:
                for line in f:
                    try:
                        meta = json.loads(line)
                        if "text" in meta:
                            passages.append(meta["text"])
                    except json.JSONDecodeError:
                        continue

        # Load classifier
        classifier_obj = None
        if model_path.exists():
            try:
                classifier_obj = joblib.load(model_path)

                # If this is a RagSklearnAdapter with no rag component, initialize it
                if hasattr(classifier_obj, "rag") and classifier_obj.rag is None:
                    from src.embeddings.openai_embedder import OpenAIEmbedder
                    from src.rag import load_centroid, load_hybrid, load_kmajority, load_llm
                    from src.rag.vector_store import VectorStore

                    # Initialize appropriate RAG model based on model_id
                    if "kmajority" in model_identifier:
                        rag_model = load_kmajority(top_k=5, use_openai="openai" in model_identifier)
                    elif "centroid" in model_identifier:
                        rag_model = load_centroid(use_openai="openai" in model_identifier)
                    elif "hybrid" in model_identifier:
                        rag_model = load_hybrid(use_openai="openai" in model_identifier)
                    else:  # LLM-based RAG
                        if "openai" in model_identifier:
                            embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)
                            rag_model = load_llm(
                                top_k=5,
                                model="ollama/llama3.1:8b",
                                embedder=embedder,
                                use_openai=True,
                            )
                        else:
                            # For local embeddings, use the same model that was used to create the index
                            def embedder(texts):
                                return VectorStore.embed(
                                    "sentence-transformers/all-MiniLM-L6-v2", texts
                                )

                            rag_model = load_llm(
                                top_k=5,
                                model="ollama/llama3.1:8b",
                                embedder=embedder,
                                use_openai=False,
                            )

                    # Set the rag component
                    classifier_obj.rag = rag_model
                    classifier_obj.rag_clf = rag_model
            except Exception as e:
                print(f"Could not load RAG classifier: {e}")

        return {"index": index, "passages": passages, "model": classifier_obj}

    # --- Plain classifier --------------------------------------------------- #
    artefact = joblib.load(model_path)

    # If the artefact is *just* the classifier, add the sibling vectorizer
    needs_wrap = not hasattr(artefact, "predict") or getattr(artefact, "_expects_vectors", False)
    if needs_wrap:
        vec_path = model_path.with_name("vectorizer.joblib")
        if vec_path.exists():
            vectorizer = joblib.load(vec_path)
            artefact = make_pipeline(vectorizer, artefact)
        else:
            raise AttributeError(
                f"Loaded object for '{model_identifier}' cannot classify raw text and "
                f"no vectorizer.joblib found next to it."
            )

    return artefact
