"""
CLINC150 Intent Classifier & Semantic Search Demo

This script launches a Gradio-based web application for interactive intent classification
and semantic search using pre-trained models from the CLINC150 Intent Classifier project.

Key Features:
- Select from multiple classification and RAG (Retrieval-Augmented Generation) models
- Paste or enter user utterances for instant intent classification or semantic search
- Robust model loading supporting various serialization formats
  (joblib, pickle, scikit-learn pipelines, or dicts)
- Automatic handling of both traditional ML and RAG-based models,
  including FAISS index and passage retrieval
- Clear error handling and informative feedback for missing or incompatible models

Usage:
- Run the script: `python scripts/demos/intent_classifier_demo.py`
- Access the Gradio web interface to interact with the models

This script is designed for production and demonstration purposes, providing a
user-friendly interface for exploring intent classification and semantic search capabilities.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import gradio as gr
import joblib
import numpy as np
import yaml

from intent_classifier.utils.model_utils import find_model_file
from intent_classifier.utils.paths import get_config_path

# --------------------------------------------------------------------------- #
# 0. Project root & imports                                                   #
# --------------------------------------------------------------------------- #
# Package is now properly installed, no path hacks needed
project_root = Path(__file__).resolve().parent.parent.parent

# Load config to get default paths (respect CONFIG_FILE environment variable)
# Use centralized path resolution so both "config/..." and absolute paths work.
config_file = os.environ.get("CONFIG_FILE", "config/dataset/clinc150/tiny.yaml")
config_path = get_config_path(config_file)
with open(config_path) as f:
    config = yaml.safe_load(f)


def substitute_vars(value: Any, cfg: dict[str, Any]) -> Any:
    """Replace variable references in string values with their actual values from config."""
    if isinstance(value, str) and "${" in value:
        import re

        var_pattern = r"\${([^}]+)}"
        for var_path in re.findall(var_pattern, value):
            if "." in var_path:
                section, var = var_path.split(".", 1)
                if section in cfg and var in cfg[section]:
                    value = value.replace(f"${{{var_path}}}", str(cfg[section][var]))
    return value


# Resolve paths from config
run_name = config["general"]["run_name"]
models_path = substitute_vars(config["paths"]["models_dir"], config)
embeddings_path = substitute_vars(config["paths"]["embeddings_dir"], config)

# Convert to absolute paths, but allow environment overrides when launching the demo
_default_models_path = str(project_root / models_path)
_default_embeddings_path = str(project_root / embeddings_path)

DEFAULT_MODELS_PATH = os.environ.get("MODELS_PATH", _default_models_path)
DEFAULT_EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", _default_embeddings_path)

# --------------------------------------------------------------------------- #
# 1. Model‑info table (kept in sync with config/algorithm/models_config.yaml) #
# --------------------------------------------------------------------------- #
MODELS_INFO = {
    # Text classification models
    "naive_bayes": {
        "name": "Naive Bayes",
        "description": (
            "Multinomial Naive Bayes classifier using TF-IDF features. "
            "Fast and efficient for intent classification."
        ),
        "dir": "Naive Bayes",
        "type": "classifier",
    },
    "linear_svm": {
        "name": "Linear SVM",
        "description": (
            "Support Vector Machine with linear kernel. Good balance of accuracy and speed."
        ),
        "dir": "Linear SVM",
        "type": "classifier",
    },
    "linear_svm_bigrams": {
        "name": "TF-IDF bigrams + SVM",
        "description": (
            "Linear SVM trained on TF-IDF bigram features. "
            "Stronger performance for nuanced intent expressions."
        ),
        "dir": "TF-IDF bigrams + SVM",
        "type": "classifier",
    },
    "transformer_logreg": {
        "name": "MiniLM + LogReg",
        "description": (
            "MiniLM transformer embeddings with Logistic Regression. "
            "High-accuracy semantic intent classification."
        ),
        "dir": "MiniLM + LogReg",
        "type": "classifier",
    },
    "embedding_logreg": {
        "name": "Embedding + LogReg",
        "description": (
            "Generic embedding backend (SBERT/OpenAI) with Logistic Regression. "
            "Use for flexible, backend-agnostic intent classification."
        ),
        "dir": "Embedding + LogReg",
        "type": "classifier",
    },
    # RAG-based models
    "rag_kmajority": {
        "name": "RAG-kMajority",
        "description": "RAG model with k-majority voting to determine the most relevant intents.",
        "dir": "RAG-kMajority",
        "type": "rag",
    },
    "rag_centroid": {
        "name": "RAG-CentroidNN",
        "description": (
            "Retrieval Augmented Generation using centroid-based nearest neighbors search."
        ),
        "dir": "RAG-CentroidNN",
        "type": "rag",
    },
    "rag_llm_local": {
        "name": "RAG-LLM (TF-IDF, default prompt)",
        "description": "RAG-LLM using TF-IDF retrieval and the default LLM prompt.",
        "dir": "RAG-LLM (TF-IDF, default prompt)",
        "type": "rag",
    },
    "rag_llm_local_short": {
        "name": "RAG-LLM (TF-IDF, short prompt)",
        "description": "RAG-LLM using TF-IDF retrieval and a concise LLM prompt.",
        "dir": "RAG-LLM (TF-IDF, short prompt)",
        "type": "rag",
    },
    "rag_llm_local_n8n": {
        "name": "RAG-LLM (TF-IDF, n8n prompt)",
        "description": "RAG-LLM using TF-IDF retrieval and the n8n-style email prompt.",
        "dir": "RAG-LLM (TF-IDF, n8n prompt)",
        "type": "rag",
    },
    "rag_llm_openai": {
        "name": "RAG-LLM (OpenAI-embeddings)",
        "description": ("RAG model using OpenAI embeddings and an OpenAI LLM for classification."),
        "dir": "RAG-LLM (OpenAI-embeddings)",
        "type": "rag",
    },
    "rag_llm_sbert_embeddings": {
        "name": "RAG-LLM (SBERT embeddings, default prompt)",
        "description": "RAG-LLM using SBERT FAISS embeddings and the default LLM prompt.",
        "dir": "RAG-LLM (SBERT embeddings, default prompt)",
        "type": "rag",
    },
    "rag_llm_ollama_embeddings": {
        "name": "RAG-LLM (Qwen embeddings, default prompt)",
        "description": "RAG-LLM using Qwen/Ollama FAISS embeddings and the default LLM prompt.",
        "dir": "RAG-LLM (Qwen embeddings, default prompt)",
        "type": "rag",
    },
}

# --------------------------------------------------------------------------- #
# 2. Helpers                                                                   #
# --------------------------------------------------------------------------- #


class DictModelWrapper:
    """Wrap a ``dict`` → ( vectorizer , classifier ) so that it behaves like a
    scikit‑learn estimator / Pipeline. This allows us to transparently support
    training code that persisted ``{'vectorizer': v, 'classifier': clf}``
    instead of an actual ``Pipeline`` object."""

    def __init__(self, obj: dict[str, Any]):
        # Heuristically locate components
        vec = obj.get("vectorizer") or obj.get("tfidf") or obj.get("vect")
        clf = obj.get("classifier") or obj.get("model") or obj.get("clf")

        if vec is None or clf is None:
            raise ValueError(
                "Cannot wrap dictionary model – expected keys like "
                f"`vectorizer` + `classifier`, got: {list(obj.keys())}"
            )
        self._vectorizer = vec
        self._classifier = clf
        # Delegate everything else to the underlying classifier
        self.classes_ = getattr(clf, "classes_", None)

    # --- scikit‑learn‑style API ------------------------------------------- #
    def predict(self, texts: list[str]):
        X = self._vectorizer.transform(texts)
        return self._classifier.predict(X)

    def predict_proba(self, texts: list[str]):
        if hasattr(self._classifier, "predict_proba"):
            X = self._vectorizer.transform(texts)
            return self._classifier.predict_proba(X)
        return None

    # Anything we don't explicitly implement → delegate
    def __getattr__(self, item):
        return getattr(self._classifier, item)


# Simple cache so we don't re‑load models all the time
_CACHE: dict[str, Any] = {}


def scan_models_directory(models_path: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Return a mapping *model_id → info dict* for every model that is actually
    available on disk (accepts both *.joblib* and *.pkl*)."""
    models: dict[str, dict[str, Any]] = {}
    root = Path(models_path).expanduser().resolve()
    if not root.exists():
        print(f"[scan_models_directory] models_path does not exist: {root}")
        return models

    for model_id, meta in MODELS_INFO.items():
        model_dir = root / meta["dir"]
        if not model_dir.exists():
            print(f"[scan_models_directory] Model directory not found: {model_dir}")
            continue

        # Use the find_model_file utility function to locate the model file
        model_file_path = find_model_file(root, meta["dir"])
        if model_file_path:
            print(f"[scan_models_directory] Found model file: {model_file_path}")
            models[model_id] = {**meta, "path": model_file_path}
        else:
            print(f"[scan_models_directory] No model file found in: {model_dir}")

    print(f"[scan_models_directory] Found {len(models)} models")
    return models


# --------------------------------------------------------------------------- #
# 3. Loading logic                                                            #
# --------------------------------------------------------------------------- #


def _wrap_loaded(obj):
    """If *obj* is a ``dict`` containing separate components, wrap it. Otherwise
    return it unchanged."""
    if isinstance(obj, dict):
        try:
            return DictModelWrapper(obj)
        except Exception as err:
            print("[load_model] Could not wrap dictionary model:", err)
            return obj
    return obj


def load_model(model_id: str, models_path: str, embeddings_path: str) -> Any:
    cache_key = f"{Path(models_path).resolve()}::{model_id}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    available = scan_models_directory(models_path)
    if model_id not in available:
        raise FileNotFoundError(f'Model "{model_id}" not found under {models_path}')

    model_info = available[model_id]
    model_path = Path(model_info["path"])

    if model_info["type"] == "rag":
        # --- RAG: load index, passages and *optional* classifier ------------- #
        idx_dir = Path(embeddings_path).expanduser().resolve()
        if "openai" in model_id:
            index_path = idx_dir / "openai" / "index.faiss"
            meta_path = idx_dir / "openai" / "meta.jsonl"
        else:
            index_path = idx_dir / "sbert" / "index.faiss"
            meta_path = idx_dir / "sbert" / "meta.jsonl"

        index = faiss.read_index(str(index_path)) if index_path.exists() else None

        # Load passages from meta.jsonl
        passages = []
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    try:
                        meta = json.loads(line)
                        if "text" in meta:
                            passages.append(meta["text"])
                    except json.JSONDecodeError:
                        continue
            print(f"[load_model] Loaded {len(passages)} passages from {meta_path}")

        # Load classifier
        classifier_obj = None
        if model_path.exists():
            try:
                print(f"[load_model] Loading classifier from {model_path}")
                classifier_obj = joblib.load(model_path)

                # If this is a RagSklearnAdapter with no rag component, initialize it
                if hasattr(classifier_obj, "rag") and classifier_obj.rag is None:
                    print(f"[load_model] Initializing RAG component for {model_id}")
                    from intent_classifier.rag import (
                        load_centroid,
                        load_kmajority,
                        load_llm,
                        set_artifacts_dir,
                    )
                    from intent_classifier.rag.vector_store import VectorStore
                    from intent_classifier.utils.embeddings import EmbeddingGenerator

                    # Set embeddings directory before loading RAG models
                    set_artifacts_dir(embeddings_path, embeddings_path)

                    # Get top_k from config or use default
                    top_k = config.get("model", {}).get("rag_top_k", 25)

                    # Initialize appropriate RAG model based on model_id
                    if "kmajority" in model_id:
                        rag_model = load_kmajority(
                            use_openai="openai" in model_id,
                            top_k=top_k,
                            artifacts_dir=embeddings_path,
                        )
                    elif "centroid" in model_id:
                        rag_model = load_centroid(
                            use_openai="openai" in model_id, artifacts_dir=embeddings_path
                        )
                    else:  # LLM-based RAG
                        if "openai" in model_id:
                            api_key = os.getenv("OPENAI_API_KEY")
                            if not api_key:
                                raise ValueError(
                                    "OPENAI_API_KEY environment variable must be set "
                                    "for OpenAI embeddings"
                                )
                            embedder = EmbeddingGenerator(
                                api_key=api_key, model="text-embedding-3-small", batch_size=50
                            )
                            rag_model = load_llm(
                                top_k=top_k,
                                model=config.get("model", {}).get(
                                    "llm_model", "ollama/llama3.1:8b"
                                ),
                                embedder=embedder,
                                use_openai=True,
                                artifacts_dir=embeddings_path,
                            )
                        else:
                            # For local embeddings, use the same model that was used
                            # to create the index
                            def local_embedder(texts):
                                return VectorStore.embed(
                                    config.get("model", {}).get(
                                        "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"
                                    ),
                                    texts,
                                )

                            rag_model = load_llm(
                                top_k=top_k,
                                model=config.get("model", {}).get(
                                    "llm_model", "ollama/llama3.1:8b"
                                ),
                                embedder=local_embedder,
                                use_openai=False,
                                artifacts_dir=embeddings_path,
                            )

                    # Set the rag component
                    classifier_obj.rag = rag_model
                    classifier_obj.rag_clf = rag_model

                if hasattr(classifier_obj, "classes_"):
                    print(f"[load_model] Classifier has classes: {classifier_obj.classes_}")
            except Exception as e:
                print(f"[load_model] Could not load RAG classifier: {e}")

        rag_bundle = {"index": index, "passages": passages, "model": classifier_obj}
        _CACHE[cache_key] = rag_bundle
        return rag_bundle

    # --- Plain classifier --------------------------------------------------- #
    try:
        obj = joblib.load(model_path)
    except Exception as err_joblib:
        print("[load_model] joblib load failed – falling back to pickle:", err_joblib)
        with model_path.open("rb") as f:
            obj = pickle.load(f)  # type: ignore[arg-type]

    obj = _wrap_loaded(obj)
    _CACHE[cache_key] = obj
    return obj


# --------------------------------------------------------------------------- #
# 4. Prediction handler (used by Gradio)                                      #
# --------------------------------------------------------------------------- #


def _format_top_probas(model, probas: np.ndarray, top: int = 3) -> str:
    """Return a markdown list of top-k intent names with probabilities."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return ""

    # Ensure we always work with a list of strings
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
    classes = [str(c) for c in classes]

    idx = np.argsort(probas)[-top:][::-1]
    lines = []
    for i in idx:
        label = classes[i] if i < len(classes) else f"class_{i}"
        lines.append(f"- {label}: {probas[i]:.2%}")
    return "\n".join(lines)


def predict(
    model_choice: str, text: str, top_k: int, models_path: str, embeddings_path: str
) -> str:
    if not text.strip():
        return "⚠️ Please enter some text first."

    available_models = scan_models_directory(models_path)
    # Allow the user to pass the *display* name coming from the dropdown
    # (i.e. '📊 Naive Bayes') – strip leading emoji, then map back.
    if model_choice.startswith(("📊", "🔍")):
        clean_name = model_choice.lstrip("📊🔍 ").strip()
        for mid, info in available_models.items():
            if info["name"] == clean_name:
                model_choice = mid
                break

    try:
        model = load_model(model_choice, models_path, embeddings_path)
    except Exception as e:
        return f"❌ Error loading model: {e}"

    m_type = MODELS_INFO[model_choice]["type"]
    if m_type == "rag":
        # -------- Retrieval‑Augmented Generation ---------------------------- #
        passages: list[str] = model.get("passages") or []
        clf = model.get("model")

        # Get classification if available
        result_lines: list[str] = []
        try:
            if clf is not None and hasattr(clf, "predict"):
                raw_pred = clf.predict([text])
                label = raw_pred[0] if raw_pred is not None else ""
                # Normalize possible list/array outputs to a single label string
                if isinstance(label, (list, tuple)):
                    label = label[0] if label else ""
                result_lines.append(f"**Predicted intent:** {label}")

                if hasattr(clf, "predict_proba"):
                    probas = clf.predict_proba([text])[0]
                    max_proba = float(np.max(probas))
                    result_lines.append(f"**Confidence:** {max_proba:.2%}")
                    result_lines.append("")
                    # Top‑k labeled intents
                    result_lines.append("**Top predictions:**")
                    result_lines.append(_format_top_probas(clf, probas, top=3))
                    result_lines.append("")
        except Exception as e:  # pragma: no cover - defensive
            print(f"[predict] RAG classifier failed: {e}")
            import traceback

            traceback.print_exc()

        # Get retrieved documents
        retrieved = passages[:top_k] if passages else []
        documents = "\n\n".join(f"• {p[:400]}..." for p in retrieved) or "No documents found."

        # Combine classification and documents
        result_lines.append("**Retrieved Similar Utterances:**")
        result_lines.append("")
        result_lines.append(documents)
        return "\n".join(result_lines)

    # -------------------- Plain classification ----------------------------- #
    try:
        pred = model.predict([text])
        label = pred[0] if pred is not None else "unknown"
        # Normalize potential list/array outputs to a single label string
        if isinstance(label, (list, tuple)):
            label = label[0] if label else "unknown"

        result_lines = [f"**Predicted intent:** {label}"]
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba([text])
            if probas is not None:
                probas = probas[0]
                result_lines.append(f"**Confidence:** {probas.max():.2%}")
                result_lines.append("")
                result_lines.append("**Top predictions:**")
                result_lines.append(_format_top_probas(model, probas, 3))

        return "\n".join(result_lines)
    except Exception as e:
        import textwrap
        import traceback

        print(
            "[predict] Error during classifier prediction:\n",
            textwrap.indent(traceback.format_exc(), "    "),
        )
        return f"❌ Error during prediction: {e}"


# --------------------------------------------------------------------------- #
# 5. Gradio UI                                                               #
# --------------------------------------------------------------------------- #


def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(title="CLINC150 Intent Classifier & Search") as demo:
        gr.Markdown(
            """
# 🎯 CLINC150 Intent Classifier & Search

Use this demo to **classify user utterances into intents** and, for RAG models,
to **retrieve similar training utterances**.

- **Prerequisites**
  - Run the training pipeline first:
    `python scripts/pipeline/run_all.py`
  - This will create models and embeddings used by this demo.
  - The *Models Directory Path* and *Embeddings Directory Path* below should point to that run.
- **How to use**
  1. Pick a model in **Select Model** (classic ML or RAG).
  2. Optionally adjust the number of similar utterances (RAG only).
  3. Paste a user utterance and click **Analyze**.
  4. Read the predicted intent, confidence, and (for RAG) retrieved examples.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Input")
                models_path = gr.Textbox(value=DEFAULT_MODELS_PATH, label="Models Directory Path")

                embeddings_path = gr.Textbox(
                    value=DEFAULT_EMBEDDINGS_PATH, label="Embeddings Directory Path"
                )

                # Model dropdown with icons
                model_dropdown = gr.Dropdown(
                    choices=[
                        (
                            (
                                f"📊 {info['name']}"
                                if info["type"] == "classifier"
                                else f"🔍 {info['name']}"
                            ),
                            mid,
                        )
                        for mid, info in MODELS_INFO.items()
                    ],
                    value="naive_bayes",
                    label="Select Model",
                    interactive=True,
                )

                # Model description area
                with gr.Accordion("Model Description", open=True):
                    # Get initial description for default model
                    default_info = MODELS_INFO["naive_bayes"]
                    default_type = "Classification"  # Since naive_bayes is a classifier
                    available_models = scan_models_directory(DEFAULT_MODELS_PATH)
                    default_path = available_models.get("naive_bayes", {}).get("path", "Not found")

                    initial_description = f"""
                    ## {default_info["name"]}

                    **Type:** {default_type}

                    **Description:**
                    {default_info["description"]}

                    **Model File:** `{os.path.basename(default_path)}`
                    """
                    model_description = gr.Markdown(initial_description)

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Similar Utterances (for RAG models)",
                )

            with gr.Column(scale=4):
                gr.Markdown("### Text Input")
                text_input = gr.Textbox(
                    lines=8,
                    label="Enter User Utterance or Query",
                    placeholder="Paste user utterance or query text here...",
                )

                analyze_btn = gr.Button("Analyze", variant="primary")

                gr.Markdown("### Results")
                output = gr.Markdown()

        # Update description function
        def update_model_description(model_id):
            if not model_id:
                return "Please select a model to see its description."

            if model_id in MODELS_INFO:
                info = MODELS_INFO[model_id]
                model_type = info["type"]
                type_text = (
                    "Retrieval-Augmented Generation" if model_type == "rag" else "Classification"
                )

                # Get the model path from scanned models
                available_models = scan_models_directory(models_path.value)
                model_path = available_models.get(model_id, {}).get("path", "Not found")

                description = f"""
                ## {info["name"]}

                **Type:** {type_text}

                **Description:**
                {info["description"]}

                **Model File:** `{os.path.basename(model_path)}`
                """
                return description
            else:
                return f"Model information not available for: {model_id}"

        # Connect model selection to description update
        model_dropdown.change(
            fn=update_model_description, inputs=model_dropdown, outputs=model_description
        )

        # Analyze button action
        analyze_btn.click(
            fn=predict,
            inputs=[model_dropdown, text_input, top_k, models_path, embeddings_path],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    print("[main] Launching demo…")
    create_demo().launch()
