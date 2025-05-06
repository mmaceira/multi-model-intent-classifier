"""scripts/gradio_demo.py
Gradio app: choose a model, paste a query or article, get classification or search results.

**IMPORTANT**  
This version adds robust loading logic that copes with different model‑serialization
formats (``joblib`` *or* ``pickle``; dictionaries with separate vectorizer/classifier;
pipelines; plain estimators) and prevents the infamous *'NoneType' object is not
subscriptable / has no attribute'* errors that you were seeing with the Naive Bayes,
MiniLM + LogReg and RAG‑based models.

Only this **single** file needs to be replaced – drop it in
``reuters‑rag‑classifier‑clean/scripts`` and restart the Gradio demo.
"""

from __future__ import annotations

import os
import sys
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import faiss
import gradio as gr

# Default directory paths
DEFAULT_MODELS_PATH = "output/experiment_with_03_classes/models"
DEFAULT_EMBEDDINGS_PATH = "output/experiment_with_03_classes/embeddings"

# --------------------------------------------------------------------------- #
# 0. Project root & imports                                                   #
# --------------------------------------------------------------------------- #
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --------------------------------------------------------------------------- #
# 1. Model‑info table (UNCHANGED, but shortened here for clarity)             #
# --------------------------------------------------------------------------- #
MODELS_INFO = {
    'naive_bayes': {
        'name': 'Naive Bayes',
        'description': 'Multinomial Naive Bayes classifier using TF-IDF features. Fast and efficient for text classification.',
        'dir': 'Naive Bayes',
        'type': 'classifier'
    },
    'linear_svm': {
        'name': 'Linear SVM',
        'description': 'Support Vector Machine with linear kernel. Good balance of accuracy and speed.',
        'dir': 'Linear SVM',
        'type': 'classifier'
    },
    'tfidf_svm': {
        'name': 'TF-IDF + SVM',
        'description': 'SVM classifier using TF-IDF bigram features. Strong performance on news category prediction.',
        'dir': 'TF-IDF bigrams + SVM',
        'type': 'classifier'
    },
    'minilm_logreg': {
        'name': 'MiniLM + LogReg',
        'description': 'Transformer embeddings with Logistic Regression. Leverages semantic understanding from MiniLM.',
        'dir': 'MiniLM + LogReg',
        'type': 'classifier'
    },
    'rag_centroid': {
        'name': 'RAG CentroidNN',
        'description': 'Retrieval Augmented Generation using centroid-based nearest neighbors search.',
        'dir': 'RAG-CentroidNN',
        'type': 'rag'
    },
    'rag_kmajority': {
        'name': 'RAG k-Majority',
        'description': 'RAG model with k-majority voting to determine the most relevant documents.',
        'dir': 'RAG-kMajority',
        'type': 'rag'
    },
    'rag_llm_local': {
        'name': 'RAG LLM (Local)',
        'description': 'RAG model using locally computed embeddings for document retrieval.',
        'dir': 'RAG-LLM (local-embeddings)',
        'type': 'rag'
    },
    'rag_llm_openai': {
        'name': 'RAG LLM (OpenAI)',
        'description': 'RAG model using OpenAI embeddings for improved semantic search capabilities.',
        'dir': 'RAG-LLM (OpenAI-embeddings)',
        'type': 'rag'
    }
}

# --------------------------------------------------------------------------- #
# 2. Helpers                                                                   #
# --------------------------------------------------------------------------- #

ALLOWED_MODEL_FILENAMES = (
    'model.joblib', 'model.pkl',
    'classifier.joblib', 'classifier.pkl',
    'pipeline.joblib', 'pipeline.pkl',
)

class DictModelWrapper:
    """Wrap a ``dict`` → ( vectorizer , classifier ) so that it behaves like a
    scikit‑learn estimator / Pipeline. This allows us to transparently support
    training code that persisted ``{'vectorizer': v, 'classifier': clf}``
    instead of an actual ``Pipeline`` object."""

    def __init__(self, obj: Dict[str, Any]):
        # Heuristically locate components
        vec = obj.get('vectorizer') or obj.get('tfidf') or obj.get('vect')
        clf = obj.get('classifier') or obj.get('model') or obj.get('clf')

        if vec is None or clf is None:
            raise ValueError(
                'Cannot wrap dictionary model – expected keys like '
                '`vectorizer` + `classifier`, got: %s' % list(obj.keys())
            )
        self._vectorizer = vec
        self._classifier = clf
        # Delegate everything else to the underlying classifier
        self.classes_ = getattr(clf, 'classes_', None)

    # --- scikit‑learn‑style API ------------------------------------------- #
    def predict(self, texts: List[str]):
        X = self._vectorizer.transform(texts)
        return self._classifier.predict(X)

    def predict_proba(self, texts: List[str]):
        if hasattr(self._classifier, 'predict_proba'):
            X = self._vectorizer.transform(texts)
            return self._classifier.predict_proba(X)
        return None

    # Anything we don't explicitly implement → delegate
    def __getattr__(self, item):
        return getattr(self._classifier, item)

def _resolve_model_file(model_dir: Path) -> Optional[Path]:
    """Return the first existing model file in *model_dir*."""
    for fname in ALLOWED_MODEL_FILENAMES:
        f = model_dir / fname
        if f.exists():
            return f
    return None

# Simple cache so we don't re‑load models all the time
_CACHE: Dict[str, Any] = {}

def scan_models_directory(models_path: str | os.PathLike) -> Dict[str, Dict[str, Any]]:
    """Return a mapping *model_id → info dict* for every model that is actually
    available on disk (accepts both *.joblib* and *.pkl*)."""
    models: Dict[str, Dict[str, Any]] = {}
    root = Path(models_path).expanduser().resolve()
    if not root.exists():
        print(f'[scan_models_directory] models_path does not exist: {root}')
        return models

    for model_id, meta in MODELS_INFO.items():
        model_dir = root / meta['dir']
        if not model_dir.exists():
            print(f'[scan_models_directory] Model directory not found: {model_dir}')
            continue

        # Try different model file extensions
        for ext in ('model.joblib', 'model.pkl', 'classifier.joblib', 'classifier.pkl'):
            model_file = model_dir / ext
            if model_file.exists():
                print(f'[scan_models_directory] Found model file: {model_file}')
                models[model_id] = {**meta, 'path': str(model_file)}
                break
        else:
            print(f'[scan_models_directory] No model file found in: {model_dir}')

    print(f'[scan_models_directory] Found {len(models)} models')
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
            print('[load_model] Could not wrap dictionary model:', err)
            return obj
    return obj

def load_model(model_id: str, models_path: str, embeddings_path: str) -> Any:
    cache_key = f'{Path(models_path).resolve()}::{model_id}'
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    available = scan_models_directory(models_path)
    if model_id not in available:
        raise FileNotFoundError(f'Model "{model_id}" not found under {models_path}')

    model_info = available[model_id]
    model_path = Path(model_info['path'])

    if model_info['type'] == 'rag':
        # --- RAG: load index, passages and *optional* classifier ------------- #
        idx_dir = Path(embeddings_path).expanduser().resolve()
        if 'openai' in model_id:
            index_path = idx_dir / 'openai' / 'index.faiss'
            meta_path = idx_dir / 'openai' / 'meta.jsonl'
        else:
            index_path = idx_dir / 'sbert' / 'index.faiss'
            meta_path = idx_dir / 'sbert' / 'meta.jsonl'

        index = faiss.read_index(str(index_path)) if index_path.exists() else None
        
        # Load passages from meta.jsonl
        passages = []
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                for line in f:
                    try:
                        meta = json.loads(line)
                        if 'text' in meta:
                            passages.append(meta['text'])
                    except json.JSONDecodeError:
                        continue
            print(f'[load_model] Loaded {len(passages)} passages from {meta_path}')

        # Load classifier
        classifier_obj = None
        if model_path.exists():
            try:
                print(f'[load_model] Loading classifier from {model_path}')
                classifier_obj = joblib.load(model_path)
                
                # If this is a RagSklearnAdapter with no rag component, initialize it
                if hasattr(classifier_obj, 'rag') and classifier_obj.rag is None:
                    print(f'[load_model] Initializing RAG component for {model_id}')
                    from src.rag import load_kmajority, load_centroid, load_llm
                    from src.embeddings.openai_embedder import OpenAIEmbedder
                    from src.rag.vector_store import VectorStore
                    
                    # Initialize appropriate RAG model based on model_id
                    if 'kmajority' in model_id:
                        rag_model = load_kmajority(top_k=5, use_openai='openai' in model_id)
                    elif 'centroid' in model_id:
                        rag_model = load_centroid(use_openai='openai' in model_id)
                    else:  # LLM-based RAG
                        if 'openai' in model_id:
                            embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)
                            rag_model = load_llm(
                                top_k=5,
                                model="gpt-4o-mini",
                                embedder=embedder,
                                use_openai=True
                            )
                        else:
                            # For local embeddings, use the same model that was used to create the index
                            embedder = lambda texts: VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", texts)
                            rag_model = load_llm(
                                top_k=5,
                                model="gpt-4o-mini",
                                embedder=embedder,
                                use_openai=False
                            )
                    
                    # Set the rag component
                    classifier_obj.rag = rag_model
                    classifier_obj.rag_clf = rag_model
                
                if hasattr(classifier_obj, 'classes_'):
                    print(f'[load_model] Classifier has classes: {classifier_obj.classes_}')
            except Exception as e:
                print(f'[load_model] Could not load RAG classifier: {e}')

        rag_bundle = {'index': index, 'passages': passages, 'model': classifier_obj}
        _CACHE[cache_key] = rag_bundle
        return rag_bundle

    # --- Plain classifier --------------------------------------------------- #
    try:
        obj = joblib.load(model_path)
    except Exception as err_joblib:
        print('[load_model] joblib load failed – falling back to pickle:', err_joblib)
        with model_path.open('rb') as f:
            obj = pickle.load(f)

    obj = _wrap_loaded(obj)
    _CACHE[cache_key] = obj
    return obj

# --------------------------------------------------------------------------- #
# 4. Prediction handler (used by Gradio)                                      #
# --------------------------------------------------------------------------- #

def _format_top_probas(model, probas: np.ndarray, top: int = 3) -> str:
    classes = getattr(model, 'classes_', None)
    if classes is None:
        return ''
    idx = np.argsort(probas)[-top:][::-1]
    return '\n'.join(f'- {classes[i]}: {probas[i]:.2%}' for i in idx)

def predict(model_choice: str, text: str, top_k: int,
            models_path: str, embeddings_path: str) -> str:
    if not text.strip():
        return '⚠️ Please enter some text first.'

    available_models = scan_models_directory(models_path)
    # Allow the user to pass the *display* name coming from the dropdown
    # (i.e. '📊 Naive Bayes') – strip leading emoji, then map back.
    if model_choice.startswith(('📊', '🔍')):
        clean_name = model_choice.lstrip('📊🔍 ').strip()
        for mid, info in available_models.items():
            if info['name'] == clean_name:
                model_choice = mid
                break

    try:
        model = load_model(model_choice, models_path, embeddings_path)
    except Exception as e:
        return f'❌ Error loading model: {e}'

    m_type = MODELS_INFO[model_choice]['type']
    if m_type == 'rag':
        # -------- Retrieval‑Augmented Generation ---------------------------- #
        index     : faiss.Index = model['index']
        passages  : List[str]   = model['passages']
        clf       = model['model']  # optional

        # Get classification if available
        classification = ""
        if clf is not None:
            try:
                if hasattr(clf, 'predict'):
                    label = clf.predict([text])[0]
                    classification = f'**Predicted category:** {label}\n\n'
                    
                    if hasattr(clf, 'predict_proba'):
                        probas = clf.predict_proba([text])[0]
                        max_proba = max(probas)
                        classification += f'**Confidence:** {max_proba:.2%}\n\n'
                        
                        if hasattr(clf, 'classes_'):
                            classes = clf.classes_
                            top_indices = probas.argsort()[-3:][::-1]
                            
                            classification += '**Top predictions:**\n'
                            for idx in top_indices:
                                classification += f'- {classes[idx]}: {probas[idx]:.2%}\n'
                            classification += '\n'
            except Exception as e:
                print(f'[predict] RAG classifier failed: {e}')
                import traceback
                traceback.print_exc()

        # Get retrieved documents
        retrieved = passages[:top_k] if passages else []
        documents = '\n\n'.join(f'• {p[:400]}...' for p in retrieved) or 'No documents found.'

        # Combine classification and documents
        answer = f"{classification}**Retrieved Documents:**\n\n{documents}"
        return answer

    # -------------------- Plain classification ----------------------------- #
    try:
        pred = model.predict([text])
        label = pred[0] if pred is not None else 'unknown'

        result_lines = [f'**Predicted category:** {label}']
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba([text])
            if probas is not None:
                probas = probas[0]
                result_lines.append(f'**Confidence:** {probas.max():.2%}')
                result_lines.append('')
                result_lines.append(_format_top_probas(model, probas, 3))

        return '\n'.join(result_lines)
    except Exception as e:
        import traceback, textwrap
        print('[predict] Error during classifier prediction:\n',
              textwrap.indent(traceback.format_exc(), '    '))
        return f'❌ Error during prediction: {e}'

# --------------------------------------------------------------------------- #
# 5. Gradio UI                                                               #
# --------------------------------------------------------------------------- #

def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(title="Reuters News Classifier & Search") as demo:
        gr.Markdown("# 📰 Reuters News Classifier & Search")
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Input")
                models_path = gr.Textbox(
                    value=DEFAULT_MODELS_PATH,
                    label="Models Directory Path"
                )
                
                embeddings_path = gr.Textbox(
                    value=DEFAULT_EMBEDDINGS_PATH,
                    label="Embeddings Directory Path"
                )
                
                # Model dropdown with icons
                model_dropdown = gr.Dropdown(
                    choices=[(f'📊 {info["name"]}' if info['type']=="classifier" else f'🔍 {info["name"]}', mid)
                             for mid, info in MODELS_INFO.items()],
                    value='naive_bayes',
                    label="Select Model",
                    interactive=True,
                )
                
                # Model description area
                with gr.Accordion("Model Description", open=True) as model_info_accordion:
                    # Get initial description for default model
                    default_info = MODELS_INFO['naive_bayes']
                    default_type = "Classification"  # Since naive_bayes is a classifier
                    available_models = scan_models_directory(DEFAULT_MODELS_PATH)
                    default_path = available_models.get('naive_bayes', {}).get('path', 'Not found')
                    
                    initial_description = f"""
                    ## {default_info['name']}
                    
                    **Type:** {default_type}
                    
                    **Description:**  
                    {default_info['description']}
                    
                    **Model File:** `{os.path.basename(default_path)}`
                    """
                    model_description = gr.Markdown(initial_description)
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Documents (for RAG models)"
                )
                
            with gr.Column(scale=4):
                gr.Markdown("### Text Input")
                text_input = gr.Textbox(
                    lines=8,
                    label="Enter Article Text or Query",
                    placeholder="Paste news article or query text here..."
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
                model_type = info['type']
                type_text = "Retrieval-Augmented Generation" if model_type == "rag" else "Classification" 
                
                # Get the model path from scanned models
                available_models = scan_models_directory(models_path.value)
                model_path = available_models.get(model_id, {}).get('path', 'Not found')
                
                description = f"""
                ## {info['name']}
                
                **Type:** {type_text}
                
                **Description:**  
                {info['description']}
                
                **Model File:** `{os.path.basename(model_path)}`
                """
                return description
            else:
                return f"Model information not available for: {model_id}"
        
        # Connect model selection to description update
        model_dropdown.change(
            fn=update_model_description,
            inputs=model_dropdown,
            outputs=model_description
        )
        
        # Analyze button action
        analyze_btn.click(
            fn=predict,
            inputs=[model_dropdown, text_input, top_k, models_path, embeddings_path],
            outputs=output
        )
    
    return demo

if __name__ == '__main__':
    print('[main] Launching demo…')
    create_demo().launch()
