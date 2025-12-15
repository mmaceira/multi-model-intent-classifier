"""
Semantic Search Demo Application

This script provides a Gradio-based web interface for semantic search over the CLINC150
intent classification dataset using vector embeddings and FAISS for efficient similarity search.

Key Features:
- Search for semantically similar user utterances using configurable embedding models
- Adjustable number of results and similarity thresholds
- Interactive and batch search modes
- Clear error handling and informative feedback

Usage:
- Run the script: `python scripts/demos/semantic_search_demo.py`
- Access the Gradio web interface to perform semantic searches

This script is suitable for production and demonstration, enabling users to explore
semantic search capabilities interactively.
"""

import json

# Package is now properly installed, no path hacks needed
import os
from pathlib import Path

import faiss
import gradio as gr
import yaml
from sentence_transformers import SentenceTransformer

from intent_classifier.utils.paths import get_config_path

project_root = Path(__file__).resolve().parent.parent.parent

# Load config to get default paths (respect CONFIG_FILE environment variable)
# Use centralized path resolution so both "config/..." and absolute paths work.
config_file = os.environ.get("CONFIG_FILE", "config/dataset/clinc150/tiny.yaml")
config_path = get_config_path(config_file)
with open(config_path) as f:
    config = yaml.safe_load(f)


def substitute_vars(value: str, cfg: dict) -> str:
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
embeddings_path_str = substitute_vars(config["paths"]["embeddings_dir"], config)
embeddings_path = project_root / embeddings_path_str

# Allow overriding the embeddings root when launching the demo
_default_embeddings_root = str(embeddings_path)
EMBEDDINGS_ROOT = os.environ.get("EMBEDDINGS_PATH", _default_embeddings_root)

# Default paths
DEFAULT_INDEX_PATH = str(Path(EMBEDDINGS_ROOT) / "sbert" / "index.faiss")
DEFAULT_META_PATH = str(Path(EMBEDDINGS_ROOT) / "sbert" / "meta.jsonl")
DEFAULT_MODEL_NAME = config.get("model", {}).get(
    "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"
)


def load_index_and_meta(index_path: str, meta_path: str):
    """Load FAISS index and metadata."""
    try:
        index = faiss.read_index(index_path)
        meta = []
        with open(meta_path) as f:
            for line in f:
                meta.append(json.loads(line))
        return index, meta
    except Exception as e:
        raise Exception(f"Error loading index or metadata: {e}") from e


def search_similar_documents(
    query: str, k: int = 5, index_path: str = DEFAULT_INDEX_PATH, meta_path: str = DEFAULT_META_PATH
):
    """Search for similar documents using semantic search."""
    try:
        # Load model, index and metadata
        model = SentenceTransformer(DEFAULT_MODEL_NAME)
        index, meta = load_index_and_meta(index_path, meta_path)

        # Encode query
        q_emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # Search
        distances, indices = index.search(q_emb, k)

        # Format results
        results = []
        for score, idx in zip(distances[0], indices[0], strict=False):
            results.append(
                {
                    "text": meta[idx]["text"],
                    "label": meta[idx].get("label", "unknown"),
                    "score": float(score),
                }
            )

        # Format output for display
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"### Result {i} (Score: {result['score']:.4f})")
            output.append(f"**Intent:** {result['label']}")
            output.append(f"**Utterance:** {result['text'][:500]}...")
            output.append("---")

        return "\n\n".join(output)
    except Exception as e:
        return f"Error during search: {e!s}"


def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(title="Semantic Search Demo") as demo:
        gr.Markdown(
            """
# 🔍 Semantic Search Demo

Use this demo to **search for semantically similar user utterances** in the CLINC150
dataset using SBERT embeddings and a FAISS index.

- **Prerequisites**
  - Run the embeddings step of the pipeline (`python scripts/pipeline/02_build_embeddings.py`).
  - Make sure the FAISS index and metadata paths below point to that run.
- **How to use**
  1. Verify or adjust the FAISS index and metadata paths.
  2. Choose how many results to return with **Number of Results**.
  3. Enter a search query and click **Search**.
  4. Inspect the returned intents, scores, and utterances.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Configuration")
                index_path = gr.Textbox(value=DEFAULT_INDEX_PATH, label="FAISS Index Path")
                meta_path = gr.Textbox(value=DEFAULT_META_PATH, label="Metadata JSONL Path")
                top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")

            with gr.Column(scale=4):
                gr.Markdown("### Search")
                query = gr.Textbox(
                    lines=3, label="Enter your search query", placeholder="Type your query here..."
                )
                search_btn = gr.Button("Search", variant="primary")

                gr.Markdown("### Results")
                output = gr.Markdown()

        # Search button action
        search_btn.click(
            fn=search_similar_documents,
            inputs=[query, top_k, index_path, meta_path],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    print("[main] Launching semantic search demo...")
    create_demo().launch()
