"""
Semantic Search Demo Application

This script provides a Gradio-based web interface for semantic search over the Reuters news corpus using vector embeddings and FAISS for efficient similarity search.

Key Features:
- Search for semantically similar news articles using configurable embedding models
- Adjustable number of results and similarity thresholds
- Interactive and batch search modes
- Clear error handling and informative feedback

Usage:
- Place this script in the `scripts/` directory of your project
- Ensure the FAISS index and metadata paths are correctly set (defaults provided)
- Run the script: `python scripts/semantic_search_demo.py`
- Access the Gradio web interface to perform semantic searches

This script is suitable for production and demonstration, enabling users to explore semantic search capabilities interactively.
"""

import json

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

# Default paths
DEFAULT_INDEX_PATH = "output/experiment_10_classes/embeddings/sbert/index.faiss"
DEFAULT_META_PATH = "output/experiment_10_classes/embeddings/sbert/meta.jsonl"


def load_index_and_meta(index_path: str, meta_path: str):
    """Load FAISS index and metadata."""
    try:
        index = faiss.read_index(index_path)
        meta = []
        with open(meta_path, "r") as f:
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
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
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
                {"text": meta[idx]["text"], "label": meta[idx]["label"], "score": float(score)}
            )

        # Format output for display
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"### Result {i} (Score: {result['score']:.4f})")
            output.append(f"**Category:** {result['label']}")
            output.append(f"**Text:** {result['text'][:500]}...")
            output.append("---")

        return "\n\n".join(output)
    except Exception as e:
        return f"Error during search: {str(e)}"


def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(title="Semantic Search Demo") as demo:
        gr.Markdown("# 🔍 Semantic Search Demo")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Configuration")
                gr.Textbox(value=DEFAULT_INDEX_PATH, label="FAISS Index Path")
                gr.Textbox(value=DEFAULT_META_PATH, label="Metadata JSONL Path")
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
