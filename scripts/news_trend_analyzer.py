"""
News Trend Analyzer Application

This script provides a Gradio-based web interface for analyzing news trends and topic evolution using semantic search and large language models (LLMs) on the Reuters news corpus.

Key Features:
- Time-series and trend analysis of news topics
- Retrieval of semantically similar articles using vector embeddings and FAISS
- LLM-powered relevance classification and trend commentary
- Interactive configuration and visualization of results
- Clear error handling and informative feedback

Usage:
- Place this script in the `scripts/` directory of your project
- Ensure the FAISS index, metadata, and OpenAI API key are correctly set (defaults provided)
- Run the script: `python scripts/news_trend_analyzer.py`
- Access the Gradio web interface to analyze news trends interactively

This script is suitable for production and demonstration, enabling users to explore topic evolution and trend analysis in news data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import faiss
import gradio as gr
import numpy as np
from litellm import completion
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)

# Default configuration
DEFAULT_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "index_path": "output/experiment_10_classes/embeddings/sbert/index.faiss",
    "meta_path": "output/experiment_10_classes/embeddings/sbert/meta.jsonl",
    "top_k": 5,
    "max_tokens": 512,
    "relevance_labels": ("high", "medium", "low"),
    "classification_model": "ollama/llama3.1:8b",
    "analysis_model": "ollama/llama3.1:8b",
    "classification_temp": 0.3,
    "analysis_temp": 0.2,
}


class NewsTrendAnalyzer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize required resources (model, index, OpenAI client)."""
        logging.info("Initializing resources...")

        # Load SBERT model
        self.model = SentenceTransformer(self.config["model_name"])
        logging.info(f"Loaded SentenceTransformer: {self.config['model_name']}")

        # Note: Using litellm which supports Ollama and other providers
        # No API key needed for local Ollama models
        logging.info("Using litellm for LLM access (supports Ollama and other providers).")

        # Load FAISS index and metadata
        try:
            self.index = faiss.read_index(self.config["index_path"])
            with open(self.config["meta_path"]) as f:
                self.meta = [json.loads(line) for line in f]
            logging.info("Loaded FAISS index and metadata")
        except Exception as e:
            raise Exception(f"Error loading index or metadata: {e}") from e

    def _embed_and_normalize(self, texts: List[str]) -> np.ndarray:
        """Embed texts and normalize vectors."""
        emb = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    def _classify_relevance(self, query: str, doc: str) -> Tuple[str, str]:
        """Classify relevance of a document to the query using LLM."""
        system = (
            "You are an expert assistant. Label how relevant this previous news article is to understanding a new article. "
            f"Use labels {self.config['relevance_labels']}. Respond in JSON: {{ relevance: label, comment: rationale }}."
        )
        user = f"New Article:\n{query}\n\nPrevious Article:\n{doc}\n"
        resp = completion(
            model=self.config["classification_model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=64,
            temperature=self.config["classification_temp"],
        )
        content = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            return parsed.get("relevance", "low"), parsed.get("comment", "")
        except json.JSONDecodeError:
            return "low", content.replace("\n", " ")

    def analyze_news_trend(self, article: str, k: int = None) -> str:
        """Analyze news trends for a given article."""
        k = k or self.config["top_k"]

        # Retrieve similar docs
        q_emb = self._embed_and_normalize([article])
        scores, idxs = self.index.search(q_emb, k)

        # Process retrieved documents
        retrieved = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0], strict=False), start=1):
            prev_text = self.meta[idx]["text"]
            rel, comment = self._classify_relevance(article, prev_text)
            retrieved.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "text": prev_text,
                    "rel": rel,
                    "comment": comment,
                }
            )

        # Build context from medium/high relevance articles
        context = "\n\n".join(
            f"[Article {r['rank']} | {r['rel']}] {r['text']}"
            for r in retrieved
            if r["rel"] in ("high", "medium")
        )

        # Generate trend analysis
        system = (
            "You are a news analyst. Given a new article and context of past articles, "
            "assess whether the new information is surprising or in line with previous trends, "
            "and what we might expect next."
        )
        prompt = f"New Article:\n{article}\n\nContext of Past Articles:\n{context}\n\nQ: Is this new article surprising compared to past trends? What can we expect next?"
        analysis = completion(
            model=self.config["analysis_model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["analysis_temp"],
        )
        analysis_text = analysis.choices[0].message.content.strip()

        # Build commentary
        commentary = "\n\n".join(
            f"Previous {r['rank']} ({r['rel']}): {r['comment']}" for r in retrieved
        )

        return (
            f"New Article Analysis:\n{analysis_text}\n\n"
            "----\nPrevious Article Commentary:\n"
            f"{commentary}"
        )


def create_demo():
    """Create the Gradio demo interface."""
    analyzer = NewsTrendAnalyzer()

    with gr.Blocks(title="News Trend Analyzer") as demo:
        gr.Markdown("# 📰 News Trend Analyzer")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Configuration")
                gr.Textbox(value=DEFAULT_CONFIG["index_path"], label="FAISS Index Path")
                gr.Textbox(value=DEFAULT_CONFIG["meta_path"], label="Metadata JSONL Path")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=DEFAULT_CONFIG["top_k"],
                    step=1,
                    label="Number of Similar Articles",
                )

            with gr.Column(scale=4):
                gr.Markdown("### Analysis")
                article = gr.Textbox(
                    lines=10,
                    label="Enter News Article",
                    placeholder="Paste your news article here...",
                )
                analyze_btn = gr.Button("Analyze Trends", variant="primary")

                gr.Markdown("### Results")
                output = gr.Markdown()

        # Analyze button action
        analyze_btn.click(fn=analyzer.analyze_news_trend, inputs=[article, top_k], outputs=output)

    return demo


if __name__ == "__main__":
    print("[main] Launching news trend analyzer...")
    create_demo().launch()
