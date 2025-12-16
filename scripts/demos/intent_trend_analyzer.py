"""
Intent Trend Analyzer Application

This script provides a Gradio-based web interface for analyzing intent trends and
evolution using semantic search and large language models (LLMs) on the CLINC150
intent classification dataset.

Key Features:
- Time-series and trend analysis of user intents
- Retrieval of semantically similar utterances using vector embeddings and FAISS
- LLM-powered relevance classification and trend commentary
- Interactive configuration and visualization of results
- Clear error handling and informative feedback

Usage:
- Run the script: `python scripts/demos/intent_trend_analyzer.py`
- Access the Gradio web interface to analyze intent trends interactively

This script is suitable for production and demonstration, enabling users to explore
intent evolution and trend analysis in user utterance data.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import gradio as gr
import numpy as np
import yaml
from litellm import completion
from sentence_transformers import SentenceTransformer

from intent_classifier.utils.paths import get_config_path

# Configure logging
logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)

# Project root (package is now properly installed, no path hacks needed)
project_root = Path(__file__).resolve().parent.parent.parent

# Load config to get default paths (respect CONFIG_FILE environment variable)
# Use centralized path resolution so both "config/..." and absolute paths work.
config_file = os.environ.get("CONFIG_FILE", "config/experiments/clinc150/tiny.yaml")
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
embeddings_path = substitute_vars(config["paths"]["embeddings_dir"], config)
embeddings_path = str(project_root / embeddings_path)

# Allow overriding the embeddings root when launching the demo
EMBEDDINGS_ROOT = os.environ.get("EMBEDDINGS_PATH", embeddings_path)

# Default configuration
DEFAULT_CONFIG = {
    "model_name": config.get("model", {}).get(
        "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"
    ),
    "index_path": str(Path(EMBEDDINGS_ROOT) / "sbert" / "index.faiss"),
    "meta_path": str(Path(EMBEDDINGS_ROOT) / "sbert" / "meta.jsonl"),
    "top_k": 5,
    "max_tokens": 512,
    "relevance_labels": ("high", "medium", "low"),
    "classification_model": config.get("model", {}).get("llm_model", "ollama/llama3.1:8b"),
    "analysis_model": config.get("model", {}).get("llm_model", "ollama/llama3.1:8b"),
    "classification_temp": 0.3,
    "analysis_temp": 0.2,
}


class IntentTrendAnalyzer:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or DEFAULT_CONFIG
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize required resources (model, index, LLM client)."""
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

    def _embed_and_normalize(self, texts: list[str]) -> np.ndarray:
        """Embed texts and normalize vectors."""
        emb = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    def _classify_relevance(self, query: str, doc: str) -> tuple[str, str]:
        """Classify relevance of a document to the query using LLM."""
        system = (
            "You are an expert assistant. Label how relevant this previous user "
            "utterance is to understanding a new utterance.\n"
            f"- Use one of these labels exactly: {self.config['relevance_labels']}.\n"
            "- Respond with ONLY a single JSON object, no prose, no markdown, no code fences.\n"
            '- JSON schema: {"relevance": <label>, "comment": <short rationale>}.'
        )
        user = f"New Utterance:\n{query}\n\nPrevious Utterance:\n{doc}\n"
        resp = completion(
            model=self.config["classification_model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=64,
            temperature=self.config["classification_temp"],
        )
        content = resp.choices[0].message.content.strip()

        # Try to robustly extract JSON even if the model wraps it in prose or code fences
        cleaned = content

        # Strip common code fence wrappers, e.g. ```json ... ```
        if "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        # Extract the first JSON-like object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
        else:
            candidate = cleaned

        try:
            parsed = json.loads(candidate)
            relevance = str(parsed.get("relevance", "low")).lower()
            if relevance not in self.config["relevance_labels"]:
                relevance = "low"
            comment = str(parsed.get("comment", "")).replace("\n", " ")
            return relevance, comment
        except json.JSONDecodeError:
            # Fallback: infer relevance roughly from text; keep raw content as comment
            text_lower = cleaned.lower()
            if "high" in text_lower:
                relevance = "high"
            elif "medium" in text_lower:
                relevance = "medium"
            else:
                relevance = "low"
            return relevance, cleaned.replace("\n", " ")

    def analyze_intent_trend(self, utterance: str, k: int | None = None) -> str:
        """Analyze intent trends for a given utterance."""
        k = k or self.config["top_k"]

        # Retrieve similar docs
        q_emb = self._embed_and_normalize([utterance])
        scores, idxs = self.index.search(q_emb, k)

        # Process retrieved documents
        retrieved = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0], strict=False), start=1):
            prev_text = self.meta[idx]["text"]
            rel, comment = self._classify_relevance(utterance, prev_text)
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
            f"[Utterance {r['rank']} | {r['rel']}] {r['text']}"
            for r in retrieved
            if r["rel"] in ("high", "medium")
        )

        # Generate trend analysis
        system = (
            "You are an intent classification analyst. Given a new user utterance "
            "and context of past utterances, assess whether the new intent is "
            "surprising or in line with previous patterns, and what we might expect next."
        )
        prompt = (
            f"New Utterance:\n{utterance}\n\nContext of Past Utterances:\n{context}\n\n"
            "Q: Is this new utterance surprising compared to past patterns? "
            "What can we expect next?"
        )
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
            f"New Utterance Analysis:\n{analysis_text}\n\n"
            "----\nPrevious Utterance Commentary:\n"
            f"{commentary}"
        )


def create_demo():
    """Create the Gradio demo interface."""
    analyzer = IntentTrendAnalyzer()

    # Re-import gradio to ensure it's available (handles any import issues)
    if not hasattr(gr, "Blocks"):
        raise RuntimeError(
            f"Gradio Blocks not available. Version: {getattr(gr, '__version__', 'unknown')}. "
            f"This may be a Gradio installation issue."
        )

    with gr.Blocks(title="Intent Trend Analyzer") as demo:
        gr.Markdown(
            """
# 🎯 Intent Trend Analyzer

Use this demo to **analyze how a new user utterance fits into past behavior** by
retrieving similar utterances and asking an LLM to comment on relevance and trends.

- **Prerequisites**
  - Run the training pipeline up to embeddings (`python scripts/pipeline/02_build_embeddings.py`).
  - Ensure the FAISS index and metadata paths below point to that experiment run.
  - Have an LLM backend configured in your config file (defaults to `ollama/llama3.1:8b`).
- **How to use**
  1. Verify or adjust the FAISS index and metadata paths.
  2. Optionally change how many similar utterances to retrieve.
  3. Paste a new user utterance and click **Analyze Trends**.
  4. Read the LLM’s summary plus commentary on previous utterances.
"""
        )

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
                    label="Number of Similar Utterances",
                )

            with gr.Column(scale=4):
                gr.Markdown("### Analysis")
                utterance = gr.Textbox(
                    lines=10,
                    label="Enter User Utterance",
                    placeholder="Paste your user utterance here...",
                )
                analyze_btn = gr.Button("Analyze Trends", variant="primary")

                gr.Markdown("### Results")
                output = gr.Markdown()

        # Analyze button action
        analyze_btn.click(
            fn=analyzer.analyze_intent_trend, inputs=[utterance, top_k], outputs=output
        )

    return demo


if __name__ == "__main__":
    print("[main] Launching intent trend analyzer...")
    create_demo().launch()
