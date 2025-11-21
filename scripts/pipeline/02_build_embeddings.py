#!/usr/bin/env python
"""
Build Embeddings

This script generates embeddings using SBERT (Sentence-BERT) models and stores
the indices inside the experiment folder. OpenAI embeddings are optional and only
built if OPENAI_API_KEY is set. These embeddings are used for semantic search and
RAG-based classification on the CLINC150 intent classification dataset.

By default, only SBERT embeddings are built (no API keys required).
"""

import os
import sys
from pathlib import Path

import numpy as np

# Infer repo root from the location of this file
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))  # allow `import src.*`

# Import config setup
from sentence_transformers import SentenceTransformer

from config.notebook_setup import *

# Import dataset and embedding modules
from src.datasets.dataset import get_dataset
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.rag import _EMBEDDINGS_DIR, _OPENAI_DIR, _SBERT_DIR
from src.rag.vector_store import VectorStore

# Parameters
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "text-embedding-3-small"


def main():
    """Main function to build embeddings. SBERT is built by default, OpenAI is optional."""

    print("=" * 60)
    print("Build Embeddings")
    print("=" * 60)
    print("Note: SBERT embeddings are built by default (no API keys needed).")
    print("      OpenAI embeddings are optional and only built if OPENAI_API_KEY is set.")

    # Create directories
    print(f"\nEmbeddings root: {_EMBEDDINGS_DIR}")
    _SBERT_DIR.mkdir(parents=True, exist_ok=True)
    _OPENAI_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test, classes = get_dataset(
        dataset_name="clinc150",
        use_oos=config_vars.get("DATASET_USE_OOS", False),
        max_classes=config_vars.get("DATASET_MAX_CLASSES", None),
        max_train_samples=config_vars.get("DATASET_MAX_TRAIN_SAMPLES", None),
        max_test_samples=config_vars.get("DATASET_MAX_TEST_SAMPLES", None),
        seed=config_vars.get("GENERAL_SEED", 42),
    )

    print(f"Loaded {len(X_train)} training utterances with {len(classes)} intent classes")
    print(f"Test set contains {len(X_test)} utterances")

    # Build SBERT embeddings
    print("\n" + "=" * 60)
    print("Building SBERT Embeddings")
    print("=" * 60)
    print(f"Using model: {SBERT_MODEL}")
    print(f"Processing {len(X_train)} utterances...")

    try:
        sbert = SentenceTransformer(SBERT_MODEL)
        print("Encoding utterances...")
        vectors = sbert.encode(
            X_train, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")

        print(f"Generated embeddings with shape: {vectors.shape}")
        print("Building metadata...")
        meta = []
        for i, (txt, label, vec) in enumerate(zip(X_train, y_train, vectors, strict=False)):
            meta.append({"id": i, "label": label, "text": txt, "vector": vec.tolist()})

        print("Building FAISS index...")
        faiss_path = _SBERT_DIR / "index.faiss"
        meta_path = _SBERT_DIR / "meta.jsonl"
        VectorStore.build(vectors, meta, vectors.shape[1], faiss_path, meta_path)
        print(f"✅ SBERT index saved at {faiss_path}")
    except Exception as e:
        print(f"❌ Error building SBERT embeddings: {e}")
        raise

    # Build OpenAI embeddings (optional - only if API key is available)
    print("\n" + "=" * 60)
    print("Building OpenAI Embeddings (Optional)")
    print("=" * 60)

    if os.getenv("OPENAI_API_KEY"):
        print(f"Using model: {OPENAI_MODEL}")
        print("OPENAI_API_KEY found - building OpenAI embeddings...")
        try:
            openai_embedder = OpenAIEmbedder(model=OPENAI_MODEL, batch_size=50)
            openai_vecs = openai_embedder.encode(X_train)
            openai_vecs = np.array(openai_vecs, dtype="float32")

            meta_openai = []
            for i, (txt, label, vec) in enumerate(zip(X_train, y_train, openai_vecs, strict=False)):
                meta_openai.append({"id": i, "label": label, "text": txt, "vector": vec.tolist()})

            openai_faiss = _OPENAI_DIR / "index.faiss"
            openai_meta = _OPENAI_DIR / "meta.jsonl"
            VectorStore.build(
                openai_vecs, meta_openai, openai_vecs.shape[1], openai_faiss, openai_meta
            )
            print(f"✅ OpenAI index saved at {openai_faiss}")
        except Exception as e:
            print(f"⚠️  Failed to build OpenAI embeddings: {e}")
            print("   Continuing with SBERT embeddings only...")
    else:
        print("OPENAI_API_KEY not set - skipping OpenAI embeddings.")
        print("   (This is fine - SBERT embeddings are sufficient for most use cases)")

    print("\n✅ Embedding build complete!")
    print(f"Results saved to: {_EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
