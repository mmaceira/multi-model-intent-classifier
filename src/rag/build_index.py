# NOTE: Embedding generation moved to notebook `0_build_embeddings.ipynb`.
# This script now only builds indices if precomputed embeddings are supplied.
# ----- Patched for OpenAI embedding support -----
from __future__ import annotations
import os
from pathlib import Path
import sys

# Add the parent directory to Python path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src.embeddings.openai_embedder import OpenAIEmbedder
except ImportError:
    # Fall back to relative import if absolute import fails
    from ..embeddings.openai_embedder import OpenAIEmbedder

"""
Build Index Module

This module provides functionality for building FAISS indices for the RAG
(Retrieval-Augmented Generation) system. It supports both SentenceTransformer
and OpenAI embeddings, with options for backward compatibility and parallel
index building.

Key Features:
- FAISS index building
- Support for multiple embedding types
- Backward compatibility options
- Parallel index building
- Embedding generation
- Metadata management

Classes:
- None (Module-level functions only)

Functions:
- _load_precomputed_embeddings: Load precomputed embeddings from file
- _load_csv: Load data from CSV file
- _load_reuters: Load Reuters dataset
- main: Main entry point for building indices
- build_openai_index: Build index using OpenAI embeddings
- load_embedder: Load appropriate embedder based on configuration

Dependencies:
- argparse
- csv
- json
- numpy
- tqdm
- pathlib
- os
- sys
- faiss
- sentence_transformers
- openai

Example Usage:
    >>> # Build indices using default settings
    >>> python build_index.py
    
    >>> # Build indices with custom settings
    >>> python build_index.py --use_openai --build_both --legacy_compat
"""

import argparse, csv, json, re
from typing import List
import numpy as np
from tqdm import tqdm

from .vector_store import VectorStore
from . import _ARTIFACTS_DIR, _EMBEDDINGS_DIR, _SBERT_DIR, _OPENAI_DIR


# ---------- Helper to load precomputed embeddings ----------
def _load_precomputed_embeddings(meta_path):
    import json, numpy as np
    if not meta_path.exists():
        raise FileNotFoundError(f"Precomputed embeddings not found at {meta_path}. "
                                "Run the notebook `0_build_embeddings.ipynb` first.")
    vectors=[]
    meta=[]
    with open(meta_path) as fh:
        for line in fh:
            rec=json.loads(line)
            vectors.append(rec['vector'])
            meta.append(rec)
    emb=np.array(vectors,dtype='float32')
    return emb, meta
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SOURCE = "reuters"
DEFAULT_CUTOFF = 1996
DEFAULT_N_CLASSES = 10

# Create the base artifacts and embeddings directories
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Legacy path - for backward compatibility
DEFAULT_LEGACY_FAISS_PATH = _ARTIFACTS_DIR / "index.faiss"
DEFAULT_LEGACY_META_PATH = _ARTIFACTS_DIR / "meta.jsonl"

# Create embedder-specific directories
_SBERT_DIR.mkdir(parents=True, exist_ok=True)
_OPENAI_DIR.mkdir(parents=True, exist_ok=True)

# New paths in separate directories
DEFAULT_SBERT_FAISS_PATH = _SBERT_DIR / "index.faiss"
DEFAULT_SBERT_META_PATH = _SBERT_DIR / "meta.jsonl"
DEFAULT_OPENAI_FAISS_PATH = _OPENAI_DIR / "index.faiss"
DEFAULT_OPENAI_META_PATH = _OPENAI_DIR / "meta.jsonl"

# Default to SentenceTransformer embeddings 
DEFAULT_FAISS_PATH = DEFAULT_SBERT_FAISS_PATH
DEFAULT_META_PATH = DEFAULT_SBERT_META_PATH



def _load_csv(csv_path: str) -> tuple[List[str], List[str], List[int]]:
    texts, labels, years = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
            years.append(int(row.get("year", 0)))
    return texts, labels, years


def _load_reuters(cutoff_year: int = DEFAULT_CUTOFF, n_classes: int = None) -> tuple[list, list, list]:
    from src.datasets.dataset import get_dataset
    X_train, y_train, _, _, _ = get_dataset(n_classes=n_classes)
    # Assume all docs are pre-cutoff
    return X_train, y_train, [cutoff_year - 1] * len(X_train)


def main():
    p = argparse.ArgumentParser(description="Build FAISS index for RAG")
    p.add_argument("--emb_model", default=DEFAULT_EMB_MODEL)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--train_csv")
    src.add_argument("--source", default=DEFAULT_SOURCE, choices=["reuters"])
    p.add_argument("--cutoff_year", type=int, default=DEFAULT_CUTOFF)
    p.add_argument("--n_classes", type=int, default=DEFAULT_N_CLASSES)
    p.add_argument("--faiss_path", default=str(DEFAULT_FAISS_PATH))
    p.add_argument("--meta_path", default=str(DEFAULT_META_PATH))
    p.add_argument("--use_openai", action="store_true", help="Use OpenAI embeddings")
    p.add_argument("--build_both", action="store_true", help="Build both SBERT and OpenAI indices")
    p.add_argument("--legacy_compat", action="store_true", help="Also save to legacy paths for compatibility")
    args = p.parse_args()

    if args.train_csv:
        texts, labels, years = _load_csv(args.train_csv)
    else:
        texts, labels, years = _load_reuters(args.cutoff_year, args.n_classes)

    # Set appropriate paths based on the specified model
    # The paths now refer to our new directory structure
    sbert_faiss_path = DEFAULT_SBERT_FAISS_PATH
    sbert_meta_path = DEFAULT_SBERT_META_PATH
    openai_faiss_path = DEFAULT_OPENAI_FAISS_PATH
    openai_meta_path = DEFAULT_OPENAI_META_PATH
    
    # Print the paths being used
    print(f"Using SBERT index path: {sbert_faiss_path}")
    print(f"Using SBERT meta path: {sbert_meta_path}")
    print(f"Using OpenAI index path: {openai_faiss_path}")
    print(f"Using OpenAI meta path: {openai_meta_path}")

    # Build indices as requested
    if args.use_openai or args.build_both:
        build_openai_index(texts, labels, years, openai_faiss_path, openai_meta_path)
    
    if not args.use_openai or args.build_both:
        # Original SentenceTransformer index
        print(f"Embedding {len(texts)} documents with {args.emb_model}…")
        emb = VectorStore.embed(args.emb_model, texts)

        meta: List[dict] = []
        for i, (t,l,y,v) in enumerate(zip(texts, labels, years, emb)):
            meta.append({"id": i, "label": l, "year": y, "text": t, "vector": v.tolist()})

        VectorStore.build(emb, meta, emb.shape[1], sbert_faiss_path, sbert_meta_path)
        print(f"✅ SBERT index saved → {sbert_faiss_path}\n✅ SBERT meta saved  → {sbert_meta_path}")
        
        # Also save to legacy paths if requested (for backward compatibility)
        if args.legacy_compat:
            DEFAULT_LEGACY_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
            VectorStore.build(emb, meta, emb.shape[1], DEFAULT_LEGACY_FAISS_PATH, DEFAULT_LEGACY_META_PATH)
            print(f"✅ Legacy index saved → {DEFAULT_LEGACY_FAISS_PATH}\n✅ Legacy meta saved  → {DEFAULT_LEGACY_META_PATH}")


def build_openai_index(texts, labels, years, faiss_path=DEFAULT_OPENAI_FAISS_PATH, meta_path=DEFAULT_OPENAI_META_PATH):
    """Build a FAISS index using OpenAI embeddings.
    
    Args:
        texts: List of document texts
        labels: List of document labels
        years: List of document years
        faiss_path: Path to save the FAISS index
        meta_path: Path to save the metadata
    """
    print(f"Embedding {len(texts)} documents with OpenAI embeddings…")
    
    # Ensure the directory exists
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI embedder
    openai_embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)
    
    # Generate embeddings
    raw_embeddings = openai_embedder.encode(texts)
    emb = np.array(raw_embeddings, dtype="float32")
    
    print(f"Generated {len(emb)} OpenAI embeddings with dimension {emb.shape[1]}")
    
    # Create metadata
    meta: List[dict] = []
    for i, (t,l,y,v) in enumerate(zip(texts, labels, years, emb)):
        meta.append({"id": i, "label": l, "year": y, "text": t, "vector": v.tolist()})
    
    # Build the index
    VectorStore.build(emb, meta, emb.shape[1], faiss_path, meta_path)
    print(f"✅ OpenAI index saved → {faiss_path}\n✅ OpenAI meta saved  → {meta_path}")


if __name__ == "__main__":
    main()


def load_embedder(model_name: str = None, use_openai: bool = False, batch_size: int = 100):
    """
    Load either a local SBERT model (CPU) or OpenAI remote embedder.

    Parameters
    ----------
    model_name
        Local transformer model name.
    use_openai
        When True, route to OpenAI. Can also be toggled with env var USE_OPENAI_EMBEDDINGS=1.
    batch_size
        Batch size for OpenAI requests.
    """
    if use_openai or os.getenv("USE_OPENAI_EMBEDDINGS") == "1":
        print("Using OpenAI embeddings endpoint...")
        return OpenAIEmbedder(batch_size=batch_size)
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name or "all-MiniLM-L6-v2")
    except ImportError as e:
        raise RuntimeError("SentenceTransformer not installed; install or set USE_OPENAI_EMBEDDINGS=1") from e
