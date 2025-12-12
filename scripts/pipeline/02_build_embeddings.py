#!/usr/bin/env python
"""
Build Embeddings

This script generates embeddings using SBERT (Sentence-BERT) models and stores
the indices inside the experiment folder. OpenAI embeddings are optional and only
built if OPENAI_API_KEY is set. These embeddings are used for semantic search and
RAG-based classification.

By default, only SBERT embeddings are built (no API keys required).
"""

import os
import sys

import numpy as np

# Import path utilities
from intent_classifier.utils.paths import get_repo_root  # noqa: E402

# Get repo root and add to path for config imports (config is not part of the installed package)
repo_root = get_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import config setup
from sentence_transformers import SentenceTransformer  # noqa: E402

from config.notebook_setup import config_vars  # noqa: E402

# Import dataset and embedding modules
from intent_classifier.datasets.dataset import get_dataset  # noqa: E402
from intent_classifier.rag import _EMBEDDINGS_DIR, _OPENAI_DIR, _SBERT_DIR  # noqa: E402
from intent_classifier.rag.vector_store import VectorStore  # noqa: E402

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
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_dataset(
        dataset_name=config_vars.get("DATASET_NAME", "clinc150"),
        use_oos=config_vars.get("DATASET_USE_OOS", False),
        multilabel=config_vars.get("DATASET_MULTILABEL", False),
        max_classes=config_vars.get("DATASET_MAX_CLASSES", None),
        max_train_samples=config_vars.get("DATASET_MAX_TRAIN_SAMPLES", None),
        max_test_samples=config_vars.get("DATASET_MAX_TEST_SAMPLES", None),
        seed=config_vars.get("GENERAL_SEED", 42),
    )

    # Merge validation into training for embedding building
    # Note: For RAG-based models, we index all available training data (train+val)
    # to maximize the retrieval corpus. The training pipeline (03_model_training.py)
    # keeps validation separate for model training.
    X_train = X_train + X_val
    y_train = y_train + y_val

    print(
        f"Loaded {len(X_train)} training utterances "
        f"(train+val merged for indexing) with {len(classes)} intent classes"
    )
    print(f"Test set contains {len(X_test)} utterances")

    # Build SBERT embeddings
    print("\n" + "=" * 60)
    print("Building SBERT Embeddings")
    print("=" * 60)
    print(f"Using model: {SBERT_MODEL}")

    # Check if index already exists
    SBERT_INDEX = _SBERT_DIR / "index.faiss"
    SBERT_META = _SBERT_DIR / "meta.jsonl"
    FORCE_REBUILD = os.getenv("FORCE_REBUILD", "0").lower() in ("1", "true", "yes")

    if (SBERT_INDEX.exists() and SBERT_META.exists()) and not FORCE_REBUILD:
        print("⏭️  SBERT index already exists, skipping. Use FORCE_REBUILD=1 to rebuild.")
    else:
        print(f"Processing {len(X_train)} utterances...")

        try:
            sbert = SentenceTransformer(SBERT_MODEL)
            print("Encoding utterances...")
            vectors = sbert.encode(
                X_train,
                batch_size=int(os.getenv("SBERT_BATCH", "32")),
                show_progress_bar=True,
                convert_to_numpy=True,
            ).astype("float32")

            print(f"Generated embeddings with shape: {vectors.shape}")
            print("Building metadata...")
            meta = []
            for i, (txt, label) in enumerate(zip(X_train, y_train, strict=False)):
                meta.append({"id": i, "label": label, "text": txt})

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

    # Check if OpenAI index already exists
    OPENAI_INDEX = _OPENAI_DIR / "index.faiss"
    OPENAI_META = _OPENAI_DIR / "meta.jsonl"

    if os.getenv("OPENAI_API_KEY"):
        if (OPENAI_INDEX.exists() and OPENAI_META.exists()) and not FORCE_REBUILD:
            print("⏭️  OpenAI index already exists, skipping. Use FORCE_REBUILD=1 to rebuild.")
        else:
            print(f"Using model: {OPENAI_MODEL}")
            print("OPENAI_API_KEY found - building OpenAI embeddings...")
            try:
                # Use OpenAI API directly
                import openai

                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                print("Encoding utterances with OpenAI API...")
                openai_vecs = []
                batch_size = int(os.getenv("OPENAI_BATCH", "100"))
                for i in range(0, len(X_train), batch_size):
                    batch = X_train[i : i + batch_size]
                    response = client.embeddings.create(model=OPENAI_MODEL, input=batch)
                    batch_vecs = [item.embedding for item in response.data]
                    openai_vecs.extend(batch_vecs)
                    print(
                        f"  Processed {min(i+batch_size, len(X_train))}/{len(X_train)} utterances..."
                    )

                openai_vecs = np.array(openai_vecs, dtype="float32")
                print(f"Generated embeddings with shape: {openai_vecs.shape}")

                meta_openai = []
                for i, (txt, label) in enumerate(zip(X_train, y_train, strict=False)):
                    meta_openai.append({"id": i, "label": label, "text": txt})

                openai_faiss = _OPENAI_DIR / "index.faiss"
                openai_meta = _OPENAI_DIR / "meta.jsonl"
                VectorStore.build(
                    openai_vecs, meta_openai, openai_vecs.shape[1], openai_faiss, openai_meta
                )
                print(f"✅ OpenAI index saved at {openai_faiss}")
            except ImportError:
                print("⚠️  OpenAI package not installed - skipping OpenAI embeddings.")
                print("   Install with: pip install openai")
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
