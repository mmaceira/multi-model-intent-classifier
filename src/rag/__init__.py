"""
RAG Module Initialization

This module provides initialization and configuration for the RAG (Retrieval-Augmented Generation)
system. It handles path management, model loading, and configuration for different RAG implementations.

Key Features:
- Path management for artifacts and embeddings
- Model loading utilities
- Configuration handling
- Support for multiple embedding types (OpenAI, SentenceTransformer)
- Optimized model loading

Classes:
- None (Module-level functions only)

Functions:
- set_artifacts_dir: Configure artifact and embedding directories
- get_index_paths: Get paths for index and metadata files
- _get_index_paths_internal: Internal implementation of path resolution
- _lazy_load: Lazy loading utility for modules
- load_kmajority: Load K-Majority RAG model
- load_centroid: Load Centroid NN model
- load_llm: Load RAG LLM model
- load_optimized_llm: Load optimized RAG-LLM model
- load_hybrid: Load RAG Hybrid model

Dependencies:
- importlib
- pathlib
- os
- sys
- typing
- config.notebook_setup

Example Usage:
    >>> # Set up directories
    >>> set_artifacts_dir("path/to/artifacts", "path/to/embeddings")

    >>> # Load a model
    >>> model = load_optimized_llm(
    ...     use_openai_embeddings=True,
    ...     model="gpt-4",
    ...     top_k=5
    ... )
"""

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union

# Import PATHS_EMBEDDINGS_DIR from notebook_setup
if "config.notebook_setup" in sys.modules:
    from config.notebook_setup import PATHS_EMBEDDINGS_DIR
else:
    # Explicitly import it if not already imported
    try:
        from config.notebook_setup import PATHS_EMBEDDINGS_DIR
    except ImportError as e:
        raise ImportError(
            "Cannot import PATHS_EMBEDDINGS_DIR from config.notebook_setup. "
            "Make sure to import notebook_setup before importing rag module."
        ) from e

# Set _EMBEDDINGS_DIR to PATHS_EMBEDDINGS_DIR
_EMBEDDINGS_DIR = Path(PATHS_EMBEDDINGS_DIR)

# Create subdirectories within the embeddings directory
_SBERT_DIR = _EMBEDDINGS_DIR / "sbert"
_OPENAI_DIR = _EMBEDDINGS_DIR / "openai"
OPENAI_ARTIFACTS = _EMBEDDINGS_DIR / "openai"
SBERT_ARTIFACTS = _EMBEDDINGS_DIR / "sbert"

# New paths with embedder type in separate directories
_SBERT_INDEX = _SBERT_DIR / "index.faiss"
_SBERT_META = _SBERT_DIR / "meta.jsonl"
_OPENAI_INDEX = _OPENAI_DIR / "index.faiss"
_OPENAI_META = _OPENAI_DIR / "meta.jsonl"

# Default to SentenceTransformer paths
_DEFAULT_INDEX = _SBERT_INDEX
_DEFAULT_META = _SBERT_META


def set_artifacts_dir(
    artifacts_dir: Union[str, Path], embeddings_dir: Optional[Union[str, Path]] = None
) -> None:
    """Set the artifacts directory path and update all derived paths.

    Args:
        artifacts_dir: Path to the artifacts directory
        embeddings_dir: Path to the embeddings directory (if None, uses artifacts_dir)
    """
    global _EMBEDDINGS_DIR, _SBERT_DIR, _OPENAI_DIR, OPENAI_ARTIFACTS, SBERT_ARTIFACTS
    global _SBERT_INDEX, _SBERT_META, _OPENAI_INDEX, _OPENAI_META, _DEFAULT_INDEX, _DEFAULT_META

    # Use provided embeddings_dir or default to using artifacts_dir
    if embeddings_dir is not None:
        _EMBEDDINGS_DIR = Path(embeddings_dir)
    else:
        _EMBEDDINGS_DIR = Path(artifacts_dir)

    # Update all derived paths
    _SBERT_DIR = _EMBEDDINGS_DIR / "sbert"
    _OPENAI_DIR = _EMBEDDINGS_DIR / "openai"
    OPENAI_ARTIFACTS = _EMBEDDINGS_DIR / "openai"
    SBERT_ARTIFACTS = _EMBEDDINGS_DIR / "sbert"

    _SBERT_INDEX = _SBERT_DIR / "index.faiss"
    _SBERT_META = _SBERT_DIR / "meta.jsonl"
    _OPENAI_INDEX = _OPENAI_DIR / "index.faiss"
    _OPENAI_META = _OPENAI_DIR / "meta.jsonl"

    # Update default paths
    _DEFAULT_INDEX = _SBERT_INDEX
    _DEFAULT_META = _SBERT_META


def get_index_paths(
    use_openai: bool = False,
    artifacts_dir: Optional[Union[str, Path]] = None,
    embeddings_dir: Optional[Union[str, Path]] = None,
) -> tuple[Path, Path]:
    """Get the appropriate index and meta paths based on embedder type.

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        embeddings_dir: Optional alternative embeddings directory

    Returns:
        Tuple of (index_path, meta_path)
    """
    # Use provided artifacts_dir and embeddings_dir if given
    if artifacts_dir is not None or embeddings_dir is not None:
        old_embeddings_dir = _EMBEDDINGS_DIR
        set_artifacts_dir(artifacts_dir or _EMBEDDINGS_DIR, embeddings_dir)
        result = _get_index_paths_internal(use_openai)
        set_artifacts_dir(_EMBEDDINGS_DIR, old_embeddings_dir)  # Restore original
        return result

    return _get_index_paths_internal(use_openai)


def _get_index_paths_internal(use_openai: bool) -> tuple[Path, Path]:
    """Internal implementation of get_index_paths without artifacts_dir handling"""
    # Ensure directories exist
    _SBERT_DIR.mkdir(parents=True, exist_ok=True)
    _OPENAI_DIR.mkdir(parents=True, exist_ok=True)

    if use_openai:
        # Check if OpenAI index exists, otherwise fall back to SBERT
        if _OPENAI_INDEX.exists() and _OPENAI_META.exists():
            return _OPENAI_INDEX, _OPENAI_META
        else:
            # If OpenAI files don't exist, use SBERT (after warning)
            import logging

            logging.warning(
                f"OpenAI index files not found at {_OPENAI_INDEX}. "
                f"Falling back to SentenceTransformer index at {_SBERT_INDEX}. "
                f"Run build_index.py with --use_openai to create OpenAI index files."
            )
            return _get_index_paths_internal(False)

    # Use SBERT paths
    if _SBERT_INDEX.exists() and _SBERT_META.exists():
        return _SBERT_INDEX, _SBERT_META
    else:
        # Default to SBERT paths (which may not exist yet)
        return _SBERT_INDEX, _SBERT_META


def _lazy_load(mod: str, cls: str, **kwargs: Any):
    module = import_module(f".{mod}", package=__name__)
    return getattr(module, cls).load_default(**kwargs)


def load_kmajority(
    use_openai: bool = False, artifacts_dir: Optional[Union[str, Path]] = None, **cfg
):
    """Load K-Majority RAG model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A RagKMajority instance
    """
    # Use provided artifacts_dir if given
    if artifacts_dir is not None:
        old_dir = _EMBEDDINGS_DIR
        set_artifacts_dir(artifacts_dir)

    # Add use_openai to kwargs
    cfg["use_openai"] = use_openai

    # Forward any embedding_model parameter if provided
    result = _lazy_load("rag_kmajority", "RagKMajority", **cfg)

    # Restore original artifacts_dir if changed
    if artifacts_dir is not None:
        set_artifacts_dir(old_dir)

    return result


def load_centroid(
    use_openai: bool = False, artifacts_dir: Optional[Union[str, Path]] = None, **cfg
):
    """Load Centroid NN model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A CentroidNN instance
    """
    # Use provided artifacts_dir if given
    if artifacts_dir is not None:
        old_dir = _EMBEDDINGS_DIR
        set_artifacts_dir(artifacts_dir)

    # Add use_openai to kwargs
    cfg["use_openai"] = use_openai
    result = _lazy_load("centroid_nn", "CentroidNN", **cfg)

    # Restore original artifacts_dir if changed
    if artifacts_dir is not None:
        set_artifacts_dir(old_dir)

    return result


def load_llm(use_openai: bool = None, artifacts_dir: Optional[Union[str, Path]] = None, **cfg):
    """Load RAG LLM model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A RagLLM instance
    """
    # Use provided artifacts_dir if given
    if artifacts_dir is not None:
        old_dir = _EMBEDDINGS_DIR
        set_artifacts_dir(artifacts_dir)

    # Determine use_openai from embedder if provided
    if use_openai is None and "embedder" in cfg:
        # If embedder is an OpenAIEmbedder
        embedder_class_name = (
            cfg["embedder"].__class__.__name__ if hasattr(cfg["embedder"], "__class__") else ""
        )
        use_openai = embedder_class_name == "OpenAIEmbedder"

    # Add use_openai to kwargs if it's not None
    if use_openai is not None:
        cfg["use_openai"] = use_openai
        if use_openai and "embedder" not in cfg:
            # Set environment variable for OpenAI embeddings
            os.environ["USE_OPENAI_EMBEDDINGS"] = "1"
            # Instantiate the OpenAI embedder
            from src.embeddings.openai_embedder import OpenAIEmbedder

            cfg["embedder"] = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)

    result = _lazy_load("rag_llm", "RagLLM", **cfg)

    # Restore original artifacts_dir if changed
    if artifacts_dir is not None:
        set_artifacts_dir(old_dir)

    return result


def load_optimized_llm(
    use_openai_embeddings: bool = False, artifacts_dir: Optional[Union[str, Path]] = None, **cfg
):
    """Load an optimized LLM model with optional OpenAI embeddings."""
    # Use provided artifacts_dir if given
    if artifacts_dir is not None:
        old_dir = _EMBEDDINGS_DIR
        set_artifacts_dir(artifacts_dir)

    # Add use_openai to kwargs
    cfg["use_openai"] = use_openai_embeddings
    result = _lazy_load("rag_llm", "RagLLM", **cfg)

    # Restore original artifacts_dir if changed
    if artifacts_dir is not None:
        set_artifacts_dir(old_dir)

    return result


def load_hybrid(use_openai: bool = False, artifacts_dir: Optional[Union[str, Path]] = None, **cfg):
    """Load Hybrid RAG model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A RagHybrid instance
    """
    # Use provided artifacts_dir if given
    if artifacts_dir is not None:
        old_dir = _EMBEDDINGS_DIR
        set_artifacts_dir(artifacts_dir)

    # Add use_openai to kwargs
    cfg["use_openai"] = use_openai
    result = _lazy_load("rag_hybrid", "RagHybrid", **cfg)

    # Restore original artifacts_dir if changed
    if artifacts_dir is not None:
        set_artifacts_dir(old_dir)

    return result
