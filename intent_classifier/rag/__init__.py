"""
RAG Module Initialization

This module provides initialization and configuration for the RAG
(Retrieval-Augmented Generation) system. It handles path management, model loading,
and configuration for different RAG implementations.

Key Features:
- Path management for artifacts and embeddings
- Model loading utilities
- Configuration handling
- Support for multiple embedding types (OpenAI, SentenceTransformer)
- Optimized model loading

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
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union

# Module-level path attributes (initialized by _apply_paths)
# Initialize with placeholder values that will be overwritten
_EMBEDDINGS_DIR: Path = Path()
_ARTIFACTS_DIR: Path = Path()
_OPENAI_DIR: Path = Path()
_SBERT_DIR: Path = Path()
_OPENAI_INDEX: Path = Path()
_SBERT_INDEX: Path = Path()
_OPENAI_META: Path = Path()
_SBERT_META: Path = Path()
_DEFAULT_INDEX: Path = Path()
_DEFAULT_META: Path = Path()
OPENAI_ARTIFACTS: Path = Path()
SBERT_ARTIFACTS: Path = Path()


def _resolve_repo_root() -> Path:
    """Best-effort repository root discovery."""
    from intent_classifier.utils.paths import get_repo_root

    return get_repo_root()


def _resolve_default_embeddings_dir() -> Path:
    """Use env var if available, otherwise fall back to repo-local folder."""
    env_override = os.getenv("EMBEDDINGS_DIR") or os.getenv("RAG_EMBEDDINGS_DIR")
    if env_override:
        return Path(env_override)
    return _resolve_repo_root() / "embeddings"


def _apply_paths(base_embeddings: Path, base_artifacts: Path | None = None) -> None:
    """Update module-level path references."""
    global _EMBEDDINGS_DIR, _ARTIFACTS_DIR
    global _SBERT_DIR, _OPENAI_DIR, OPENAI_ARTIFACTS, SBERT_ARTIFACTS
    global _SBERT_INDEX, _SBERT_META, _OPENAI_INDEX, _OPENAI_META
    global _DEFAULT_INDEX, _DEFAULT_META

    _EMBEDDINGS_DIR = Path(base_embeddings)
    _ARTIFACTS_DIR = Path(base_artifacts) if base_artifacts else _EMBEDDINGS_DIR

    _SBERT_DIR = _EMBEDDINGS_DIR / "sbert"
    _OPENAI_DIR = _EMBEDDINGS_DIR / "openai"
    OPENAI_ARTIFACTS = _OPENAI_DIR
    SBERT_ARTIFACTS = _SBERT_DIR

    _SBERT_INDEX = _SBERT_DIR / "index.faiss"
    _SBERT_META = _SBERT_DIR / "meta.jsonl"
    _OPENAI_INDEX = _OPENAI_DIR / "index.faiss"
    _OPENAI_META = _OPENAI_DIR / "meta.jsonl"

    _DEFAULT_INDEX = _SBERT_INDEX
    _DEFAULT_META = _SBERT_META


_apply_paths(_resolve_default_embeddings_dir())


def set_artifacts_dir(artifacts_dir: str | Path, embeddings_dir: str | Path | None = None) -> None:
    """Set the artifacts/embeddings directory path and update derived paths."""
    base_artifacts = Path(artifacts_dir)
    base_embeddings = Path(embeddings_dir) if embeddings_dir is not None else base_artifacts
    _apply_paths(base_embeddings, base_artifacts)


@contextmanager
def _temporary_dirs(
    artifacts_dir: str | Path | None = None,
    embeddings_dir: str | Path | None = None,
):
    """Temporarily override the active directories."""
    if artifacts_dir is None and embeddings_dir is None:
        yield
        return

    old_artifacts, old_embeddings = _ARTIFACTS_DIR, _EMBEDDINGS_DIR
    set_artifacts_dir(artifacts_dir or old_artifacts, embeddings_dir or old_embeddings)
    try:
        yield
    finally:
        set_artifacts_dir(old_artifacts, old_embeddings)


def get_index_paths(
    use_openai: bool = False,
    artifacts_dir: str | Path | None = None,
    embeddings_dir: str | Path | None = None,
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
    with _temporary_dirs(artifacts_dir, embeddings_dir):
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


def load_kmajority(use_openai: bool = False, artifacts_dir: str | Path | None = None, **cfg):
    """Load K-Majority RAG model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A RagKMajority instance
    """
    with _temporary_dirs(artifacts_dir):
        cfg["use_openai"] = use_openai
        return _lazy_load("rag_kmajority", "RagKMajority", **cfg)


def load_centroid(use_openai: bool = False, artifacts_dir: str | Path | None = None, **cfg):
    """Load Centroid NN model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A CentroidNN instance
    """
    with _temporary_dirs(artifacts_dir):
        cfg["use_openai"] = use_openai
        return _lazy_load("centroid_nn", "CentroidNN", **cfg)


def load_llm(use_openai: bool | None = None, artifacts_dir: str | Path | None = None, **cfg):
    """Load RAG LLM model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters (can include log_dir for saving prompts/responses)

    Returns:
        A RagLLM instance
    """
    with _temporary_dirs(artifacts_dir):
        if use_openai is None and "embedder" in cfg:
            embedder_class_name = (
                cfg["embedder"].__class__.__name__ if hasattr(cfg["embedder"], "__class__") else ""
            )
            use_openai = embedder_class_name in ("OpenAIEmbedder", "EmbeddingGenerator")

        if use_openai is not None:
            cfg["use_openai"] = use_openai
            if use_openai and "embedder" not in cfg:
                os.environ["USE_OPENAI_EMBEDDINGS"] = "1"
                from intent_classifier.utils.embeddings import EmbeddingGenerator

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable must be set for OpenAI embeddings"
                    )
                cfg["embedder"] = EmbeddingGenerator(
                    api_key=api_key, model="text-embedding-3-small", batch_size=50
                )

        return _lazy_load("rag_llm", "RagLLM", **cfg)


def load_optimized_llm(
    use_openai_embeddings: bool = False, artifacts_dir: str | Path | None = None, **cfg
):
    """Load an optimized LLM model with optional OpenAI embeddings."""
    with _temporary_dirs(artifacts_dir):
        cfg["use_openai"] = use_openai_embeddings
        return _lazy_load("rag_llm", "RagLLM", **cfg)


def load_hybrid(use_openai: bool = False, artifacts_dir: str | Path | None = None, **cfg):
    """Load Hybrid RAG model

    Args:
        use_openai: Whether to use OpenAI embeddings
        artifacts_dir: Optional alternative artifacts directory
        **cfg: Additional config parameters

    Returns:
        A RagHybrid instance
    """
    with _temporary_dirs(artifacts_dir):
        cfg["use_openai"] = use_openai
        return _lazy_load("rag_hybrid", "RagHybrid", **cfg)


__all__ = [
    "set_artifacts_dir",
    "get_index_paths",
    "load_kmajority",
    "load_centroid",
    "load_llm",
    "load_optimized_llm",
    "load_hybrid",
]
