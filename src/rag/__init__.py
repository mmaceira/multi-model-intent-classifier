"""\
  Init   module.

Classes:
- None

Functions:
- _lazy_load
- load_kmajority
- load_centroid
- load_llm
- load_optimized_llm

Created: 2025-05-03
"""

from importlib import import_module
from pathlib import Path
import os

# Dynamic experiment root based on number of classes
_N_CLASSES = int(os.getenv('N_CLASSES', '10'))
_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / f"experiment_with_{_N_CLASSES}_classes/artifacts"

_SBERT_DIR = _ARTIFACTS_DIR / 'sbert'
_OPENAI_DIR = _ARTIFACTS_DIR / 'openai'
OPENAI_ARTIFACTS = _ARTIFACTS_DIR / 'openai'
SBERT_ARTIFACTS  = _ARTIFACTS_DIR / 'sbert'
from typing import Any, Optional

# New paths with embedder type in separate directories
_SBERT_INDEX = _SBERT_DIR / "index.faiss"
_SBERT_META  = _SBERT_DIR / "meta.jsonl"
_OPENAI_INDEX = _OPENAI_DIR / "index.faiss"
_OPENAI_META  = _OPENAI_DIR / "meta.jsonl"

# Default to SentenceTransformer paths
_DEFAULT_INDEX = _SBERT_INDEX
_DEFAULT_META = _SBERT_META

def get_index_paths(use_openai: bool = False) -> tuple[Path, Path]:
    """Get the appropriate index and meta paths based on embedder type.
    
    Args:
        use_openai: Whether to use OpenAI embeddings
        
    Returns:
        Tuple of (index_path, meta_path)
    """
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
            return get_index_paths(use_openai=False)
    
    # Use SBERT paths
    if _SBERT_INDEX.exists() and _SBERT_META.exists():
        return _SBERT_INDEX, _SBERT_META
    else:
        # Default to SBERT paths (which may not exist yet)
        return _SBERT_INDEX, _SBERT_META

def _lazy_load(mod: str, cls: str, **kwargs: Any):
    module = import_module(f".{mod}", package=__name__)
    return getattr(module, cls).load_default(**kwargs)

def load_kmajority(use_openai: bool = False, **cfg): 
    # Add use_openai to kwargs
    cfg['use_openai'] = use_openai
    return _lazy_load("rag_kmajority", "RagKMajority", **cfg)

def load_centroid(use_openai: bool = False, **cfg):
    # Add use_openai to kwargs  
    cfg['use_openai'] = use_openai
    return _lazy_load("centroid_nn", "CentroidNN", **cfg)

def load_llm(use_openai: bool = None, **cfg):
    # Determine use_openai from embedder if provided
    if use_openai is None and 'embedder' in cfg:
        # If embedder is an OpenAIEmbedder
        embedder_class_name = cfg['embedder'].__class__.__name__ if hasattr(cfg['embedder'], '__class__') else ''
        use_openai = embedder_class_name == 'OpenAIEmbedder'
    
    # Add use_openai to kwargs if it's not None
    if use_openai is not None:
        cfg['use_openai'] = use_openai
        
    return _lazy_load("rag_llm", "RagLLM", **cfg)

def load_optimized_llm(use_openai_embeddings: bool = False, **cfg):
    """
    Load an optimized RAG-LLM model with safer defaults:
    - Smaller batch size (1) to avoid hanging
    - Smaller top_k (3) for faster processing
    - Can specify whether to use OpenAI embeddings or not
    
    Args:
        use_openai_embeddings: Whether to use OpenAI embeddings (not index)
        **cfg: Additional config parameters for the RagLLM class
    
    Returns:
        A RagLLM instance with optimized settings
    """
    import os
    from .rag_llm import RagLLM
    from ..embeddings.openai_embedder import OpenAIEmbedder
    
    # Set environment variable for OpenAI embeddings
    if use_openai_embeddings:
        os.environ['USE_OPENAI_EMBEDDINGS'] = '1'
        embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=5)
    else:
        os.environ['USE_OPENAI_EMBEDDINGS'] = '0'
        embedder = None
    
    # Set defaults if not provided
    defaults = {
        'top_k': 3,
        'batch_size': 1,
        'model': 'gpt-4o-mini'
    }
    
    # Override defaults with provided values
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
    
    # Always use sentence-transformer index unless explicitly asked for OpenAI
    use_openai_index = cfg.pop('use_openai', False)
    
    # Create the RAG-LLM model
    model = load_llm(
        use_openai=use_openai_index,
        embedder=embedder,
        **cfg
    )
    
    return model
