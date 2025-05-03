
"""RAG package entry point with sensible defaults."""
from importlib import import_module
from pathlib import Path
from typing import Any

_DEFAULT_INDEX = (Path(__file__).resolve().parent.parent.parent / "artifacts" / "index.faiss")
_DEFAULT_META  = (Path(__file__).resolve().parent.parent.parent / "artifacts" / "meta.jsonl")

def _lazy_load(mod: str, cls: str, **kwargs: Any):
    module = import_module(f".{mod}", package=__name__)
    return getattr(module, cls).load_default(**kwargs)

def load_kmajority(**cfg): return _lazy_load("rag_kmajority", "RagKMajority", **cfg)
def load_centroid(**cfg):  return _lazy_load("centroid_nn", "CentroidNN", **cfg)
def load_llm(**cfg):       return _lazy_load("rag_llm", "RagLLM", **cfg)
