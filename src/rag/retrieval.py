
"""Retriever abstraction."""
import numpy as np
from .vector_store import VectorStore
from . import _DEFAULT_INDEX, _DEFAULT_META

class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def top_k(self, q_emb: np.ndarray, k: int) -> list[dict]:
        return self.store.search(q_emb, k)

    @classmethod
    def from_default(cls):
        return cls(VectorStore(_DEFAULT_INDEX, _DEFAULT_META))
