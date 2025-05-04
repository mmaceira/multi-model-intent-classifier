"""\
Centroid Nn module.

Classes:
- CentroidNN

Functions:
- None

Created: 2025-05-03
"""

from typing import Sequence
import numpy as np
from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore

class CentroidNN(RagClassifierBase):
    def __init__(self, centroids: dict[str, np.ndarray]):
        super().__init__(centroids.keys())
        self.centroids = {k: v/np.linalg.norm(v) for k, v in centroids.items()}

    @classmethod
    def load_default(cls, use_openai: bool = False, **kwargs):
        retriever = Retriever.from_default(use_openai=use_openai)
        by_lbl: dict[str, list[np.ndarray]] = {}
        for m in retriever.store.meta:
            by_lbl.setdefault(m['label'], []).append(np.array(m['vector']))
        cents = {k: np.mean(v, axis=0) for k, v in by_lbl.items()}
        return cls(cents)

    def predict(self, docs: Sequence[str], **_):
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        labels = list(self.centroids)
        cent_mat = np.stack(list(self.centroids.values()))
        sims = emb @ cent_mat.T
        return [labels[i] for i in sims.argmax(axis=1)]
