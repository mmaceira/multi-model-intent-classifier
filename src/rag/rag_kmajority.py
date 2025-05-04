"""\
Rag Kmajority module.

Classes:
- RagKMajority

Functions:
- None

Created: 2025-05-03
"""

from typing import Sequence
from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore

class RagKMajority(RagClassifierBase):
    def __init__(self, retriever: Retriever, labels: Sequence[str], top_k: int = 5):
        super().__init__(labels)
        self.retriever = retriever
        self.top_k = top_k

    @classmethod
    def load_default(cls, top_k: int = 5, use_openai: bool = False):
        retriever = Retriever.from_default(use_openai=use_openai)
        labels = {m['label'] for m in retriever.store.meta}
        return cls(retriever, sorted(labels), top_k)

    def predict(self, docs: Sequence[str], **_):
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        preds = []
        for e in emb:
            neighbors = self.retriever.top_k(e[None, :], self.top_k)
            preds.append(self._majority_vote([n['label'] for n in neighbors]))
        return preds
