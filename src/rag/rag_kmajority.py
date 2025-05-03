
"""k‑NN majority vote classifier."""
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
    def load_default(cls, top_k: int = 5):
        retriever = Retriever.from_default()
        labels = {m['label'] for m in retriever.store.meta}
        return cls(retriever, sorted(labels), top_k)

    def predict(self, docs: Sequence[str], **_):
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        preds = []
        for e in emb:
            neighbors = self.retriever.top_k(e[None, :], self.top_k)
            preds.append(self._majority_vote([n['label'] for n in neighbors]))
        return preds
