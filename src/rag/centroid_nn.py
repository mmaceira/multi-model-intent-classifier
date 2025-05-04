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
from scipy.special import softmax
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
        
    def predict_proba(self, docs: Sequence[str], **_):
        """
        Generate probability estimates for each class.
        
        This method computes similarity scores between document embeddings and 
        class centroids, then converts them to probability estimates using softmax.
        
        Parameters
        ----------
        docs : Sequence[str]
            The documents to classify.
            
        Returns
        -------
        np.ndarray
            An array of shape (n_samples, n_classes) containing probability estimates
            for each class.
        """
        # Get embeddings and normalize
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        # Prepare centroid matrix (ensuring consistent order with labels)
        labels = sorted(self.centroids.keys())
        cent_mat = np.stack([self.centroids[label] for label in labels])
        
        # Compute similarity scores
        similarity_scores = emb @ cent_mat.T
        
        # Convert similarity scores to probabilities using softmax
        # Apply temperature scaling factor to make the distribution less peaked
        temperature = 0.1  # Can be adjusted
        probabilities = softmax(similarity_scores / temperature, axis=1)
        
        return probabilities
