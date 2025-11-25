"""
Centroid Nearest Neighbor Classifier Module

This module implements a Centroid-based Nearest Neighbor classifier for RAG
(Retrieval-Augmented Generation) systems. It uses class centroids and cosine
similarity to make predictions.

Key Features:
- Centroid-based classification
- Cosine similarity computation
- Softmax probability estimation
- Support for multiple embedding types
- Scikit-learn compatible interface

Implementation Details:
1. Initialization:
   - Loads ALL documents from the database
   - Groups them by label
   - For each label, computes the mean vector (centroid) of ALL documents in that class
   - The centroids represent a summarized form of the full database

2. Prediction:
   - Embeds new documents using MiniLM
   - Normalizes the embeddings
   - Compares each document to the pre-computed centroids using cosine similarity
   - Assigns the label of the most similar centroid
   - No additional retrieval or filtering of the database is performed
   - Decisions are based solely on similarity to class centroids

Advantages:
- Fast prediction (only needs to compare to centroids)
- Memory efficient (stores only centroids, not full database)
- Simple and interpretable

Disadvantages:
- May miss subtle patterns not well represented by centroids
- Less flexible than approaches using full database or k-nearest neighbors

Classes:
- CentroidNN: Centroid-based classifier for RAG systems

Functions:
- None (Class methods only)

Dependencies:
- numpy
- scipy
- typing
- sentence_transformers

Example Usage:
    >>> # Initialize the classifier
    >>> classifier = CentroidNN.load_default()

    >>> # Make predictions
    >>> predictions = classifier.predict(documents)

    >>> # Get probability estimates
    >>> probabilities = classifier.predict_proba(documents)
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
        self.centroids = {k: v / np.linalg.norm(v) for k, v in centroids.items()}

    @classmethod
    def load_default(cls, use_openai: bool = False, **kwargs):
        retriever = Retriever.from_default(use_openai=use_openai)
        by_lbl: dict[str, list[np.ndarray]] = {}

        # Vectors are stored in FAISS index, not in metadata
        # Reconstruct vectors from FAISS index using metadata IDs
        index = retriever.store.index
        meta = retriever.store._meta

        for i, m in enumerate(meta):
            label = m.get("label", "")
            if not label:
                continue

            # Reconstruct vector from FAISS index
            # The index position corresponds to the metadata position
            try:
                vector = index.reconstruct(i)
                by_lbl.setdefault(label, []).append(np.array(vector, dtype=np.float32))
            except Exception:
                # Skip if vector reconstruction fails
                continue

        # Compute centroids for each label
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
