"""
RAG K-Majority Classifier Module

This module implements a K-Majority classifier for RAG (Retrieval-Augmented Generation)
systems. It uses a majority voting scheme on the k-nearest neighbors to make predictions.

Key Features:
- K-nearest neighbors retrieval
- Majority voting classification
- Probability estimation
- Support for multiple embedding types
- Scikit-learn compatible interface

Implementation Details:
1. Initialization:
   - Loads the vector store containing all document embeddings
   - Extracts unique labels from the metadata
   - Configures the number of nearest neighbors (k) to consider
   - Creates a label-to-index mapping for efficient probability computation

2. Prediction Process:
   - For each new document:
     1. Generates its embedding using MiniLM
     2. Retrieves the k most similar documents from the database
     3. Applies majority voting on the labels of these neighbors
     4. Returns the most frequent label as the prediction
   - For probability estimation:
     1. Counts occurrences of each label among the k neighbors
     2. Normalizes counts to get probability estimates
     3. Returns a probability distribution over all possible labels

3. Database Usage:
   - Unlike CentroidNN which uses summarized centroids, RagKMajority:
     - Performs actual nearest neighbor search for each query
     - Uses the full database during prediction
     - Considers the actual distribution of labels in the neighborhood

Advantages:
- More flexible than centroid-based approaches
- Can capture local patterns in the data
- Provides probability estimates based on actual neighbor distributions
- Better at handling documents that fall between class boundaries

Disadvantages:
- Slower than centroid-based approaches (requires nearest neighbor search)
- More memory intensive (needs to store and search through all embeddings)
- Performance depends on the choice of k (number of neighbors)

Classes:
- RagKMajority: K-Majority classifier for RAG systems

Functions:
- None (Class methods only)

Dependencies:
- numpy
- collections
- sentence_transformers
- typing

Example Usage:
    >>> # Initialize the classifier
    >>> classifier = RagKMajority.load_default(top_k=5)
    
    >>> # Make predictions
    >>> predictions = classifier.predict(documents)
    
    >>> # Get probability estimates
    >>> probabilities = classifier.predict_proba(documents)
"""

from typing import Sequence
import numpy as np
from collections import Counter
from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore

class RagKMajority(RagClassifierBase):
    def __init__(self, retriever: Retriever, labels: Sequence[str], top_k: int = 5):
        super().__init__(labels)
        self.retriever = retriever
        self.top_k = top_k
        self._label_to_idx = {label: i for i, label in enumerate(sorted(labels))}

    @classmethod
    def load_default(cls, top_k: int = 5, use_openai: bool = False):
        retriever = Retriever.from_default(use_openai=use_openai)
        labels = {m['label'] for m in retriever.store.meta}
        return cls(retriever, sorted(labels), top_k)

    def predict(self, docs: Sequence[str], **_):
        from sentence_transformers import SentenceTransformer
        
        # Use the static embed method from VectorStore to generate embeddings
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        
        # Process each embedding
        preds = []
        for e in emb:
            # Add a batch dimension for single query
            query_emb = np.expand_dims(e, axis=0)
            # Get nearest neighbors for this embedding
            neighbors = self.retriever.top_k(query_emb, self.top_k)
            # Apply majority vote to get prediction
            preds.append(self._majority_vote([n['label'] for n in neighbors]))
        
        return preds
    
    def predict_proba(self, docs: Sequence[str], **_):
        """
        Generate probability estimates for each class.
        
        This method transforms the k-nearest neighbors results into probability estimates
        by calculating the proportion of neighbors belonging to each class.
        
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
        from sentence_transformers import SentenceTransformer
        
        # Get embeddings
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        
        # Sort labels for consistency
        sorted_labels = sorted(self.labels)
        n_classes = len(sorted_labels)
        
        # Initialize probabilities array
        probas = np.zeros((len(docs), n_classes))
        
        # Process each embedding
        for i, e in enumerate(emb):
            # Add a batch dimension for single query
            query_emb = np.expand_dims(e, axis=0)
            # Get nearest neighbors
            neighbors = self.retriever.top_k(query_emb, self.top_k)
            # Count occurrences of each label
            label_counts = Counter([n['label'] for n in neighbors])
            
            # Convert counts to probabilities
            for j, label in enumerate(sorted_labels):
                probas[i, j] = label_counts.get(label, 0) / self.top_k
        
        return probas
