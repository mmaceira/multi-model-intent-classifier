"""
RAG K-Majority Classifier Module

This module implements a K-Majority classifier for RAG (Retrieval-Augmented Generation)
systems. It uses a similarity-weighted majority voting scheme on the k-nearest neighbors to make predictions.

Key Features:
- K-nearest neighbors retrieval
- Similarity-weighted majority voting classification
- Probability estimation
- Support for multiple embedding types
- Scikit-learn compatible interface
- Efficient vector similarity search
- Flexible label handling

Implementation Details:
1. Initialization:
   - Loads the vector store containing all document embeddings
   - Extracts unique labels from the metadata
   - Configures the number of nearest neighbors (k) to consider from config
   - Creates a label-to-index mapping for efficient probability computation

2. Prediction Process:
   - For each new document:
     1. Generates its embedding using MiniLM
     2. Retrieves the k most similar documents from the database
     3. Applies similarity-weighted majority voting on the labels
     4. Returns the label with highest weighted score as the prediction
   - For probability estimation:
     1. Computes weighted scores for each label based on similarities
     2. Normalizes scores to get probability estimates
     3. Returns a probability distribution over all possible labels

3. Database Usage:
   - Unlike CentroidNN which uses summarized centroids, RagKMajority:
     - Performs actual nearest neighbor search for each query
     - Uses the full database during prediction
     - Considers both similarity and label distribution in the neighborhood

Advantages:
- More flexible than centroid-based approaches
- Can capture local patterns in the data
- Provides probability estimates based on weighted neighbor distributions
- Better at handling documents that fall between class boundaries
- Simple and interpretable classification method
- No training required

Disadvantages:
- Slower than centroid-based approaches (requires nearest neighbor search)
- More memory intensive (needs to store and search through all embeddings)
- Performance depends on the choice of k (number of neighbors)
- Sensitive to imbalanced class distributions

Classes:
    RagKMajority: K-Majority classifier for RAG systems
        - Implements k-nearest neighbors classification
        - Provides probability estimates
        - Supports both OpenAI and local embeddings

Functions:
    None (Class methods only)

Dependencies:
    numpy: For numerical operations
    collections: For efficient counting
    sentence_transformers: For embedding generation
    typing: For type hints

Example Usage:
    >>> # Initialize the classifier
    >>> classifier = RagKMajority.load_default()

    >>> # Make predictions
    >>> predictions = classifier.predict(documents)

    >>> # Get probability estimates
    >>> probabilities = classifier.predict_proba(documents)
"""

from collections import defaultdict
from typing import List, Sequence

import numpy as np

from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore


class RagKMajority(RagClassifierBase):
    """
    K-Majority classifier for RAG systems.

    This classifier implements a k-nearest neighbors approach with similarity-weighted
    majority voting for document classification. It uses vector similarity search to
    find the most similar documents and then applies weighted voting on their labels.

    Attributes:
        retriever (Retriever): Component for retrieving similar documents
        top_k (int): Number of nearest neighbors to consider
        _label_to_idx (Dict[str, int]): Mapping from labels to indices for probability computation
        embedding_model (str): Model name for generating embeddings
    """

    def __init__(
        self,
        retriever: Retriever,
        labels: Sequence[str],
        top_k: int = 5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the K-Majority classifier.

        Args:
            retriever (Retriever): The retrieval component for finding similar documents
            labels (Sequence[str]): List of possible classification labels
            top_k (int): Number of nearest neighbors to consider (default: 5)
            embedding_model (str): Model name for generating embeddings

        Example:
            >>> retriever = Retriever.from_default()
            >>> labels = ["business", "sports", "politics"]
            >>> classifier = RagKMajority(retriever, labels)
        """
        super().__init__(labels)
        self.retriever = retriever
        self.top_k = top_k
        self.embedding_model = embedding_model
        self._label_to_idx = {label: i for i, label in enumerate(sorted(labels))}

    @classmethod
    def load_default(
        cls,
        top_k: int = 5,
        use_openai: bool = False,
        embedding_model: str = None,
        config: dict = None,
    ) -> "RagKMajority":
        """
        Create a classifier with default configuration.

        This factory method simplifies creation by automatically loading the
        retriever and extracting labels from the metadata.

        Args:
            top_k (int): Number of nearest neighbors to consider (default: 5)
            use_openai (bool): Whether to use OpenAI embeddings (default: False)
            embedding_model (str): Optional model name for generating embeddings
            config (dict): Optional configuration dictionary

        Returns:
            RagKMajority: A configured classifier instance

        Example:
            >>> # Basic usage
            >>> classifier = RagKMajority.load_default()

            >>> # With custom parameters
            >>> classifier = RagKMajority.load_default(top_k=10, use_openai=True)
        """
        # If embedding_model is not provided, try to get it from config
        if embedding_model is None:
            # Default embedding models
            default_openai = "text-embedding-3-small"
            default_sbert = "sentence-transformers/all-MiniLM-L6-v2"

            if config is not None:
                # Use config values if available
                if use_openai:
                    embedding_model = config.get("model", {}).get(
                        "openai_model_name", default_openai
                    )
                else:
                    embedding_model = config.get("model", {}).get("sbert_model_name", default_sbert)
            else:
                # Use defaults if no config provided
                embedding_model = default_openai if use_openai else default_sbert

        retriever = Retriever.from_default(use_openai=use_openai)
        labels = {str(m["label"]) for m in retriever.store.meta}
        return cls(retriever, sorted(labels), top_k, embedding_model)

    def predict(self, docs: Sequence[str], **_) -> List[str]:
        """
        Predict labels for a sequence of documents using similarity-weighted voting.

        This method:
        1. Generates embeddings for the documents
        2. Retrieves k nearest neighbors for each document
        3. Applies similarity-weighted voting to determine the label

        Args:
            docs (Sequence[str]): Documents to classify
            **_: Additional arguments (ignored)

        Returns:
            List[str]: Predicted labels for each document

        Example:
            >>> classifier = RagKMajority.load_default()
            >>> docs = ["Apple's stock rose 2%", "Manchester United signed a new player"]
            >>> labels = classifier.predict(docs)
            >>> print(labels)
            ["business", "sports"]
        """
        # Use the static embed method from VectorStore to generate embeddings
        emb = VectorStore.embed(self.embedding_model, docs)

        # Process each embedding
        preds = []
        for e in emb:
            # Add a batch dimension for single query
            query_emb = np.expand_dims(e, axis=0)
            # Get nearest neighbors for this embedding
            neighbors = self.retriever.top_k(query_emb, self.top_k)

            # Apply similarity-weighted voting
            scores = defaultdict(float)
            for n in neighbors:
                scores[n["label"]] += n["score"]

            # Get label with highest weighted score
            preds.append(max(scores.items(), key=lambda kv: kv[1])[0])

        return preds

    def predict_proba(self, docs: Sequence[str], **_) -> np.ndarray:
        """
        Generate probability estimates for each class using similarity weights.

        This method transforms the k-nearest neighbors results into probability estimates
        by calculating the weighted proportion of neighbors belonging to each class.

        Args:
            docs (Sequence[str]): Documents to classify
            **_: Additional arguments (ignored)

        Returns:
            np.ndarray: Matrix of probability estimates with shape (n_docs, n_classes)

        Example:
            >>> classifier = RagKMajority.load_default()
            >>> docs = ["Apple's stock rose 2%", "Manchester United signed a new player"]
            >>> probas = classifier.predict_proba(docs)
            >>> print(probas.shape)
            (2, 10)  # Assuming 10 possible classes
        """
        # Get embeddings
        emb = VectorStore.embed(self.embedding_model, docs)

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

            # Compute weighted scores for each label
            label_scores = defaultdict(float)
            total_score = 0.0
            for n in neighbors:
                label_scores[n["label"]] += n["score"]
                total_score += n["score"]

            # Convert scores to probabilities
            for j, label in enumerate(sorted_labels):
                probas[i, j] = label_scores.get(label, 0) / total_score

        return probas
