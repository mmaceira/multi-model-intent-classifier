"""
Retrieval Module

This module provides a retriever component for the RAG (Retrieval-Augmented Generation)
system. It handles the retrieval of similar documents from a vector store based on
query embeddings.

Key Features:
- Efficient nearest neighbor search
- Support for multiple embedding types (OpenAI and SentenceTransformer)
- Performance logging and monitoring
- Default configuration loading
- Flexible vector store integration

Classes:
    Retriever: Main class for document retrieval
        - Handles document similarity search
        - Supports different embedding types
        - Provides performance monitoring
        - Offers default configuration setup

Functions:
    None (Class methods only)

Dependencies:
    numpy: For numerical operations
    logging: For performance monitoring
    time: For timing operations
    vector_store: For vector storage and search
    get_index_paths: For default path configuration

Example Usage:
    >>> # Create a default retriever
    >>> retriever = Retriever.from_default()

    >>> # Or create with custom vector store
    >>> from rag.vector_store import VectorStore
    >>> store = VectorStore("path/to/index", "path/to/meta")
    >>> retriever = Retriever(store)

    >>> # Retrieve similar documents
    >>> results = retriever.top_k(query_embedding, k=5)
"""

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import get_index_paths
from .vector_store import VectorStore

# Use module-level logger (no basicConfig - that's for entry points only)
logger = logging.getLogger(__name__)


class Retriever:
    """
    A class for retrieving similar documents from a vector store.

    This class provides functionality for efficient document retrieval based on
    vector similarity search. It supports different embedding types and includes
    performance monitoring capabilities.

    Attributes:
        store (VectorStore): The vector store instance used for document retrieval

    Methods:
        top_k: Retrieve the k most similar documents
        from_default: Create a retriever with default configuration
    """

    def __init__(self, store: VectorStore) -> None:
        """
        Initialize the retriever with a vector store.

        Args:
            store (VectorStore): The vector store instance to use for retrieval
        """
        self.store = store

    def top_k(self, q_emb: np.ndarray, k: int) -> List[Dict]:
        """
        Retrieve the k most similar documents for a given query embedding.

        Args:
            q_emb (np.ndarray): Query embedding vector
            k (int): Number of similar documents to retrieve

        Returns:
            List[Dict]: List of dictionaries containing retrieved documents and their metadata

        Example:
            >>> retriever = Retriever.from_default()
            >>> results = retriever.top_k(query_embedding, k=5)
            >>> for doc in results:
            ...     print(f"Document ID: {doc['id']}, Score: {doc['score']}")
        """
        start_time = time.time()
        logger.debug(f"Starting retrieval for top-{k} neighbors")

        results = self.store.search(q_emb, k)

        end_time = time.time()
        retrieval_time = end_time - start_time
        logger.debug(f"Retrieved {len(results)} documents in {retrieval_time:.4f} seconds")

        return results

    @classmethod
    def from_default(
        cls, use_openai: bool = False, embed_model: Optional[str] = None
    ) -> "Retriever":
        """
        Create a retriever with the default index and meta files.

        This method provides a convenient way to create a retriever with default
        configuration, supporting both OpenAI and SentenceTransformer embeddings.

        Args:
            use_openai (bool): Whether to use OpenAI index files (default: False)
            embed_model (Optional[str]): SBERT model name to use for embeddings.
                If None, reads from MODEL_SBERT_MODEL_NAME env var or defaults to
                "sentence-transformers/all-MiniLM-L6-v2". Only used when use_openai=False.

        Returns:
            Retriever: A configured retriever instance

        Example:
            >>> # Create a retriever with OpenAI embeddings
            >>> retriever = Retriever.from_default(use_openai=True)
            >>>
            >>> # Create a retriever with SentenceTransformer embeddings
            >>> retriever = Retriever.from_default(use_openai=False)
            >>>
            >>> # Create with custom embed model
            >>> retriever = Retriever.from_default(
            ...     use_openai=False,
            ...     embed_model="sentence-transformers/all-mpnet-base-v2"
            ... )
        """
        logger.info(f"Loading {'OpenAI' if use_openai else 'SentenceTransformer'} retriever")
        start_time = time.time()

        # Get appropriate paths based on embedder type
        index_path, meta_path = get_index_paths(use_openai)
        logger.info(f"Using index: {index_path}, meta: {meta_path}")

        # Get SBERT model name if not using OpenAI
        if not use_openai:
            if embed_model is None:
                # Try environment variable first, then fallback to default
                embed_model = os.getenv(
                    "MODEL_SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
                )

        # Create retriever with appropriate embedder config
        retriever = cls(VectorStore(index_path, meta_path, embed_model=embed_model))
        logger.info(f"Default retriever loaded in {time.time() - start_time:.2f} seconds")
        return retriever
