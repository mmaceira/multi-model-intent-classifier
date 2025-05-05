"""
Retrieval Module

This module provides a retriever component for the RAG (Retrieval-Augmented Generation)
system. It handles the retrieval of similar documents from a vector store based on
query embeddings.

Key Features:
- Efficient nearest neighbor search
- Support for multiple embedding types
- Performance logging
- Default configuration loading

Classes:
- Retriever: Main class for document retrieval

Functions:
- None (Class methods only)

Dependencies:
- numpy
- logging
- time
- vector_store
- get_index_paths

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

import time
import logging
import numpy as np
from .vector_store import VectorStore
from . import get_index_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def top_k(self, q_emb: np.ndarray, k: int) -> list[dict]:
        start_time = time.time()
        logger.debug(f"Starting retrieval for top-{k} neighbors")
        
        results = self.store.search(q_emb, k)
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        logger.debug(f"Retrieved {len(results)} documents in {retrieval_time:.4f} seconds")
        
        return results

    @classmethod
    def from_default(cls, use_openai: bool = False):
        """Create a retriever with the default index and meta files.
        
        Args:
            use_openai: Whether to use OpenAI index files
            
        Returns:
            Retriever instance
        """
        logger.info(f"Loading {'OpenAI' if use_openai else 'SentenceTransformer'} retriever")
        start_time = time.time()
        
        # Get appropriate paths based on embedder type
        index_path, meta_path = get_index_paths(use_openai)
        logger.info(f"Using index: {index_path}, meta: {meta_path}")
        
        # Create retriever
        retriever = cls(VectorStore(index_path, meta_path))
        logger.info(f"Default retriever loaded in {time.time() - start_time:.2f} seconds")
        return retriever
