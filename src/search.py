"""\
Search module for semantic text search.

This module provides functionality for semantic search in text collections
using sentence embeddings. It includes functions for building embedding
indices and performing similarity-based search.

Functions:
- build_index: Create embedding index for documents
- semantic_search: Perform semantic search

Created: 2025-05-03
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

def build_index(texts: List[str], embedder: SentenceTransformer,
               batch_size: int = 32) -> np.ndarray:
    """Create embedding index for documents.
    
    This function generates embeddings for a collection of text documents
    using a pre-trained sentence transformer model. The embeddings are
    computed in batches for efficiency.
    
    Args:
        texts: List of text documents
        embedder: Pre-trained sentence transformer model
        batch_size: Batch size for embedding computation (default: 32)
        
    Returns:
        Numpy array of document embeddings
        
    Example:
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> embeddings = build_index(documents, model)
        >>> print(f"Embedding shape: {embeddings.shape}")
    """
    return embedder.encode(texts, batch_size=batch_size)

def semantic_search(query: str, texts: List[str], embedder: SentenceTransformer,
                   embeddings: np.ndarray = None, top_k: int = 5) -> List[Tuple[int, float]]:
    """Perform semantic search.
    
    This function performs semantic search by finding the most similar
    documents to a query using cosine similarity between embeddings.
    It can use pre-computed embeddings or compute them on the fly.
    
    Args:
        query: Search query text
        texts: List of text documents to search in
        embedder: Pre-trained sentence transformer model
        embeddings: Pre-computed document embeddings (default: None)
        top_k: Number of top results to return (default: 5)
        
    Returns:
        List of (index, similarity) tuples for top matches
        
    Example:
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> results = semantic_search("machine learning", documents, model)
        >>> for idx, score in results:
        ...     print(f"Score: {score:.3f} - {documents[idx][:100]}...")
    """
    # Compute query embedding
    query_embedding = embedder.encode([query])[0]
    
    # Use pre-computed embeddings or compute new ones
    if embeddings is None:
        doc_embeddings = embedder.encode(texts)
    else:
        doc_embeddings = embeddings
    
    # Compute cosine similarities
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(idx, float(similarities[idx])) for idx in top_indices]
